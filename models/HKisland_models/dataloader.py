import os
from datetime import datetime
building_name_list = ['CP1', 'CP4', 'CPN', 'CPS', 'DEH', 'DOH', 'OIE', 'OXH', 'LIH']
import pickle
import numpy as np
import pandas as pd
import time

# Turning all classification to a number
def one_hot(data, feature):
    onehot = pd.get_dummies(data[feature], dtype=float)
    df = data.drop(feature, axis=1)
    data = df.join(onehot)
    return data

# Scaling the Cooling Load Values between 0 to 1
def normalize(df, temperature_min, temperature_max):
    flow_rate_delta_min = df['flow_rate_delta'].min()
    flow_rate_delta_max = df['flow_rate_delta'].max()
    df['nor_flow_rate_delta'] = 0
    df['nor_flow_rate_delta'] = (df['flow_rate_delta'] - flow_rate_delta_min) / (flow_rate_delta_max - flow_rate_delta_min)

    load_min = df['coolingLoad'].min()
    load_max = df['coolingLoad'].max()
    df['nor_cl'] = 0
    df['nor_cl'] = (df['coolingLoad'] - load_min) / (load_max - load_min)

    temperature_min = df['temperature'].min()
    temperature_max = df['temperature'].max()
    df['nor_temp'] = 0
    df['nor_temp'] = (df['temperature'] - temperature_min) / (temperature_max - temperature_min)

    return df, load_min, load_max

# Converting the data into an hourly database
def build_original_hk_island_load_table(df):
    begin_time = df.loc[0, 'time']
    end_time = df.loc[df.shape[0] - 1, 'time']
    should_total_hours = (datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S")).days * 24

    time_col = pd.date_range(begin_time, periods=should_total_hours, freq='1h')
    df2 = pd.DataFrame(columns=['time'])
    df2['time'] = time_col
    df['time'] = df['time'].astype('datetime64[ns]')
    df3 = pd.merge(df2, df, how='outer', on=['time'])

    df3.sort_values('time', inplace=True)
    df3.reset_index(drop=True, inplace=True)
    df3.fillna(0, inplace=True)
    return df3[['time', 'coolingLoad', 'temperature', 'flow_rate_delta']]

# Setting 0 values to the value from the previous day
def fill_value_into_null(df):
    df = df.copy()  # Avoid modifying the input DataFrame directly
    mask_cl = df['coolingLoad'] == 0
    df.loc[mask_cl, 'coolingLoad'] = df['coolingLoad'].shift(168)[mask_cl]
    mask_frd = df['flow_rate_delta'] == 0
    df.loc[mask_frd, 'flow_rate_delta'] = df['flow_rate_delta'].shift(168)[mask_frd]
    print(f"Filled {mask_cl.sum()} coolingLoad zeros and {mask_frd.sum()} flow_rate_delta zeros")
    return df

def make_big_df(path):
    # Check if input file exists
    if not os.path.exists(path):
        print(f"ERROR: Input file not found: {path}")
        return pd.DataFrame(columns=['time', 'coolingLoad', 'temperature', 'flow_rate_delta'])

    # Loading the Data
    try:
        data = pd.read_csv(path)
        print(f"Input data shape for {path}: {data.shape}")
        print(f"Input columns: {data.columns.tolist()}")
        # Check for flow_rate_delta or similar column names
        possible_flow_cols = [col for col in data.columns if 'flow_rate' in col.lower() or 'flowrate' in col.lower()]
        if 'flow_rate_delta' not in data.columns:
            if possible_flow_cols:
                print(f"WARNING: 'flow_rate_delta' not found, possible matches: {possible_flow_cols}")
                # Attempt to rename a likely match
                for col in possible_flow_cols:
                    if col.lower() in ['flowrate', 'flow_rate', 'flowratedelta', 'flow_rate_delta']:
                        print(f"Renaming {col} to 'flow_rate_delta'")
                        data = data.rename(columns={col: 'flow_rate_delta'})
                        break
            else:
                print(f"ERROR: 'flow_rate_delta' not found in input CSV: {path}")
        # Print flow_rate_delta statistics
        if 'flow_rate_delta' in data.columns:
            print(f"Input flow_rate_delta stats: non-zero={data['flow_rate_delta'].ne(0).sum()}, "
                  f"NaN={data['flow_rate_delta'].isna().sum()}, mean={data['flow_rate_delta'].mean()}")
    except Exception as e:
        print(f"ERROR: Failed to read {path}: {str(e)}")
        return pd.DataFrame(columns=['time', 'coolingLoad', 'temperature', 'flow_rate_delta'])

    # Compiling the Chillers into one
    print('chillerName:', data['chillerName'].unique())
    data = data.loc[data['cop'] > 0, :]
    print(f"Shape after filtering cop > 0: {data.shape}")
    if data.empty:
        print(f"ERROR: No data remains after filtering cop > 0 for {path}")
        return pd.DataFrame(columns=['time', 'coolingLoad', 'temperature', 'flow_rate_delta'])
    data.reset_index(drop=True, inplace=True)
    print(f"Data shape after reset: {data.shape}")
    count = data['time'].value_counts()
    print(f"Max run simultaneously: {count.max()}")

    # Sorting the rows according to their Chiller Names
    chillers_list = []
    for i in data['chillerName'].unique():
        data_i = data[data['chillerName'] == i]
        if data_i.shape[0] <= 0:
            print(f"WARNING: No data for chiller {i}")
            continue
        print(f"Chiller {i} data shape: {data_i.shape}")
        columns = data_i.columns.to_list()
        for j in range(2, len(columns)):
            columns[j] = columns[j] + '_' + str(i)
        data_i.columns = columns
        chillers_list.append(data_i)

    if not chillers_list:
        print(f"ERROR: No chiller data available for {path}")
        return pd.DataFrame(columns=['time', 'coolingLoad', 'temperature', 'flow_rate_delta'])

    # Combining all the chiller Lists into one mega list, sorted by time
    df_big = chillers_list[0]
    for i in range(len(chillers_list) - 1):
        df_big = pd.merge(df_big, chillers_list[i + 1], on=['time', 'building'], how='outer')

    df_big.sort_values('time', inplace=True)
    df_big.reset_index(drop=True, inplace=True)
    print(f"Shape after merge by time: {df_big.shape}")

    # Debug: Check all columns after merge
    print(f"All columns after merge: {df_big.columns.tolist()}")

    # Taking the sum of the cooling load and flow_rate_delta, and maximum temperature
    df_big['coolingLoad'] = 0
    load_columns_name_list = [item for item in df_big.columns.to_list() if 'coolingLoad' in item]
    print(f"Cooling load columns: {load_columns_name_list}")
    if load_columns_name_list:
        df_loads = df_big[load_columns_name_list]
        df_loads = df_loads.fillna(0)
        df_big['coolingLoad'] = df_loads.sum(axis=1)
    else:
        print(f"WARNING: No coolingLoad columns found for {path}")

    df_big['temperature'] = 0
    temp_columns_name_list = [item for item in df_big.columns.to_list() if 'temperature' in item]
    print(f"Temperature columns: {temp_columns_name_list}")
    if temp_columns_name_list:
        df_temp = df_big[temp_columns_name_list]
        df_temp = df_temp.fillna(0)
        df_big['temperature'] = df_temp.max(axis=1)
    else:
        print(f"WARNING: No temperature columns found for {path}")

    df_big['flow_rate_delta'] = 0
    flow_columns_name_list = [item for item in df_big.columns.to_list() if 'flow_rate_delta' in item]
    print(f"Flow rate delta columns: {flow_columns_name_list}")
    if flow_columns_name_list:
        df_flow = df_big[flow_columns_name_list]
        df_flow = df_flow.fillna(0)
        df_big['flow_rate_delta'] = df_flow.sum(axis=1)
        print(f"Flow rate delta sample values (first 5 rows): {df_big['flow_rate_delta'].head().tolist()}")
    else:
        print(f"ERROR: No flow_rate_delta columns found for {path}")

    # Verify columns before saving
    required_columns = ['time', 'coolingLoad', 'temperature', 'flow_rate_delta']
    missing_cols = [col for col in required_columns if col not in df_big.columns]
    if missing_cols:
        print(f"ERROR: Missing columns in df_big: {missing_cols}")
    print(f"Columns to be saved: {df_big[required_columns].columns.tolist()}")

    # Save to separate CSV file for this building
    df_save = df_big[required_columns].copy()
    building_name = os.path.basename(path).split('.')[0]
    output_dir = 'models/HKisland_models/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{building_name}.csv")
    df_save.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path} with columns: {df_save.columns.tolist()}")

    # Verify CSV was created and contains data
    if os.path.exists(csv_path):
        print(f"Confirmed: CSV file exists at {csv_path}")
        saved_df = pd.read_csv(csv_path)
        print(f"CSV row count: {len(saved_df)}")
        print(f"CSV flow_rate_delta stats: non-zero={saved_df['flow_rate_delta'].ne(0).sum()}, "
              f"NaN={saved_df['flow_rate_delta'].isna().sum()}, mean={saved_df['flow_rate_delta'].mean()}")
    else:
        print(f"ERROR: CSV file not created at {csv_path}")

    return df_big[required_columns].copy()

# Capturing Seasonal patterns for statistical analysis & Machine learning evaluation
def context_setting_weather_day(df):
    df['8_class'] = -1
    df['season'] = -1
    df['weekend_flag'] = -1

    seasons = {
        1: 'Winter',
        2: 'Spring',
        3: 'Summer',
        4: 'Autumn'
    }

    def judge_season(x):
        return (x.month % 12 + 3) // 3

    df['season'] = df['time'].map(judge_season)

    def judge_is_weekend_or_weekday(x):
        the_day = x.weekday()
        if the_day > 4:
            return 1
        else:
            return 0

    df['weekend_flag'] = df['time'].map(judge_is_weekend_or_weekday)

    def get_year_column(x):
        return x.year

    top_peak_num = 20
    df_temperature = df[['time', 'coolingLoad', 'temperature', 'season', 'weekend_flag']].copy()
    df_temperature['year'] = df_temperature['time'].map(get_year_column)
    for index in range(df_temperature.shape[0]):
        if df_temperature.loc[index, 'temperature'] == 0:
            df_temperature.loc[index, 'temperature'] = df_temperature.loc[index - 1, 'temperature']
    df_temperature['hour'] = df_temperature['time'].dt.hour
    df_temperature_12clock = df_temperature[df_temperature['hour'] == 12]

    the_peak_date_winter_list = []
    for year in pd.unique(df_temperature['year']):
        df_temperature_12clock_this_year = df_temperature_12clock[df_temperature_12clock['year'] == year]
        df_winter = df_temperature_12clock_this_year[df_temperature_12clock_this_year['season'] == 1].reset_index(drop=True)
        top_k_idx = np.array(df_winter['temperature']).argsort()[::-1][0: top_peak_num]
        the_peak_date_winter = df_winter.loc[top_k_idx, 'time']
        for ii in the_peak_date_winter.index:
            the_peak_date_winter_list.append(str(the_peak_date_winter.loc[ii]).split(' ')[0])

    the_peak_date_summer_list = []
    for year in pd.unique(df_temperature['year']):
        df_temperature_12clock_this_year = df_temperature_12clock[df_temperature_12clock['year'] == year]
        df_summer = df_temperature_12clock_this_year[df_temperature_12clock_this_year['season'] == 3].reset_index(drop=True)
        top_k_idx = np.array(df_summer['temperature']).argsort()[::-1][0: top_peak_num]
        the_peak_date_summer = df_summer.loc[top_k_idx, 'time']
        for ii in the_peak_date_summer.index:
            the_peak_date_summer_list.append(str(the_peak_date_summer.loc[ii]).split(' ')[0])

    print('build contextual column for df...')
    for index in range(df.shape[0]):
        if df.loc[index, 'season'] == 1:
            if df.loc[index, 'weekend_flag'] == 1:
                df.loc[index, '8_class'] = 2
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 1
                if str(df.loc[index, 'time']).split(' ')[0] in the_peak_date_winter_list:
                    df.loc[index, '8_class'] = 0
        elif df.loc[index, 'season'] == 3:
            if df.loc[index, 'weekend_flag'] == 1:
                df.loc[index, '8_class'] = 5
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 4
                if str(df.loc[index, 'time']).split(' ')[0] in the_peak_date_summer_list:
                    df.loc[index, '8_class'] = 3
        elif df.loc[index, 'season'] == 2:
            df.loc[index, '8_class'] = 6
        elif df.loc[index, 'season'] == 4:
            df.loc[index, '8_class'] = 7
        else:
            raise ValueError(df.loc[index, 'time'] + '找不到`8分类`')

    return df

# Splitting the Processed data into training & testing
def create_test_X_Y(df, seq_length=24):
    X = []
    Y = []
    for i in range(df.shape[0] // seq_length - 1):
        X.append(np.array(df.iloc[i * seq_length: (i + 1) * seq_length, 6:]))
        Y.append(np.array(df.loc[(i + 1) * seq_length: (i + 2) * seq_length - 1, 'nor_cl']))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def create_train_X_Y(df, seq_length=24):
    X = []
    Y = []
    for i in range(df.shape[0] - 2 * seq_length):
        X.append(np.array(df.iloc[i: i + seq_length, 6:]))
        Y.append(np.array(df.loc[i + seq_length: i + 2 * seq_length - 1, 'nor_cl']))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def get_timeseries_units(df, seq_length=24):
    X = []
    for i in range(df.shape[0] // seq_length - 1):
        X.append(np.array(df.loc[(i + 1) * seq_length: (i + 2) * seq_length - 1, 'coolingLoad']))
    X = np.array(X)
    return X

def read_data_for_hkisland_model(building_name, temperature_min, temperature_max):
    # load data
    csv_path = 'models/HKisland_models/raw_data/'
    building_csv_path = os.path.join(csv_path, f"{building_name}.csv")
    print(f"csv path: {building_csv_path}")
    df = make_big_df(building_csv_path)

    if df.empty:
        print(f"ERROR: No data processed for {building_name}")
        return

    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # fill in missing val
    print(f"Shape before build_original_hk_island_load_table: {df.shape}")
    df = build_original_hk_island_load_table(df=df)
    df = fill_value_into_null(df=df)
    print(f"Shape after processing: {df.shape}")

    # config context
    df = context_setting_weather_day(df=df)

    # normalize
    df, load_min, load_max = normalize(df=df, temperature_min=temperature_min, temperature_max=temperature_max)

    # one_hot time feature and weekday feature
    time_feature = []
    weekday_feature = []
    for i in range(df.shape[0]):
        time_feature_i = str(df.loc[i, 'time']).split(' ')[-1]
        weekday_feature_i = time.strptime(str(df.loc[i, 'time']), "%Y-%m-%d %H:%M:%S").tm_wday
        time_feature.append(time_feature_i)
        weekday_feature.append(weekday_feature_i)
    df['time_feature'] = np.array(time_feature)
    df['weekday_feature'] = np.array(weekday_feature)
    df = one_hot(df, 'time_feature')
    df = one_hot(df, 'weekday_feature')

    # divide daily unit from 18PM
    start_index = 0
    while True:
        if str(df.loc[start_index, 'time']).endswith('18:00:00'):
            break
        start_index += 1
    df_new = df.loc[start_index:, :]
    df_new.reset_index(drop=True, inplace=True)

    load_trace_units = get_timeseries_units(df=df_new)
    test_X, test_Y = create_test_X_Y(df=df_new)
    train_X, train_Y = create_train_X_Y(df=df_new)

    save_dict = {
        'df': df_new,
        'train_X': train_X,
        'train_Y': train_Y,
        'test_X': test_X,
        'test_Y': test_Y,
        'load_trace_units': load_trace_units,
        'load_max': load_max,
        'load_min': load_min,
    }

    # save pickle file
    output_dir = 'models/HKisland_models/tmp_pkl_data'
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{building_name}_hkisland_save_dict.pkl")
    with open(pkl_path, 'wb') as w:
        pickle.dump(save_dict, w)
    print(f"Model saved: {pkl_path}")
    if os.path.exists(pkl_path):
        print(f"Confirmed: Pickle file exists at {pkl_path}")
    else:
        print(f"ERROR: Pickle file not created at {pkl_path}")

if __name__ == '__main__':
    # Remove any existing HKisland_models.csv to avoid confusion
    combined_csv = 'models/HKisland_models/processed_data/HKisland_models.csv'
    if os.path.exists(combined_csv):
        print(f"Removing existing combined file: {combined_csv}")
        os.remove(combined_csv)

    for building_name in building_name_list:
        print(f"Processing building: {building_name}")
        read_data_for_hkisland_model(building_name=building_name,
                                     temperature_min=0,
                                     temperature_max=40)
        print(f"Completed processing for {building_name}\n")