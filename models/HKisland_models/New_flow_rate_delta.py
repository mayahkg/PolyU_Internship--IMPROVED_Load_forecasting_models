import pandas as pd
import os
import glob

# Building names
building_name_list = ['CP1', 'CP4', 'CPN', 'CPS', 'DEH', 'DOH', 'OIE', 'OXH', 'LIH']

# Correct path to raw_data folder
raw_data_folder = 'models/HKisland_models/raw_data'

print(f"Looking for files in: {raw_data_folder}")
print(f"Folder exists: {os.path.exists(raw_data_folder)}")

if os.path.exists(raw_data_folder):
    print(f"Contents of {raw_data_folder}:")
    files_in_folder = os.listdir(raw_data_folder)
    print(f"  {files_in_folder}")

# Process each building CSV file from the correct location
for building in building_name_list:
    csv_file_path = os.path.join(raw_data_folder, f'{building}.csv')
    
    print(f"\nProcessing {building}.csv...")
    print(f"Looking for: {csv_file_path}")
    print(f"File exists: {os.path.exists(csv_file_path)}")
    
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            print(f"  Warning: {csv_file_path} not found, skipping...")
            continue
            
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"  - Loaded {len(df)} rows")
        print(f"  - Columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_columns = ['flowRate', 'returnTemp', 'supplyTemp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  Warning: Missing columns {missing_columns} in {building}.csv")
            print(f"  Available columns: {list(df.columns)}")
            continue
            
        # Calculate flow_rate_delta = flow_rate * (return_temp - supply_temp)
        df['flow_rate_delta'] = df['flowRate'] * (df['returnTemp'] - df['supplyTemp'])
        
        # Save the updated DataFrame back to CSV (overwriting the original)
        df.to_csv(csv_file_path, index=False)
        
        print(f"  ✓ Successfully updated {csv_file_path}")
        print(f"  - Added flow_rate_delta column")
        print(f"  - Value range: {df['flow_rate_delta'].min():.2f} to {df['flow_rate_delta'].max():.2f}")
        
    except Exception as e:
        print(f"  ✗ Error processing {csv_file_path}: {str(e)}")

print("\n" + "="*50)
print("PROCESSING COMPLETE!")
print("="*50)