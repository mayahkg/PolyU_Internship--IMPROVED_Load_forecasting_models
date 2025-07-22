Code Execution Sequence:

read_data.py --- This script is used to read building data and save it as .pkl files. These .pkl files will be stored in the tmp_pkl_data folder with the naming format: {building_name}_hkisland_save_dict.pkl

train_model.py --- This script trains 4 load forecasting models separately on the data from 9 buildings. The trained models will be saved in the models/trained_models folder with the naming format: {model_name}_{building_name}_24h.pt

test_model.py --- This script evaluates the performance of the trained models above. It generates three metrics (i.e., MAE, RMSE, CV-RMSE). Generally, a CV-RMSE below 30% is considered acceptable in the industry. It also plots a graph comparing predicted values versus actual values.

Tips:

The config file contains the list of building and model names, along with some parameters for load forecasting.