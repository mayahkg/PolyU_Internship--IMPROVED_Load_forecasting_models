import numpy as np
import os
import torch
import torch.nn as nn
from models.HKisland_models.lstm.train import train_lstm
from models.HKisland_models.seq2seq_with_attention.train import train_seq2seq_with_attention
from models.HKisland_models.sparse_ed.train import train_sparse_ed
from models.HKisland_models.sparse_lstm.train import train_sparse_lstm
from config import building_name_list

def debug_train_lstm(building_name):
    """
    Debug wrapper for train_lstm to catch NaN and errors.
    Add these checks to your actual train_lstm implementation.
    """
    print(f"Debugging train_lstm for {building_name}")
    try:
        # Call the actual train_lstm function
        train_loss = train_lstm(building_name)
        
        # Validate output
        if train_loss is None:
            print(f"train_lstm returned None for {building_name}")
            return None
        if isinstance(train_loss, (list, np.ndarray)):
            train_loss = np.array(train_loss)
            if np.any(np.isnan(train_loss)) or np.any(np.isinf(train_loss)):
                print(f"NaN or Inf detected in train_loss for {building_name}")
                return train_loss
        else:
            print(f"Unexpected train_loss type: {type(train_loss)} for {building_name}")
            return None
        return train_loss
    except Exception as e:
        print(f"Exception in train_lstm for {building_name}: {str(e)}")
        return None

# Main training loop with enhanced debugging
if __name__ == '__main__':
    print(f"Building names: {building_name_list}")
    
    for building_name in building_name_list:
        print(f"\nStarting training for {building_name}")
        for model_name in ['lstm']:  # Only LSTM as per your focus
            result_dir = f'./results/{building_name}'
            os.makedirs(result_dir, exist_ok=True)
            try:
                if model_name == 'lstm':
                    train_loss = debug_train_lstm(building_name)
                    if train_loss is None:
                        print(f"Training failed (None returned) for {building_name} with {model_name}")
                        continue
                    if np.any(np.isnan(train_loss)) or np.any(np.isinf(train_loss)):
                        print(f"NaN or Inf detected in losses for {building_name} with {model_name}")
                        continue
                    # Save losses
                    save_path = f'{result_dir}/lstm_train_loss.npy'
                    np.save(save_path, np.array(train_loss))
                    print(f"Saved training loss to {save_path}")
                elif model_name == 'seq2seq_with_attention':
                    train_seq2seq_with_attention(building_name)
                elif model_name == 'sparse_ed':
                    train_sparse_ed(building_name)
                elif model_name == 'sparse_lstm':
                    train_sparse_lstm(building_name)
            except Exception as e:
                print(f"Error training {building_name} with {model_name}: {str(e)}")
                continue