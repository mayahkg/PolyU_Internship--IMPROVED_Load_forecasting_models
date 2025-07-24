
from models.HKisland_models.lstm.train import train_lstm
from models.HKisland_models.lstm.test import test_lstm
from models.HKisland_models.seq2seq_with_attention.test import test_seq2seq_with_attention
from models.HKisland_models.sparse_ed.test import test_sparse_ed
from models.HKisland_models.sparse_lstm.test import test_sparse_lstm
from config import *
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    for building_name in building_name_list:
        for model_name in ['lstm']:  # Modify this list if you want to test other models
            os.makedirs(f'./results/{building_name}', exist_ok=True)
            if model_name == 'lstm':
                # Train the model to get training losses
                train_losses = train_lstm(building_name)
                # Test the model, passing training losses
                test_loss = test_lstm(building_name, train_losses)
                # Save test loss
                np.save(f'./results/{building_name}/lstm_test_loss.npy', np.array(test_loss))
            elif model_name == 'seq2seq_with_attention':
                test_seq2seq_with_attention(building_name)
            elif model_name == 'sparse_ed':
                test_sparse_ed(building_name)
            elif model_name == 'sparse_lstm':
                test_sparse_lstm(building_name)
