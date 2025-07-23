from models.HKisland_models.lstm.train import train_lstm
from models.HKisland_models.seq2seq_with_attention.train import train_seq2seq_with_attention
from models.HKisland_models.sparse_ed.train import train_sparse_ed
from models.HKisland_models.sparse_lstm.train import train_sparse_lstm
from config import *
import numpy as np
import os

if __name__ == '__main__':
    for building_name in building_name_list:
        for model_name in ['lstm']:
            os.makedirs(f'./results/{building_name}', exist_ok=True)
            if model_name == 'lstm':
                # Assume train_lstm returns a list or array of training losses per epoch
                train_loss = train_lstm(building_name)
                # Save training loss
                np.save(f'./results/{building_name}/lstm_train_loss.npy', np.array(train_loss))
            elif model_name == 'seq2seq_with_attention':
                train_seq2seq_with_attention(building_name)
            elif model_name == 'sparse_ed':
                train_sparse_ed(building_name)
            elif model_name == 'sparse_lstm':
                train_sparse_lstm(building_name)