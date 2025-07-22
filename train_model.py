from models.HKisland_models.lstm.train import train_lstm
from models.HKisland_models.seq2seq_with_attention.train import train_seq2seq_with_attention
from models.HKisland_models.sparse_ed.train import train_sparse_ed
from models.HKisland_models.sparse_lstm.train import train_sparse_lstm
from config import *


if __name__ == '__main__':
    # train all models
    # for building_name in building_name_list:
    #     for model_name in model_name_list:
    #         if model_name == 'lstm': train_lstm(building_name)
    #         elif model_name == 'seq2seq_with_attention': train_seq2seq_with_attention(building_name)
    #         elif model_name == 'sparse_ed': train_sparse_ed(building_name)
    #         elif model_name == 'sparse_lstm': train_sparse_lstm(building_name)

    # train specified set of models
    for building_name in building_name_list:
        for model_name in ['lstm']:
            if model_name == 'lstm': train_lstm(building_name)
            elif model_name == 'seq2seq_with_attention': train_seq2seq_with_attention(building_name)
            elif model_name == 'sparse_ed': train_sparse_ed(building_name)
            elif model_name == 'sparse_lstm': train_sparse_lstm(building_name)
