from models.HKisland_models.lstm.test import test_lstm
from models.HKisland_models.seq2seq_with_attention.test import test_seq2seq_with_attention
from models.HKisland_models.sparse_ed.test import test_sparse_ed
from models.HKisland_models.sparse_lstm.test import test_sparse_lstm
from config import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    # test all models
    # for building_name in building_name_list:
    #     for model_name in model_name_list:
    #         if model_name == 'lstm': test_lstm(building_name)
    #         elif model_name == 'seq2seq_with_attention': test_seq2seq_with_attention(building_name)
    #         elif model_name == 'sparse_ed': test_sparse_ed(building_name)
    #         elif model_name == 'sparse_lstm': test_sparse_lstm(building_name)

    # test specified set of models
    for building_name in building_name_list:
        for model_name in ['lstm']:
            if model_name == 'lstm': test_lstm(building_name)
            elif model_name == 'seq2seq_with_attention': test_seq2seq_with_attention(building_name)
            elif model_name == 'sparse_ed': test_sparse_ed(building_name)
            elif model_name == 'sparse_lstm': test_sparse_lstm(building_name)
