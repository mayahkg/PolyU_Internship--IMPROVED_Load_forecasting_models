"""
test的数据暂时是第二年的夏天，此处应改用生成的数据test
"""
from config import *
from torch.utils.data import DataLoader
from models.HKisland_models.model_structures import *

import torch
import math
import pickle
import matplotlib.pyplot as plt


def cal_cv_rmse(pred, y_true):
    """
    liao 2021 Applied Energy 里面的metrics：CV-RMSE（公式5）
    0.328，0.352 这个level
    :param pred: vector
    :param y_true:
    :return:
    """
    import numpy as np
    pred = np.array(pred)
    y_true = np.array(y_true)
    if pred.shape != y_true.shape:
        raise ValueError("pred.shape != y_true.shape")
    return np.power(np.square(pred - y_true).sum() / pred.shape[0], 0.5) \
           / (y_true.sum() / pred.shape[0])


def draw_results(label_list, pred_list, model_name):
    mae_list = [abs(label_list[i] - pred_list[i]) for i in range(len(pred_list))]
    mae = sum(mae_list) / len(mae_list)
    rmse_list = [(label_list[i] - pred_list[i]) ** 2 for i in range(len(pred_list))]
    rmse = math.sqrt(sum(rmse_list) / len(rmse_list))
    cv_rmse = cal_cv_rmse(pred_list, label_list)

    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(111)
    plt.plot(range(len(pred_list)), pred_list, marker='8', color='orange', linewidth=1, label='predict')
    plt.plot(range(len(label_list)), label_list, color='b', linewidth=1, label='label')
    title = '{}, MAE={:.3f}, RMSE={:.3f}, CV-RMSE={:.3f}%'.format(model_name, mae, rmse, cv_rmse * 100)
    plt.title(title, loc='right')
    plt.legend()
    plt.show()


def test_seq2seq_with_attention(building_name):
    model_name = 'seq2seq_with_attention_{}'.format(building_name)

    # load data
    with open(r'tmp_pkl_data/{}_hkisland_save_dict.pkl'.format(building_name), 'rb') as r:
        save_dict = pickle.load(r)
    X = save_dict['test_X']
    Y = save_dict['test_Y']
    load_max = save_dict['load_max']
    load_min = save_dict['load_min']

    X = X[int(train_ratio * X.shape[0]):, :, :]
    Y = Y[int(train_ratio * Y.shape[0]):, :]

    # set test set and loader
    test_set = TrainSet(X, Y)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # set up model
    model = Seq2Seq_with_attention(input_dim=input_dim,
                                   output_dim=output_dim,
                                   enc_hid_dim=enc_hid_dim,
                                   dec_hid_dim=dec_hid_dim,
                                   dropout=dropout)
    save_path = r'models/trained_models/{}_24h.pt'.format(model_name)
    model.load_state_dict(torch.load(save_path))

    # start testing
    preds = []
    for data_x, _ in test_loader:
        data_x = data_x.to(torch.float32)

        pred = model(data_x)
        pred = pred.detach().numpy().squeeze()
        preds.append(pred)

    # denormalize
    _pred_list = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            _pred_list.append(preds[i][j])

    labels = test_set.label.tolist()
    _label_list = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            _label_list.append(labels[i][j])

    # note: pred may take more prediction results than label which is invalid
    _pred_list = _pred_list[:len(_label_list)]

    label_list = []
    pred_list = []
    for k in range(len(_pred_list)):
        label_list.append(_label_list[k] * (load_max - load_min) + load_min)
        pred_list.append(_pred_list[k] * (load_max - load_min) + load_min)

    draw_results(label_list, pred_list, model_name)
