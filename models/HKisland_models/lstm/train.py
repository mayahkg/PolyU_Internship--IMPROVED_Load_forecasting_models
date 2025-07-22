from config import *
from torch.utils.data import DataLoader
from models.HKisland_models.model_structures import *

import pickle


def train_lstm(building_name):
    model_name = 'lstm_{}'.format(building_name)
    # load data
    with open(r'tmp_pkl_data/{}_hkisland_save_dict.pkl'.format(building_name), 'rb') as r:
        save_dict = pickle.load(r)
    X = save_dict['train_X']
    Y = save_dict['train_Y']

    X = X[:int(train_ratio * X.shape[0]), :, :]
    Y = Y[:int(train_ratio * Y.shape[0]), :]

    # set train set and dataloader
    train_set = TrainSet(X, Y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # set up trained_models
    model = LSTM(input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 num_layers=num_layers)
    criterion = nn.L1Loss()
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_epochs):
        l_sum = 0.
        for data_x, data_y in train_loader:
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)

            pred = model(data_x)
            loss = criterion(pred, data_y)

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            l_sum += loss.item()

        print('Epoch {}: loss = {}'.format(epoch + 1, l_sum / len(train_loader)))

    # save model
    save_path = r'models/trained_models/{}_24h.pt'.format(model_name)
    torch.save(model.state_dict(), save_path)
    print('model saved: {}'.format(save_path))
    print()


