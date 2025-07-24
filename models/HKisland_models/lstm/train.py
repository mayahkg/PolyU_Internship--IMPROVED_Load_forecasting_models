
from config import *
from torch.utils.data import DataLoader
from models.HKisland_models.model_structures import *

import pickle
import matplotlib.pyplot as plt


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

    # set up model
    model = LSTM(input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 num_layers=num_layers)
    criterion = nn.L1Loss()
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # list to store average loss per epoch
    epoch_losses = []

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

        avg_loss = l_sum / len(train_loader)
        epoch_losses.append(avg_loss)
        print('Epoch {}: loss = {}'.format(epoch + 1, avg_loss))

    # save model
    save_path = r'models/trained_models/{}_24h.pt'.format(model_name)
    torch.save(model.state_dict(), save_path)
    print('model saved: {}'.format(save_path))
    print()

    # return epoch losses for plotting
    return epoch_losses


def plot_train_test_losses(train_losses, test_losses, building_name):
    """
    Plot training and testing losses versus epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    # If test_losses is a single value, repeat it for the number of epochs
    if len(test_losses) == 1:
        test_losses = [test_losses[0]] * len(train_losses)
    plt.plot(range(1, len(train_losses) + 1), test_losses, label='Testing Loss', color='red')
    plt.title(f'Training and Testing Loss vs Epoch for {building_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()