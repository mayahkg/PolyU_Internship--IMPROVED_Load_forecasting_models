building_name_list = ['CP1', 'CP4', 'CPN', 'CPS', 'DEH', 'DOH', 'OIE', 'OXH', 'LIH']
model_name_list = ['lstm', 'seq2seq_with_attention', 'sparse_ed', 'sparse_lstm']

# note: load forecasting parameters
temperature_max = 40
temperature_min = 0

seq_length = 24 # input length of sequence, can try change the sequence to, what you are predicting
input_dim = 35 # number of features

# parameters of the model, adjust with features
sparse_output_dim = 64
hidden_dim = 128
output_dim = 1
num_layers = 1

enc_hid_dim = 128
dec_hid_dim = 128
dropout = 0.5

sparse_lstm_lambda = 1e-3
sparse_ed_lambda = 1e-5

# split train and test
train_ratio = 0.5

# how many sequences do you use
batch_size = 64

# medium rate
learning_rate = 0.001

# time of iteration
num_epochs = 10


