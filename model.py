"""
Reference datasets:
https://github.com/sarthak268/Audio_Classification_using_LSTM
https://github.com/aniruddhapal211316/spoken_digit_recognition/blob/main/dataset.py
"""
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# Create Standard LSTM Model
class LSTM(nn.Module): 
	"""
    Parameters
    ----------
    x: Tensor (B x T_max x D)
        Batch tensor of padded MFCC sequences.
        
    x_lengths: Tensor (B)
        Tensor of lengths for each sequence in the batch.
    """
	def __init__(self, n_mfcc, n_label, h, d, n_lstm): 
		super().__init__()
		self.lstm_layer = nn.LSTM(input_size=n_mfcc, hidden_size=h, num_layers=n_lstm, batch_first=True, bidirectional=False)
		# Dropout layer
		self.lstm_layer_dropout = nn.Dropout()
		# Fully-connected classification layer
		self.linear_layer = nn.Linear(in_features=h, out_features=d)
		# ReLU Activation
		self.linear_layer_relu = nn.ReLU()
		# Dropout layer
		self.linear_layer_dropout = nn.Dropout()
		# (Pre-)Softmax layer for digit probabilities
		self.output_layer = nn.Linear(in_features=d, out_features=n_label)
		# Apply LogSoftmax for classification
		self.output_layer_logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x, lengths): 
		# Retrieve batch size
		batch_size = len(x)
		# Pack the padded Tensor into a PaddedSequence
		x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True)
		# Pass the PackedSequence through the bidirectional LSTM cell (x = final hidden state vectors)
		x, (hn, cn) = self.lstm_layer(x)
		hn = self.lstm_layer_dropout(hn)
		# Shape (x): n_layers*2 x B x h_dim

		# Reshape the final hidden state vectors
		hn = hn.transpose(1, 2).reshape(-1, batch_size).transpose(1, 0)
		# Shape: B x h_dim*2

		# Pass final hidden state vectors through the fully-connected ReLU layer
		hn = self.linear_layer_relu(self.linear_layer(hn))
		hn = self.linear_layer_dropout(hn)
		# Shape: B x fc_dim

		# Pass to log-softmax layer for classification
		return self.output_layer_logsoftmax(self.output_layer(hn))

# Create Bidirectional single-layer LSTM Model
class Bidirectional_LSTM(nn.Module): 

	def __init__(self, n_mfcc, n_label, h, d, n_lstm): 
		super().__init__()
		self.lstm_layer = nn.LSTM(input_size=n_mfcc, hidden_size=h, num_layers=n_lstm, batch_first=True, bidirectional=True)
		# Dropout layer
		self.lstm_layer_dropout = nn.Dropout()
		# Fully-connected classification layer
		self.linear_layer = nn.Linear(in_features=h*2, out_features=d)
		# ReLU Activation
		self.linear_layer_relu = nn.ReLU()
		# Dropout layer
		self.linear_layer_dropout = nn.Dropout()
		# (Pre-)Softmax layer for digit probabilities
		self.output_layer = nn.Linear(in_features=d, out_features=n_label)
		# Apply LogSoftmax for classification
		self.output_layer_logsoftmax = nn.LogSoftmax(dim=1)
		# Shape: B x 10

	def forward(self, x, lengths): 
		"""
        Parameters
        ----------
        x: Tensor (B x T_max x D)
            Batch tensor of padded MFCC sequences.
        
        x_lengths: Tensor (B)
            Tensor of lengths for each sequence in the batch.
        """
		# Retrieve batch size
		batch_size = len(x)
		# Pack the padded Tensor into a PaddedSequence
		x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True)
		# Pass the PackedSequence through the bidirectional LSTM cell (x = final hidden state vectors)
		x, (hn, cn) = self.lstm_layer(x)
		hn = self.lstm_layer_dropout(hn)
		# Shape (x): n_layers*2 x B x h_dim

		# Reshape the final hidden state vectors
		hn = hn.transpose(1, 2).reshape(-1, batch_size).transpose(1, 0)
		# Shape: B x h_dim*2

		# Pass final hidden state vectors through the fully-connected ReLU layer
		hn = self.linear_layer_relu(self.linear_layer(hn))
		hn = self.linear_layer_dropout(hn)
		# Shape: B x fc_dim

		# Pass to log-softmax layer for classification
		return self.output_layer_logsoftmax(self.output_layer(hn))
		# Shape: B x 10