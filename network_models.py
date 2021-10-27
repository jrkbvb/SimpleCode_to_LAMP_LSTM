from torch import nn
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bi_directional=False, drop_out=0):
        super().__init__()
        if bi_directional==True:
        	D = 2
        elif bi_directional==False:
        	D = 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bi_directional, dropout=drop_out)
        self.linear = nn.Linear(hidden_size*D, output_size)
        #self.hidden_cell = (torch.zeros(num_layers,batch_size,hidden_size),
        #                    torch.zeros(num_layers,batch_size,hidden_size))
        #the above is the default when not specified. Leave it commented out.

    def forward(self, input_seq):
        # input_seq: batch_size, seq_length, input_size
        lstm_out, _ = self.lstm(input_seq)
        # lstm_out: batch_size, seq_length, hidden_size 
        predictions = self.linear(lstm_out)
        # predictions: batch_size, output_size
        return predictions
