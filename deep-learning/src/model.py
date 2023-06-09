import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import config
import torch.nn.functional as F

class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        batch_size = x.size(0)
        # embeddings and lstm_out
        x = x.long()
        # initial hidden states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(config.DEVICE)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(config.DEVICE)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, (h0,c0))

        # # stack up lstm outputs
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        lstm_out, _ = torch.max(lstm_out,1)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

class SentimentBiLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        super(SentimentBiLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim*2, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        batch_size = x.size(0)
        # embeddings and lstm_out
        x = x.long()
        # initial hidden states
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(config.DEVICE)
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(config.DEVICE)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, (h0,c0))

        # # stack up lstm outputs
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        lstm_out, _ = torch.max(lstm_out,1)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


# Define the model
class SentimentCNN(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_outputs):
        super(SentimentCNN, self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.K = n_outputs

        self.sig = nn.Sigmoid()

        # if input is T words
        # then output is (T, D) matrix
        self.embed = nn.Embedding(self.V, self.D)

        # conv layers
        self.conv1 = nn.Conv1d(self.D, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)

        self.fc = nn.Linear(128, self.K)


    def forward(self, X):
        # embedding layer
        # turns word indexes into word vectors

        X = X.long()
        out = self.embed(X)

        # note: output of embedding is always
        # (N, T, D)
        # conv1d expects
        # (N, D, T)

        # conv layers
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = F.relu(out)

        # change it back
        out = out.permute(0, 2, 1)

        # max pool
        out, _ = torch.max(out, 1)

        # final dense layer
        out = self.fc(out)
        sig_out = self.sig(out)
        # sig_out = sig_out.view(config.BATCH_SIZE, -1)
        # sig_out = sig_out[:, -1]

        return sig_out

class SentimentCNNLSTM(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_outputs):
        super(SentimentCNNLSTM, self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.K = n_outputs

        self.sig = nn.Sigmoid()

        # if input is T words
        # then output is (T, D) matrix
        self.embed = nn.Embedding(self.V, self.D)

        # conv layers
        self.conv1 = nn.Conv1d(self.D, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)

        # self.fc = nn.Linear(128, self.K)

        self.lstm = nn.LSTM(128, config.HIDDEN_DIM, config.N_LAYERS,
                            dropout=0.3, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_SIZE)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        X = X.long()
        out = self.embed(X)
        # note: output of embedding is always
        # (N, T, D)
        # conv1d expects
        # (N, D, T)
        # conv layers
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = F.relu(out)
        # change it back
        out = out.permute(0, 2, 1)

        h0 = torch.zeros(config.N_LAYERS, X.size(0), config.HIDDEN_DIM).to(config.DEVICE)
        c0 = torch.zeros(config.N_LAYERS, X.size(0), config.HIDDEN_DIM).to(config.DEVICE)
        out, hidden = self.lstm(out, (h0, c0))

        # max pool
        out, _ = torch.max(out, 1)
        # lstm_out, _ = torch.max(lstm_out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(X.size(0), -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden
