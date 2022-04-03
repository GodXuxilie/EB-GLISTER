import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from cords.utils.data.datasets.SL.builder import loadGloveModel

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, wordvec_dim, weight_path, num_layers=2, hidden_size=150):
        super(LSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embedding_length = wordvec_dim 
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)
        weight = torch.tensor(wordvec.values, dtype=torch.float)  # word embedding for the embedding layer
        
        self.embedding = nn.Embedding(
            weight.shape[0], self.embedding_length)  # Embedding layer
        self.embedding = self.embedding.from_pretrained(
            weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing
        self.lstm = nn.LSTM(self.embedding_length,
                            self.hidden_size, num_layers=num_layers, batch_first=True)  # lstm
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_sentence, last=False, freeze=False, emb_last=False):
        if freeze:
            with torch.no_grad():
                x = self.embedding(input_sentence)  # (batch_size, batch_dim, embedding_length)
                output, (final_hidden_state, final_cell_state) = self.lstm(x)
        else:
            x = self.embedding(input_sentence)  # (batch_size, batch_dim, embedding_length)
            output, (final_hidden_state, final_cell_state) = self.lstm(x)
        logits = self.fc(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & logits.size() = (batch_size, num_classes)
        if emb_last:
            return logits, final_hidden_state[-1], x
        if last:
            return logits, final_hidden_state[-1] 
        else:
            return logits

    def get_feature_dim(self):
        return self.hidden_size

    def get_embedding_dim(self):
        return self.hidden_size


class SimplifiedClassifier(nn.Module):
    def __init__(self, num_classes, wordvec_dim, weight_path, num_layers=1, hidden_size=150):
        super(SimplifiedClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embedding_length = wordvec_dim 
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)
        weight = torch.tensor(wordvec.values, dtype=torch.float)  # word embedding for the embedding layer
        
        self.embedding = nn.Embedding(
            weight.shape[0], self.embedding_length)  # Embedding layer
        self.embedding = self.embedding.from_pretrained(
            weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing
        # self.lstm = nn.LSTM(self.embedding_length,
        #                     self.hidden_size, num_layers=num_layers, batch_first=True)  # lstm
        self.fc1 = nn.Linear(self.embedding_length, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_sentence, last=False, freeze=False, emb_last=False, emb_init=False):
        if emb_init:
            x = input_sentence
            x.requires_grad_()
            x_mean = torch.mean(x, dim=1)
            final_hidden_state= self.fc1(x_mean)
        else:
            if freeze:
                with torch.no_grad():
                    x = self.embedding(input_sentence)  # (batch_size, batch_dim, embedding_length)
                    x_mean = torch.mean(x, dim=1)
                    final_hidden_state= self.fc1(x_mean)
            else:
                x = self.embedding(input_sentence)  # (batch_size, batch_dim, embedding_length)
                x_mean = torch.mean(x, dim=1)
                final_hidden_state= self.fc1(x_mean)
        logits = self.fc2(final_hidden_state)  # final_hidden_state.size() = (1, batch_size, hidden_size) & logits.size() = (batch_size, num_classes)

        if emb_last:
            return logits, final_hidden_state[-1], x
        if last:
            return logits, final_hidden_state[-1] 
        else:
            return logits

    def get_feature_dim(self):
        return self.hidden_size

    def get_embedding_dim(self):
        return self.hidden_size



class BiLSTMClassifier(nn.Module):
    
    def __init__(self, num_classes, wordvec_dim, weight_path, num_layers=1, hidden_size=150, dropout=0.1):
        super(BiLSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embedding_length = wordvec_dim 
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)
        weight = torch.tensor(wordvec.values, dtype=torch.float)  # word embedding for the embedding layer
        
        self.embedding = nn.Embedding(
            weight.shape[0], self.embedding_length)  # Embedding layer
        self.embedding = self.embedding.from_pretrained(
            weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing

        # self.args = args
        # self.hidden_dim = args.lstm_hidden_dim
        # self.num_layers = args.lstm_num_layers
        # V = args.embed_num
        # D = args.embed_dim
        # C = args.class_num
        # self.embed = nn.Embedding(V, D, max_norm=config.max_norm)
        # self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # pretrained  embedding
        # if args.word_Embedding:
            # self.embed.weight.data.copy_(args.pretrained_weight)
        
        self.dropout = dropout

        self.bilstm = nn.LSTM(self.embedding_length, self.hidden_size, num_layers=num_layers, dropout=self.dropout, bidirectional=True, bias=False)

        # self.hidden2label1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.hidden2label2 = nn.Linear(self.hidden_size * 2, self.num_classes)
        # self.dropout = nn.Dropout(config.dropout)

    # def forward(self, x):
        
    def forward(self, input_sentence, last=False, freeze=False, emb_last=False):
        if freeze:
            with torch.no_grad():
                x = self.embedding(input_sentence)
                emb_x = x.view(len(x), x.size(1), -1)
                bilstm_out, _ = self.bilstm(emb_x)
        else:
            x = self.embedding(input_sentence)
            emb_x = x.view(len(x), x.size(1), -1)
            bilstm_out, _ = self.bilstm(emb_x)
            # print(bilstm_out.shape)

        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        # fea = y
        # print(y.shape)
        # y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(bilstm_out)
        logits = y

        # print(x, logits, )
        if emb_last:
            return logits, bilstm_out, x
        if last:
            return logits, bilstm_out
        else:
            return logits

    def get_feature_dim(self):
        return self.hidden_size

    def get_embedding_dim(self):
        return self.hidden_size