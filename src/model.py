import torch.nn as nn


class LMModel(nn.Module):
    # Language best_model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput, nhid, nlayers, dropout=0.5, bidirectional=False):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you RNN best_model here. You can add additional parameters to the function.
        self.rnn = nn.GRU(ninput, nhid, nlayers, bidirectional=bidirectional)
        ########################################
        if bidirectional:
            self.decoder = nn.Linear(2 * nhid, nvoc)
        else:
            self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        self.bidirectional = bidirectional

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden):
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        output, hidden = self.rnn(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        if self.bidirectional:
            return weight.new_zeros(self.nlayers * 2, batch_size, self.nhid, requires_grad=requires_grad)
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid, requires_grad=requires_grad)
