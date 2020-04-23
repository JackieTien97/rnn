# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt

import data
import model


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(rnn_model, valid_data, device):
    rnn_model.eval()
    total_loss = 0.
    valid_loader = iter(valid_data)
    total_count = 0.
    with torch.no_grad():
        hidden = rnn_model.init_hidden(batch_size, requires_grad=False)
        for i, batch in enumerate(valid_loader):
            batch_data, batch_target = batch.text, batch.target
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            hidden = repackage_hidden(hidden)
            output, hidden = rnn_model(batch_data, hidden)
            loss = criterion(output.view(-1, VOCAB_SIZE), batch_target.view(-1))
            total_count += np.multiply(*batch_data.size())
            total_loss += loss.item() * np.multiply(*batch_data.size())

    loss = total_loss / total_count
    return np.exp(loss)


########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train(rnn_model, train_data, optim, sched, device):
    rnn_model.train()
    train_loader = iter(train_data)
    hidden = rnn_model.init_hidden(batch_size)
    total_loss = 0.0
    total_size = 0.0
    for i, batch in enumerate(train_loader):
        batch_data, batch_target = batch.text, batch.target
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        hidden = repackage_hidden(hidden)
        rnn_model.zero_grad()
        output, hidden = rnn_model(batch_data, hidden)
        loss = criterion(output.view(-1, VOCAB_SIZE), batch_target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), GRAD_CLIP)  # CLIP,防止梯度爆炸
        optim.step()
        total_loss += loss.item() * batch_data.size(1)
        total_size += batch_data.size(1)

    sched.step()
    epoch_loss = total_loss / total_size
    return epoch_loss


if __name__ == '__main__':

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    num_epochs = 200
    batch_size = 20
    bptt_len = 35
    SEED = 12345

    print("num_epochs: ", num_epochs)
    print("batch_size: ", batch_size)
    print("bptt_len: ", bptt_len)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(SEED)
    print("seed: ", SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    # load data
    train_iter, val_iter, VOCAB_SIZE = data.get_data("../data/ptb/", batch_size, bptt_len, device)
    print("VOCAB_SIZE: ", VOCAB_SIZE)

    # WRITE CODE HERE within two '#' bar
    ########################################
    # Build LMModel best_model (build your language best_model here)
    embedding_size = 128
    hidden_size = 256
    layer_number = 5
    MyModel = model.LMModel(VOCAB_SIZE, embedding_size, hidden_size, layer_number, 0.0, True)
    print(MyModel)
    MyModel.to(device)
    ########################################

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    step_size = 20
    optimizer = torch.optim.Adam(MyModel.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    GRAD_CLIP = 1

    ########################################

    save_directory = '../best_model/'
    train_loss_array = []
    valid_loss_array = []
    best_loss = float('inf')
    # Loop over epochs.
    for epoch in range(1, num_epochs + 1):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss = train(MyModel, train_iter, optimizer, scheduler, device)
        train_loss_array.append(train_loss)
        print("training loss: {:.4f}".format(train_loss))
        valid_loss = evaluate(MyModel, val_iter, device)
        valid_loss_array.append(valid_loss)
        print("validation perplexity: {:.4f}".format(valid_loss))
        if valid_loss < best_loss:
            best_acc = valid_loss
            best_model = MyModel
            torch.save(best_model, os.path.join(save_directory, 'best_model.pt'))

    print("train_loss_array: ", train_loss_array)
    print("valid_loss_array: ", valid_loss_array)
    print("best validation perplexity is: {:.4f}".format(best_loss))

    plt.figure()
    plt.title("Training and Validation Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), train_loss_array, label="Training")
    plt.plot(range(1, num_epochs + 1), valid_loss_array, label="Validation")
    plt.xticks(np.arange(1, num_epochs + 1, 20.0))
    plt.legend()
    plt.savefig('Loss.jpg')

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
