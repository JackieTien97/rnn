# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt

import data
import model


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(rnn_model, valid_data, device):
    rnn_model.eval()
    total_loss = 0.
    total_correct = 0
    valid_loader = iter(valid_data)
    total_count = 0.
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch_data, batch_target = batch.text, batch.target
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            output = rnn_model(batch_data)
            loss = criterion(output.view(-1, VOCAB_SIZE), batch_target.view(-1))
            _, predictions = torch.max(output.view(-1, VOCAB_SIZE), 1)
            total_count += np.multiply(*batch_data.size())
            total_loss += loss.item() * np.multiply(*batch_data.size())
            total_correct += torch.sum(predictions == batch_target.view(-1).data)

    loss = total_loss / total_count
    epoch_acc = total_correct.double() / total_count
    return np.exp(loss), epoch_acc.item()


########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train(rnn_model, train_data, optim, sched, device):
    rnn_model.train()
    train_loader = iter(train_data)
    total_loss = 0.0
    total_correct = 0
    total_count = 0.0
    for i, batch in enumerate(train_loader):
        batch_data, batch_target = batch.text, batch.target
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        rnn_model.zero_grad()
        output = rnn_model(batch_data)
        loss = criterion(output.view(-1, VOCAB_SIZE), batch_target.view(-1))
        _, predictions = torch.max(output.view(-1, VOCAB_SIZE), 1)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), GRAD_CLIP)  # CLIP,防止梯度爆炸
        optim.step()

        total_loss += loss.item() * np.multiply(*batch_data.size())
        total_correct += torch.sum(predictions == batch_target.view(-1).data)
        total_count += np.multiply(*batch_data.size())

    sched.step()
    epoch_loss = total_loss / total_count
    epoch_acc = total_correct.double() / total_count
    return np.exp(epoch_loss), epoch_acc.item()


if __name__ == '__main__':

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    num_epochs = 50
    batch_size = 32
    bptt_len = 64
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
    emsize = 256
    nhid = 256
    nlayers = 2
    nhead = 2
    dropout = 0.2
    MyModel = model.TransformerModel(VOCAB_SIZE, emsize, nhead, nhid, nlayers, dropout)
    print(MyModel)
    MyModel.to(device)
    ########################################

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    step_size = 10
    optimizer = torch.optim.Adam(MyModel.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.3)
    GRAD_CLIP = 0.5

    ########################################

    save_directory = '../best_model/transformer/'
    train_loss_array = []
    train_acc_array = []
    valid_loss_array = []
    valid_acc_array = []
    best_acc = 0.0
    best_train_acc = 0.0
    best_pp = 9999999
    # Loop over epochs.
    for epoch in range(1, num_epochs + 1):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(MyModel, train_iter, optimizer, scheduler, device)
        train_acc_array.append(train_acc)
        train_loss_array.append(train_loss)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = evaluate(MyModel, val_iter, device)
        valid_acc_array.append(valid_acc)
        valid_loss_array.append(valid_loss)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if valid_loss < best_pp:
            best_pp = valid_loss
            best_model = MyModel
            torch.save(best_model, os.path.join(save_directory, 'best_model_layer2_bptt64_head2_em256.pt'))
        if valid_acc > best_acc:
            best_acc = valid_acc

    print("train_acc_array: ", train_acc_array)
    print("valid_acc_array: ", valid_acc_array)
    print("best train accuracy is: {:.4f}".format(best_train_acc))
    print("best validation accuracy is: {:.4f}".format(best_acc))
    print("best validation pp is: {:.4f}".format(best_pp))

    plt.figure()
    plt.title("Training and Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1, num_epochs + 1), train_acc_array, label="Training")
    plt.plot(range(1, num_epochs + 1), valid_acc_array, label="Validation")
    plt.xticks(np.arange(1, num_epochs + 1, 20.0))
    plt.legend()
    plt.savefig('Transformer_Accuracy_layer2_bptt64_head2_em256_dropout_0_2.jpg')

    plt.figure()
    plt.title("Training and Validation Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), train_loss_array, label="Training")
    plt.plot(range(1, num_epochs + 1), valid_loss_array, label="Validation")
    plt.xticks(np.arange(1, num_epochs + 1, 20.0))
    plt.legend()
    plt.savefig('Transformer_Loss_layer2_bptt64_head2_em256_dropout_0_2.jpg')

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
