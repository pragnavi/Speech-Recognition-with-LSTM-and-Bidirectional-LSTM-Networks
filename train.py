import torch 
from torch.utils.data import DataLoader
from torch import optim 
import torch.nn as nn
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np 
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import os
import argparse
import yaml 
from tqdm import tqdm 

from model import* 
from preprocessing import AudioMNISTDataset, collate

def train(hp): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(hp['n_mfcc'], hp['n_label'], hp['h'], hp['d'], hp['n_lstm']).to(device)
    #model = Bidirectional_LSTM(hp['n_mfcc'], hp['n_label'], hp['h'], hp['d'], hp['n_lstm']).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    dataset = AudioMNISTDataset(hp['dataset_path'], hp['sampling_rate'], hp['n_mfcc'])
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(hp['train_valid_test_split'])
    
    accuracy_history = {'train': np.zeros(hp['epochs']), 'valid': np.zeros(hp['epochs'])}	
    loss_history = {'train': np.zeros(hp['epochs']), 'valid': np.zeros(hp['epochs'])}
    
    for epoch in tqdm(range(hp['epochs'])):

        model.train()
        no_audio, train_loss, accuracy = 0, 0, 0
        for batch, lengths, labels in DataLoader(train_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True): 
            batch = batch.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            y = model(batch, lengths)
            y_pred = torch.argmax(y, dim=1)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            no_audio += len(batch)
            accuracy += (y_pred == labels).sum().item()
        accuracy /= no_audio
        loss_history['train'][epoch], accuracy_history['train'][epoch] = train_loss, accuracy

        model.eval()
        no_audio, valid_loss, valid_accuracy = 0, 0, 0
        with torch.no_grad(): 
            for batch, lengths, labels in DataLoader(valid_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True):
                batch = batch.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                y = model(batch, lengths)
                y_pred = torch.argmax(y, dim=1)
                loss = criterion(y, labels)

                valid_loss += loss
                no_audio += len(batch)
                valid_accuracy += (y_pred == labels).sum().item()
            valid_accuracy /= no_audio
            loss_history['valid'][epoch], accuracy_history['valid'][epoch] = valid_loss, valid_accuracy

    if not os.path.exists(hp['model_path']): 
        os.mkdir(hp['model_path'])
    torch.save(model.state_dict(), os.path.join(hp['model_path'],'model.pt'))

    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(accuracy_history['train'], c='r', label='training')
    plt.plot(accuracy_history['valid'], c='b', label='validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy vs. Epochs')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(hp['model_path'],'lstm_accuracy.png'))
    plt.clf()

    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(loss_history['train'], c='r', label='training')
    plt.plot(loss_history['valid'], c='b', label='validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs. Epochs')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(hp['model_path'],'lstm_loss.png'))
    plt.clf()

    model.eval()
    batch, lengths, labels = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate, shuffle=True)))
    batch = batch.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)
    y = model(batch, lengths)
    y_pred = torch.argmax(y, dim=1)
    total = (y_pred == labels).sum().item()
    acc  = total/len(y_pred)

    print(f'Train Dataset loss: {(loss_history["train"][-1]):.2f}')
    print(f'Validation Dataset loss: {(loss_history["valid"][-1]):.2f}')
    print(f'Train Dataset accuracy: {(accuracy_history["train"][-1])*100:.2f}')
    print(f'Validation Dataset accuracy: {(accuracy_history["valid"][-1]*100):.2f}')
    print(f'Test Dataset Accuracy: {(acc*100):.2f}')

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=str, default='hyperparameters.yaml', help='The path for hyperparameters.yaml file')
    args = parser.parse_args()

    hyperparameters = yaml.safe_load(open(args.hyperparameters, 'r'))
    train(hyperparameters)