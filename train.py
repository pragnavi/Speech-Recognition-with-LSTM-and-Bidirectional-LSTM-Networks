import torch 
from torch.utils.data import DataLoader
from torch import optim 
import torch.nn as nn
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np 
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import os
import argparse
import yaml 
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix

from model import* 
from preprocessing import AudioMNISTDataset, collate

def train(hp): 
    # Set Device to cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Models below: If you want to run bidirectional, please uncomment that specific model and run again
    model = LSTM(hp['n_mfcc'], hp['n_label'], hp['h'], hp['d'], hp['n_lstm']).to(device)
    #model = Bidirectional_LSTM(hp['n_mfcc'], hp['n_label'], hp['h'], hp['d'], hp['n_lstm']).to(device)
    # Loss function is Neg Log Likelihood Loss
    criterion = nn.NLLLoss()
    # Optimizer is ADAM
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    dataset = AudioMNISTDataset(hp['dataset_path'], hp['sampling_rate'], hp['n_mfcc'])
    #Split AudioMNISTDataset in to train, validation, test
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(hp['train_valid_test_split'])
    
    # Keep track of training and validation accuracies
    accuracy_history = {'train': np.zeros(hp['epochs']), 'valid': np.zeros(hp['epochs'])}	
    loss_history = {'train': np.zeros(hp['epochs']), 'valid': np.zeros(hp['epochs'])}
    
    best_accuracy = float('-inf')
    for epoch in tqdm(range(hp['epochs'])):

        # Training
        model.train()
        no_audio, train_loss, accuracy = 0, 0, 0
        for batch, lengths, labels in DataLoader(train_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True): 
            # Transfer tensors to GPU
            batch = batch.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            # Reset the optimizer
            optimizer.zero_grad()

            # Calculate predictions for batch
            y = model(batch, lengths)
            y_pred = torch.argmax(y, dim=1)

            # Calculate and back-propagate loss
            loss = criterion(y, labels)
            loss.backward()
            # Update the optimizer
            optimizer.step()

            # Calculate training loss
            train_loss += loss.item()
            no_audio += len(batch)
            # Calculate training accuracy
            accuracy += (y_pred == labels).sum().item()
        train_loss /= no_audio
        accuracy /= no_audio
        loss_history['train'][epoch], accuracy_history['train'][epoch] = train_loss, accuracy

        # Validation
        model.eval()
        no_audio, valid_loss, valid_accuracy = 0, 0, 0
        with torch.no_grad(): 
            for batch, lengths, labels in DataLoader(valid_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True):
                # Transfer to GPU
                batch = batch.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                # Calculate predictions for batch
                y = model(batch, lengths)
                y_pred = torch.argmax(y, dim=1)
                loss = criterion(y, labels)

                # Calculate validation loss
                valid_loss += loss.item()
                no_audio += len(batch)

                # Calculate validation accuracy
                valid_accuracy += (y_pred == labels).sum().item()
            valid_loss /= no_audio
            valid_accuracy /= no_audio
            # Store loss, accuracy history
            loss_history['valid'][epoch], accuracy_history['valid'][epoch] = valid_loss, valid_accuracy

            if not os.path.exists(hp['model_path']): 
                os.mkdir(hp['model_path'])
            if valid_accuracy > best_accuracy:
                # Save the model with best validation accuracy
                torch.save(model.state_dict(), os.path.join(hp['model_path'],'model.pt'))
                # Update the best accuracy
                best_accuracy = valid_accuracy

    # Plot the training and validation accuracy history
    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(accuracy_history['train'], c='r', label='training')
    plt.plot(accuracy_history['valid'], c='b', label='validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy vs. Epochs')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(hp['model_path'],'accuracy.png'))
    plt.clf()

    # Plot the training and validation loss history
    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(loss_history['train'], c='r', label='training')
    plt.plot(loss_history['valid'], c='b', label='validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs. Epochs')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(hp['model_path'],'loss.png'))
    plt.clf()
    
    # Load the best model and accuracy history
    model.load_state_dict(torch.load(os.path.join(hp['model_path'],'model.pt')))

    # Evaluation mode
    model.eval()

    # Retrieve test set as a single batch
    batch, lengths, labels = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate, shuffle=True)))
    batch = batch.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)


    # Calculate predictions for test set
    y = model(batch, lengths)
    y_pred = torch.argmax(y, dim=1)
    total = (y_pred == labels).sum().item()
    # Calculate test accuracy
    acc  = total/len(y_pred)

    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(labels.to('cpu').numpy(), y_pred.detach().to('cpu').numpy(), labels=hp['labels'])

    # Plot the confusion matrix 
    df = pd.DataFrame(cm, index=hp['labels'], columns=hp['labels'])
    plt.figure(figsize=(10,7))
    sns.heatmap(df, annot=True,fmt='g',cmap="Blues")
    plt.title('Confusion matrix for test dataset predictions', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.savefig(os.path.join(hp['model_path'],'confusion_matrix.png'))
    plt.clf() 

    print(f'Train Dataset loss: {(loss_history["train"][-1])}')
    print(f'Validation Dataset loss: {(loss_history["valid"][-1])}')
    print(f'Train Dataset accuracy: {(accuracy_history["train"][-1])*100:.2f}')
    print(f'Validation Dataset accuracy: {(accuracy_history["valid"][-1]*100):.2f}')
    print(f'Test Dataset Accuracy: {(acc*100):.2f}')

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=str, default='hyperparameters.yaml', help='The path for hyperparameters.yaml file')
    args = parser.parse_args()

    hyperparameters = yaml.safe_load(open(args.hyperparameters, 'r'))
    train(hyperparameters)