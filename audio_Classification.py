from create_AudioDataset import*
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

def create_model():
    # https://github.com/efwoods/Audio-MNIST-Digit-Recognition/blob/main/Audio_MNIST_Digit_Recognition.ipynb
    # An ANN is sufficient for our purposes.  Convolutional layers are unecessary 
    # since when we convert audios to their corresponding spectrograms, we will have similar spectrograms 
    # for similar audios irrespective of who the speaker is, and what their pitch and timber is like. 
    # So local spatiality is not going to be a problem
    # Create a Sequential Object
    model = Sequential()

    # Add first layer with 100 neurons to the sequental object
    model.add(Dense(100, input_shape = (40, ), activation = 'relu'))

    # Add second layer with 100 neurons to the sequental object
    model.add(Dense(100, activation = 'relu'))

    # Add third layer with 100 neurons to the sequental object
    model.add(Dense(100, activation = 'relu'))

    # Output layer with 10 neurons as it has 10 classes
    model.add(Dense(10, activation = 'softmax'))

    return model


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dataset = preprocess_and_create_dataset()
    # Storing the class as int 
    dataset['class'] = [int(x) for x in dataset['class']]
    
    X = np.array(dataset['features'].to_list())
    Y = np.array(dataset['class'].to_list())

    # Create train set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.75, shuffle = True, random_state = 8)
    print(X_train.shape)

    model = create_model()
    # Print Summary of the model
    print(model.summary())

    # Compile the model
    model.compile(loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')

    num_epochs = 100
    batch_size = 32

    # Fit the model
    fit_history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = num_epochs, batch_size = batch_size, verbose = 1)

    # Make predictions on the test set
    Y_pred = model.predict(X_test)
    Y_pred = [np.argmax(i) for i in Y_pred]

    dict_hist = fit_history.history



    number_of_epochs = 35

    list_ep = [i for i in range(1, number_of_epochs+1)]

    plt.figure(figsize = (8, 8))
    plt.plot(list_ep, dict_hist['accuracy'][:number_of_epochs], ls = '-', label = 'accuracy')
    plt.plot(list_ep, dict_hist['val_accuracy'][:number_of_epochs], ls = '-', label = 'val_accuracy')  
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize = (8, 8))
    plt.plot(list_ep, dict_hist['loss'][:number_of_epochs], ls = '-', label = 'loss')
    plt.plot(list_ep, dict_hist['val_loss'][:number_of_epochs], ls = '-', label = 'val_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
