#https://github.com/efwoods/Audio-MNIST-Digit-Recognition/blob/main/Audio_MNIST_Digit_Recognition.ipynb
# For Audio Preprocessing
import librosa
import librosa.display as dsp
from IPython.display import Audio

# For Data Preprocessing
import pandas as pd
import numpy as np
import os

# For Data Visualization
from tqdm import tqdm



def get_audio(digit = 0):

    # Audio Sample Directory
    sample = np.random.randint(1, 10)

    # Index of Audio
    index = np.random.randint(1, 5)
    
    # Modified file location
    if sample < 10:
        file = f'AudioMNIST/data/0{sample}/{digit}_0{sample}_{index}.wav'

    else:
        file = f'AudioMNIST/data/{sample}/{digit}_{sample}_{index}.wav'

    
    # Get Audio from the location
    # Audio will be automatically resampled to the given rate (default sr = 22050)
    data, sample_rate = librosa.load(file)
    
    # Plot the audio wave
    dsp.waveshow(data, sr = sample_rate)
    plt.show()
    
    # Show the widget
    return Audio(data = data, rate = sample_rate)



# A function which returns audio file for a mentioned digit
def get_audio_raw(digit = 0):

    # Audio Sample Directory
    sample = np.random.randint(1, 10)

    # Index of Audio
    index = np.random.randint(1, 5)
    
    # Modified file location
    if sample < 10:
        file = f'AudioMNIST/data/0{sample}/{digit}_0{sample}_{index}.wav'

    else:
        file = f'AudioMNIST/data/{sample}/{digit}_{sample}_{index}.wav'

    
    # Get Audio from the location
    data, sample_rate = librosa.load(file)

    # Return audio
    return data, sample_rate



# Will take an audio file as input and return extracted features using MEL_FREQUENCY CEPSTRAL COEFFICIENT as the output
def extract_features(file):

    # Load audio and its sample rate
    audio, sample_rate = librosa.load(file)

    # Extract features using mel-frequency coefficient
    extracted_features = librosa.feature.mfcc(y = audio,
                                              sr = sample_rate,
                                              n_mfcc = 40)
    
    # Scale the extracted features
    extracted_features = np.mean(extracted_features.T, axis = 0)

    # Return the extracted features
    return extracted_features


def preprocess_and_create_dataset():

    # Path of the folder where the audio files are present
    root_folder_path = 'AudioMNIST/data/'

    # Empty List to create dataset
    dataset = []
    
    # Iterating through folders where each folder has the audio of each digit
    for folder in tqdm(range(1, 11)):

        if folder < 10:

            # Path of the folder
            folder = os.path.join(root_folder_path, "0" + str(folder))

        else:
            folder = os.path.join(root_folder_path, str(folder))
            
        # Iterate through each file of the present folder
        for file in tqdm(os.listdir(folder)):

            # Path of the file
            abs_file_path = os.path.join(folder, file)

            # Pass path of file to the extracted_features() function to create features
            extracted_features = extract_features(abs_file_path) 

            # Class of the audio, i.e., the digit it represents
            class_label = file[0]
            
            # Append a list where the feature represents a column and class of the digit represents another column
            dataset.append([extracted_features, class_label])
    
    # After iterating through all the folders, convert the list to a DataFrame
    return pd.DataFrame(dataset, columns = ['features', 'class'])


