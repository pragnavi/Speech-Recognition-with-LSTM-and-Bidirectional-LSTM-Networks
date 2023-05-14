"""
Collation function from https://www.codefull.org/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
https://github.com/aniruddhapal211316/spoken_digit_recognition/blob/main/dataset.py
build file function from https://www.kaggle.com/code/kyrobc/audio-mnist-classifier-with-98-accuracy-15-min

"""
import torch 
import torchvision
import torchaudio
from torchaudio.transforms import Resample, MFCC
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import os

#Removes 0th coefficient
class TrimMFCCs: 

    def __call__(self, batch): 
        return batch[:, 1:, :]

#Normalize each sequence in a batch
#Subtract mean  and divide by std deviation of batch
class Standardize:
    
    def __call__(self, batch): 
        for sequence in batch: 
            sequence -= sequence.mean(axis=0)
            sequence /= sequence.std(axis=0)
        return batch 

#self.transform creates a preprocessin transformation pipeline
#1. Convert .wav file into MFCC
#2.  Remove 0th coefficient
#3.  Normalize
class AudioMNISTDataset(Dataset): 
    
    def __init__(self, path, sr, n_mfcc): 
        assert os.path.exists(path), f'The path for dataset does not exist' 
        self.path = path 
        self.audio_files = self._build_files(path)
        self.sr = sr
        self.transform = torchvision.transforms.Compose([
            MFCC(sample_rate = sr, n_mfcc = n_mfcc+1), 
            TrimMFCCs(),
            Standardize(),
            ])
        
    def _build_files(self,path):
        files = {}
        index = 0
        for i in range(1, 61):
            num = "0%d" % i if i < 10 else "%d" % i
            for j in range(50):
                for k in range(10):
                    files[index] = [
                        path + num + "/%d_%s_%d.wav" % (
                            k, num, j),
                        k
                    ]
                    index += 1         
        return files    

    def __len__(self): 
        return 30000

    # Select samples
    def __getitem__(self, index): 
        # Load data and get label
        audio, sr = torchaudio.load(os.path.join(self.audio_files[index][0]))
        audio = Resample(sr, self.sr)(audio)
        mfccs = self.transform(audio)
        return mfccs, int(self.audio_files[index][1])

    # Split dataset into train, validation, test
    def split_dataset(self, split_lengths): 
        valid_dataset_len = int((split_lengths[1]/100)*len(self))
        test_dataset_len = int((split_lengths[2]/100)*len(self))
        train_dataset_len = len(self) - (valid_dataset_len+test_dataset_len)
        train_dataset, valid_dataset, test_dataset = random_split(self, [train_dataset_len, valid_dataset_len, test_dataset_len])
        return train_dataset, valid_dataset, test_dataset

def collate(batch): 
    """
    Parameters
    ----------
    batch: list(tuple(Tensor, int))
        Collection of (sequence, label) pairs, where the nth sequence is of shape 1xDxTn
    """

    # Sort the (mfccs, label) pairs in descending order of recording duration
    batch.sort(key = (lambda x: x[0].shape[-1]), reverse=True)
    # Shape: list(tuple(1xDxT, int))
    sequences = [mfccs.squeeze(0).permute(1, 0) for mfccs, _ in batch]
    # Shape: list(TxD)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    # Shape: BxTxD

    # Store the lengths of each unpadded sequence (in order to unpad later)
    lengths = torch.LongTensor([len(mfccs) for mfccs in sequences])
    # Shape: B
    labels = torch.LongTensor([label for _, label in batch])
    # Shape: B
    return padded_sequences, lengths, labels
    # Shapes: BxTxD, B, B