from helpers.dataset import pianoroll_dataset_batch
from models import Generalist
from helpers.datapreparation import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def threshold(tensor, threshold=0.5):
    """ Thresholds the tensor in order to binarize it """
    return tensor > threshold




if __name__ == "__main__":
    DATA_DIR = "datasets/training/piano_roll_fs5"
    dataset = pianoroll_dataset_batch(DATA_DIR)


    INPUT_SIZE = 128
    HIDDEN_SIZE = 100

    #inputs = dataset[example_no][composer_no][0:-1]
    #targets = dataset[example_no][composer_no][:][:][1:].float() # Shifted one note to the right
    #print("Shape of inputs: ", inputs.shape)
    #print("Shape of targets: ", targets.shape)
    model = Generalist(INPUT_SIZE, HIDDEN_SIZE, 1)
    loss_func = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    NUM_EPOCHS = 1
    NUM_SONGS = 43
    CUR_SONG = 0
    for epoch in range(NUM_EPOCHS):
        for inputs, tags, targets in dataset:
            CUR_SONG += 1
            print("Epoch {}/{},  processing song {}/{}".format(epoch, NUM_EPOCHS, CUR_SONG, NUM_SONGS), end="\r")
            model.zero_grad() # By default, PyTorch cumulates gradients, so remove them
        
            # Detach hidden from history to clear out hidden state
            model.hidden = model.init_hidden()
            # Forward pass
            output = model(inputs)
            
            # Calculate loss and gradients
            loss = loss_func(output, targets)
            loss.backward()
            # Update weights
            optimizer.step()
        # Print progress
        if epoch%1 == 0:
            print("epoch {}, loss {}".format(epoch,loss.data.numpy()))
            print("Shape of input/output: {}/{}".format(inputs.shape, output.shape))
    output = output.detach().numpy().squeeze()
    print("Max of output: ", output.max())
    print("Min of output: ", output.min())
    print("Mean of output: ", output.mean())
    print("Shape of output:", output.shape)
    piano_roll = threshold(output, threshold=0.1)
    print("Number of elements (notes) over threshold (per timestep): ", sum(piano_roll.flatten())/1127) 
    print(piano_roll)
    np.savetxt('test_music', piano_roll)
