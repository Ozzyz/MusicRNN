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
    HIDDEN_SIZE = 15
    NUM_EPOCHS = len(dataset)

    composer_no = 0
    example_no = 0
    inputs = dataset[example_no][composer_no][0:-1]
    targets = dataset[example_no][composer_no][:][:][1:].float() # Shifted one note to the right

    model = Generalist(INPUT_SIZE, HIDDEN_SIZE, 1)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)



    for epoch in range(20):
        optimizer.zero_grad() # By default, PyTorch cumulates gradients, so remove them
        # Forward
        output, _ = model(inputs)
        
        loss = loss_func(output, targets)
        loss.backward(retain_graph=(epoch != 20))
        optimizer.step()
        if epoch%10 == 0:
            print("epoch {}, loss {}".format(epoch,loss.data.numpy()))

    output = output.detach().numpy().squeeze()
    print("Shape of output:", output.shape)
    piano_roll = threshold(output, threshold=0.02)
    print(piano_roll)
    np.savetxt('test_music', piano_roll)