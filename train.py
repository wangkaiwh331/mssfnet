import torch
import torchvision
import torch.nn as nn
from model import MOD_MSSFNet #Importing the MSSFNet model
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.nn import functional
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis
import random
import time
import h5py

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0')
torch.cuda.set_device(0)


from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# Function to load spectral data and labels based on the dataset name
def loadData_spe(name):
    data_path = os.path.join(os.getcwd(), 'data') # Get the data path
    # Load different data files depending on the dataset name
    if name == 'Train500':
        _mat=h5py.File(os.path.join(data_path, 'crism_500_train_spe.mat'))
        data = _mat['train_spe5']
        labels =_mat['train_label5']
        data=np.transpose(data,(1,0))  # Transpose data to align dimensions
    elif name == 'Train1000':
        _mat=h5py.File(os.path.join(data_path, 'crism_1000_train_spe.mat'))
        data = _mat['train_spe4']
        labels =_mat['train_label4']
        data=np.transpose(data,(1,0))
    elif name == 'Train1500':
        _mat=h5py.File(os.path.join(data_path, 'crism_1500_train_spe.mat'))
        data = _mat['train_spe3']
        labels =_mat['train_label3']
        data=np.transpose(data,(1,0))
    elif name == 'Train2000':
        _mat=h5py.File(os.path.join(data_path, 'crism_2000_train_spe.mat'))
        data = _mat['train_spe2']
        labels =_mat['train_label2']
        data=np.transpose(data,(1,0))
    elif name == 'Train2500':
        _mat=h5py.File(os.path.join(data_path, 'crism_2500_train_spe.mat'))
        data = _mat['train_spe1']
        labels =_mat['train_label1']
        data=np.transpose(data,(1,0))
    elif name == 'Train3000':
        _mat=h5py.File(os.path.join(data_path, 'crism_3000_train_spe.mat'))
        data = _mat['train_spe']
        labels = _mat['train_spe_label']
        data=np.transpose(data,(1,0))
    return data, labels

# Function to load spectral validation data based on dataset name
def loadData_val_spe(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Test1000':
        _mat=h5py.File(os.path.join(data_path, 'crism_1000_test_spe.mat'))
        data = _mat['val_spe']
        labels = _mat['val_spe_label']
        data = np.transpose(data, (1, 0))
    return data, labels

# Function to load spatial data and labels based on the dataset name
def loadData_spa(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Train500':
        _mat=h5py.File(os.path.join(data_path, 'crism_500_train_spa.mat'))
        data = _mat['train_spa5']
        labels =_mat['train_label5']
        data=np.transpose(data,(3,0,2,1))
    elif name == 'Train1000':
        _mat=h5py.File(os.path.join(data_path, 'crism_1000_train_spa.mat'))
        data = _mat['train_spa4']
        labels =_mat['train_label4']
        data=np.transpose(data,(3,0,2,1))
    elif name == 'Train1500':
        _mat=h5py.File(os.path.join(data_path, 'crism_1500_train_spa.mat'))
        data = _mat['train_spa3']
        labels =_mat['train_label3']
        data=np.transpose(data,(3,0,2,1))
    elif name == 'Train2000':
        _mat=h5py.File(os.path.join(data_path, 'crism_2000_train_spa.mat'))
        data = _mat['train_spa2']
        labels =_mat['train_label2']
        data=np.transpose(data,(3,0,2,1))
    elif name == 'Train2500':
        _mat=h5py.File(os.path.join(data_path, 'crism_2500_train_spa.mat'))
        data = _mat['train_spa1']
        labels =_mat['train_label1']
        data=np.transpose(data,(3,0,2,1))
    elif name == 'Train3000':
        _mat=h5py.File(os.path.join(data_path, 'crism_3000_train_spa.mat'))
        data = _mat['train_spa']
        labels =_mat['train_spa_label']
        data=np.transpose(data,(3,0,2,1))
    return data, labels

# Function to load spatial validation data based on dataset name
def loadData_val_spa(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Test1000':
        _mat=h5py.File(os.path.join(data_path, 'crism_1000_test_spa.mat'))
        data = _mat['val_spa']
        labels = _mat['val_spa_label']
        data = np.transpose(data, (3, 0, 2, 1))
    return data, labels

# Function to apply Factor Analysis (FA) on input data X with a specified number of components
def applyFA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2])) # Reshape X to 2D
    fa = FactorAnalysis(n_components=numComponents, random_state=0) # Initialize FA model
    newX = fa.fit_transform(newX) # Apply FA
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents)) # Reshape back to original dimensions
    return newX, fa

# Function to pad input data X with zeros along both axes (height and width)
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X # Place original data in center
    return newX

# Function to split data into training and testing sets with a specified test ratio
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y) # Split dataset
    return X_train, X_test, y_train, y_test

# Function for one-hot encoding of the labels
def one_hot(x, class_count):
	# First, construct a vector of size [class_count, class_count] with a diagonal of ones.
	# Second, retain the rows corresponding to the labels and return them.
	return torch.eye(class_count)[x,:]

# Custom dataset class for loading multiple datasets
class MyDataset():
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        x3 = self.dataset3[index]
        return x1, x2, x3

    def __len__(self):
        return len(self.dataset1)

# Function to create image cubes from input data X and labels y, with a specified window size
def createImageCubes_wk(X, y, windowSize=8, removeZeroLabels = True):
    margin = int((windowSize) / 2) # Calculate margin based on window size
    zeroPaddedX = padWithZeros(X, margin=margin)# Pad input data with zeros
    num=0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]>0:# Count non-zero labels
                num=num+1

    patchesData = np.zeros((num, windowSize, windowSize, X.shape[2]), dtype=np.float16) # Create empty array for patches
    patchesLabels = np.zeros((num))# Create empty array for labels
    patchIndex = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r,c]>0:
                patch = zeroPaddedX[r:r + 2*margin, c :c + 2*margin]# Extract patches
                patchesData[patchIndex, :, :, :] = patch# Store patches
                patchesLabels[patchIndex] = y[r , c ]# Store corresponding label
                patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:] # Remove patches with zero labels
        patchesLabels = patchesLabels[patchesLabels>0]# Remove corresponding labels
        patchesLabels -= 1# Adjust labels to start from 0
    return patchesData, patchesLabels

# Function to load training spatial data and labels
def get_train_data_spa(dataset):
    data_train, label_train = loadData_spa(dataset)
    label_train=np.squeeze(label_train)# Remove unnecessary dimensions
    data_train.shape, label_train.shape

    return data_train,label_train

# Function to load validation spatial data and labels
def get_val_data_spa(dataset):
    data_val, label_val = loadData_val_spa(dataset)
    label_val=np.squeeze(label_val) # Remove unnecessary dimensions
    return data_val,label_val

# Function to load training spectral data and labels
def get_train_data_spe(dataset):
    data_train, label_train = loadData_spe(dataset)
    label_train=np.squeeze(label_train)# Remove unnecessary dimensions
    data_train.shape, label_train.shape

    return data_train,label_train

# Function to load validation spectral data and labels
def get_val_data_spe(dataset):
    data_val, label_val = loadData_val_spe(dataset)
    label_val=np.squeeze(label_val)# Remove unnecessary dimensions
    return data_val,label_val

def main():
    start = time.time()  # Start tracking the time
    train_dataset = 'Train3000'    # Name of the training dataset
    val_dataset = 'Test1000'        # Name of the test dataset
    mod_folder = 'mod_MSSFNET'  # Folder for saving the model
    output_units = 21  # Number of output units for classification
    print("Reading training data")  # Print message for training data reading
    train_data_spa, train_label = get_train_data_spa(train_dataset)  # Load spatial training data and labels
    train_data_spe, train_label = get_train_data_spe(train_dataset)  # Load spectral training data and labels
    print("Done")  # Indicate that the training data is loaded
    print("Reading validation data")  # Print message for validation data reading
    val_data_spa, val_label = get_val_data_spa(val_dataset)  # Load spatial validation data and labels
    val_data_spe, val_label = get_val_data_spe(val_dataset)  # Load spectral validation data and labels
    print("Done")  # Indicate that the validation data is loaded
    epochs = 300  # Number of epochs for training
    mod_name = "mod_" + train_dataset + "_(" + str(epochs) + ")_24_AdamW0.0001_0.9_batch500"  # Model name

    # Initialize the model, loss function, and optimizer
    net = MOD_MSSFNet(output_units)  # Create the MSSFNet model
    net = net.cuda()  # Move the model to GPU (if available)
    loss_function = nn.CrossEntropyLoss().to(device)  # Cross-entropy loss function for classification
    optimizer = optim.AdamW(net.parameters(), lr=0.0001, betas=(0.9, 0.99))  # AdamW optimizer with a learning rate of 0.0001

    # Prepare datasets and data loaders for training and validation
    traindata = MyDataset(train_data_spa, train_data_spe, train_label)  # Create dataset for training
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=500, shuffle=False, num_workers=0)  # Data loader for training
    valdata = MyDataset(val_data_spa, val_data_spe, val_label)  # Create dataset for validation
    testnum = 5000  # Number of samples for validation
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=testnum, shuffle=False, num_workers=0)  # Data loader for validation

    # Lists to track loss and accuracy during training
    Loss_list = []
    Accuracy_list = []
    index = 0
    _accuracy = 0  # Initialize accuracy
    stepnum = math.floor(train_data_spa.shape[0] / 2)  # Number of steps per epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)  # Learning rate scheduler

    # Training loop
    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0  # Track the loss for the current epoch
        net.train()  # Set the model to training mode
        print("%d   %d" % (epoch, epochs))  # Print epoch progress

        # Inner loop for training the model with mini-batches
        for step, data in enumerate(train_loader, start=0):
            inputs_spa, inputs_spe, labels = data  # Get inputs and labels
            inputs_spa = inputs_spa.type(torch.FloatTensor)  # Convert to FloatTensor
            inputs_spe = inputs_spe.type(torch.FloatTensor)  # Convert to FloatTensor
            labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to long tensor
            inputs_spa = inputs_spa.cuda()  # Move data to GPU
            inputs_spe = inputs_spe.cuda()  # Move data to GPU
            labels = labels.cuda()  # Move labels to GPU

            optimizer.zero_grad()  # Zero the gradients before backward pass
            outputs = net(inputs_spa, inputs_spe)  # Forward pass
            loss = loss_function(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the model weights

            running_loss += loss.item()  # Accumulate loss for the current epoch

        # Validation step after certain training steps
        if index > 1:  # Perform validation if index > 1
            net.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation during validation
                val_data_iter = iter(val_loader)
                val_data_spa, val_data_spe, val_label = val_data_iter.next()  # Get a batch from validation loader
                print(mod_name)  # Print model name
                val_data_spa = val_data_spa.type(torch.FloatTensor)  # Convert validation data to FloatTensor
                val_data_spe = val_data_spe.type(torch.FloatTensor)  # Convert validation data to FloatTensor
                val_label = torch.tensor(val_label, dtype=torch.long)  # Convert labels to long tensor
                val_data_spa = val_data_spa.cuda()  # Move validation data to GPU
                val_data_spe = val_data_spe.cuda()  # Move validation data to GPU
                val_label = val_label.cuda()  # Move validation labels to GPU
                outputs = net(val_data_spa, val_data_spe)  # Get predictions
                predict_y = torch.max(outputs, dim=1)[1]  # Get the predicted class labels
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)  # Calculate accuracy

                Loss_list.append(running_loss)  # Append the current running loss to the loss list
                Accuracy_list.append(accuracy)  # Append the current accuracy to the accuracy list

                # Print training and validation statistics
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % (index + 1, 100, running_loss / stepnum, accuracy))
                running_loss = 0.0  # Reset running loss for next iteration

                # Save model checkpoints periodically
                if index > 10 and index % 2 == 0:
                    save_path = "./model/" + mod_name + "_" + str(epoch + 1) + "_" + str(accuracy) + ".pth"
                    torch.save(net.state_dict(), save_path)  # Save model state_dict
                index = index + 1
        else:
            index = index + 1  # Increment index if condition is not met

    # After training, set model to evaluation mode
    net.eval()

    # Plot and save accuracy and loss curves
    x1 = range(0, len(Accuracy_list))  # X-axis for accuracy
    x2 = range(0, len(Loss_list))  # X-axis for loss
    y1 = Accuracy_list  # Y-axis for accuracy
    y2 = Loss_list  # Y-axis for loss


    sio.savemat("./AccuracyLoss/" + "Accuracy_list-" + mod_name + ".mat", {'outputs': y1})
    sio.savemat("./AccuracyLoss/" + "Loss_list-" + mod_name + ".mat", {'outputs': y2})
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig("./AccuracyLoss/" + "accuracy_loss-" + mod_name + ".jpg")
    end = time.time()
    print(end - start)
    print('Finished Training')

    end = time.time()
    minutes=int((end - start)/60)
    seconds=int((end - start)%60)
    print("%d 分 %d 秒"%(minutes,seconds))
    plt.show()


if __name__ == '__main__':
    print(torch.__version__)
    main()
