import torch
import torchvision
import torch.nn as nn
from model import MOD_WK
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


def loadData_spe(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Train500':
        _mat=h5py.File(os.path.join(data_path, 'crism_500_train_spe.mat'))
        data = _mat['train_spe5']
        labels =_mat['train_label5']
        data=np.transpose(data,(1,0))
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

def loadData_val_spe(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Test1000':
        _mat=h5py.File(os.path.join(data_path, 'crism_1000_test_spe.mat'))
        data = _mat['val_spe']
        labels = _mat['val_spe_label']
        data = np.transpose(data, (1, 0))
    return data, labels

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

def loadData_val_spa(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Test1000':
        _mat=h5py.File(os.path.join(data_path, 'crism_1000_test_spa.mat'))
        data = _mat['val_spa']
        labels = _mat['val_spa_label']
        data = np.transpose(data, (3, 0, 2, 1))
    return data, labels


def applyFA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    fa = FactorAnalysis(n_components=numComponents, random_state=0)
    newX = fa.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, fa

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
def one_hot(x, class_count):
	# First, construct a vector of size [class_count, class_count] with a diagonal of ones.
	# Second, retain the rows corresponding to the labels and return them.
	return torch.eye(class_count)[x,:]

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

def createImageCubes_wk(X, y, windowSize=8, removeZeroLabels = True):
    margin = int((windowSize) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    num=0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]>0:
                num=num+1

    patchesData = np.zeros((num, windowSize, windowSize, X.shape[2]), dtype=np.float16)
    patchesLabels = np.zeros((num))
    patchIndex = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r,c]>0:
                patch = zeroPaddedX[r:r + 2*margin, c :c + 2*margin]
                # patch =X[r:r + 2*margin, c :c + 2*margin]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r , c ]
                patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def get_train_data_spa(dataset):
    data_train, label_train = loadData_spa(dataset)
    label_train=np.squeeze(label_train)
    data_train.shape, label_train.shape

    return data_train,label_train

def get_val_data_spa(dataset):
    data_val, label_val = loadData_val_spa(dataset)
    label_val=np.squeeze(label_val)
    return data_val,label_val

def get_train_data_spe(dataset):
    data_train, label_train = loadData_spe(dataset)
    label_train=np.squeeze(label_train)
    data_train.shape, label_train.shape

    return data_train,label_train

def get_val_data_spe(dataset):
    data_val, label_val = loadData_val_spe(dataset)
    label_val=np.squeeze(label_val)
    return data_val,label_val

def main():
    start = time.time()
    train_dataset = 'Train3000'
    val_dataset = 'Test1000'
    mod_folder = 'mod_MSSFNET'
    output_units=21
    print("读取训练数据")
    train_data_spa, train_label = get_train_data_spa(train_dataset)
    train_data_spe, train_label = get_train_data_spe(train_dataset)
    print("结束")
    print("读取验证数据")
    val_data_spa, val_label = get_val_data_spa(val_dataset)
    val_data_spe, val_label = get_val_data_spe(val_dataset)
    print("结束")
    epochs= 300
    mod_name = "mod_" + train_dataset + "_(" + str(epochs) + ")_24_AdamW0.0001_0.9_batch500"

    net = MOD_WK(output_units)
    net=net.cuda()
    loss_function = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.0001,betas=(0.9,0.99))

    traindata=MyDataset(train_data_spa,train_data_spe,train_label)

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=500,
                                               shuffle=False, num_workers=0)

    valdata=MyDataset(val_data_spa,val_data_spe,val_label)
    testnum=5000
    val_loader = torch.utils.data.DataLoader(valdata,batch_size=testnum,
                                               shuffle=False, num_workers=0)

    Loss_list = []
    Accuracy_list = []
    index=0
    _accuracy=0
    stepnum=math.floor(train_data_spa.shape[0]/2)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.8)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        print("%d   %d" % (epoch, epochs))
        # scheduler.step()
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs_spa,inputs_spe, labels = data
            inputs_spa = inputs_spa.type(torch.FloatTensor)
            inputs_spe = inputs_spe.type(torch.FloatTensor)
            labels=torch.tensor(labels,dtype=torch.long)
            # labels = labels.type(torch.FloatTensor)
            inputs_spa=inputs_spa.cuda()
            inputs_spe=inputs_spe.cuda()
            labels=labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs_spa,inputs_spe)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if index > 1:  # print every 500 mini-batches
            net.eval()
            with torch.no_grad():
                val_data_iter = iter(val_loader)
                val_data_spa, val_data_spe, val_label = val_data_iter.next()
                print(mod_name)
                val_data_spa = val_data_spa.type(torch.FloatTensor)
                val_data_spe = val_data_spe.type(torch.FloatTensor)
                val_label=torch.tensor(val_label,dtype=torch.long)
                val_data_spa = val_data_spa.cuda()
                val_data_spe = val_data_spe.cuda()
                val_label = val_label.cuda()
                outputs = net(val_data_spa,val_data_spe)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                Loss_list.append(running_loss)
                Accuracy_list.append(accuracy)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (index + 1, 100, running_loss / stepnum, accuracy))
                running_loss = 0.0

                if index >10 and index%2==0:
                    save_path = "./model/" + mod_name + "_" + str(epoch + 1) + "_" + str(accuracy) + ".pth"
                    torch.save(net.state_dict(), save_path)
                index = index + 1
        else:
            index = index + 1
    net.eval()
    x1 = range(0, Accuracy_list.__len__())
    x2 = range(0, Loss_list.__len__())
    y1 = Accuracy_list
    y2 = Loss_list

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
