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
    val_dataset = 'Test1000'
    output_units=21
    print("Load test data.")
    val_data_spa, val_label = get_val_data_spa(val_dataset)
    val_data_spe, val_label = get_val_data_spe(val_dataset)
    # val_data_spa=np.transpose(val_data_spa,(0,3,1,2)) #不是7.3mat存储格式
    print("Validation data reading is completed.")

    print("Load model weights")
    mod_name = "mod_Train3000_(300)_24_AdamW0.0001_0.9_batch500_13_0.991"
    save_path = "./model/" + mod_name + ".pth"
    net = MOD_WK(output_units)

    net.load_state_dict(torch.load(save_path))
    net.eval()

    valdata=MyDataset(val_data_spa,val_data_spe,val_label)
    testnum=5000
    val_loader = torch.utils.data.DataLoader(valdata,batch_size=testnum,
                                               shuffle=False, num_workers=0)

    Accuracy_list = []

    with torch.no_grad():
        for step, data in enumerate(val_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs_spa, inputs_spe, labels = data
            inputs_spa = inputs_spa.type(torch.FloatTensor)
            inputs_spe = inputs_spe.type(torch.FloatTensor)
            labels = torch.tensor(labels, dtype=torch.long)
            outputs = net(inputs_spa, inputs_spe)
            predict_y = torch.max(outputs, dim=1)[1]
            #  查看预测结果
            if step == 0:
                predict = torch.max(outputs, dim=1)[1].data.numpy()
                pro = np.amax(torch.softmax(outputs, dim=1).data.numpy(), axis=1)  # 查看预测概率值
                accuracy = torch.eq(predict_y, labels).sum().item() / labels.size(0)
                print(accuracy)

            else:
                predict = np.concatenate((predict, torch.max(outputs, dim=1)[1].data.numpy()), axis=0)
                pro = np.concatenate((pro, np.amax(torch.softmax(outputs, dim=1).data.numpy(), axis=1)), axis=0)
                accuracy = torch.eq(predict_y, labels).sum().item() / labels.size(0)
                print(accuracy)

    from confusion import get_confusion
    save_name = mod_name

    sio.savemat("./result/result_"+mod_name+".mat", {'out': predict})
    file_name = "./result/report-" + save_name + ".txt"
    csv_name = "./result/report-" + save_name + ".csv"
    get_confusion(val_label, predict, file_name,csv_name);
    end = time.time()
    print(end - start)
    print('Finished Training')



if __name__ == '__main__':
    print(torch.__version__)
    main()
