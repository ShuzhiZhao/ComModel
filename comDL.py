import os
import nibabel as nib
import numpy as np
from Model.vgg16 import VGG16
from Model.VGG3D import VGG3D
from Model.AlexNet import AlexNet
from Model.AlexNet3D import AlexNet3D
from Model.AlexNet3D1 import AlexNet3D1
from Model.resnet50 import resnet50
from Model.resnet503D import resnet503D
from Model.resnet503D1 import resnet503D1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot
import math
import time
import lmdb
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
import pandas as pd

## read txt filename
def readScores(name_dir):
    scores = []
    with open(name_dir, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
                pass
            for i in lines.split():
                scores.append(i) 
            pass 
    return scores

def loadDatVGG(dat_dir,names):
    T1 = []
    T1_label = []
    DTI = []
    DTI_label = []
    Bold = []
    Bold_label = []
#     # T1 store in lmdb
#     for file in os.listdir(dat_dir+'/T1'):
#         for ii in names:
#             if ii in file:
#                 print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#                 myFile = os.fsencode(file)
#                 myFile = myFile.decode('utf-8')
#                 myNifti = nib.load((dat_dir+'/T1/'+myFile))
#                 data = myNifti.get_fdata()
#                 data = data*(185.0/np.percentile(data, 97))
#                 if len(data.shape) == 3:
#                     # get Feature with VGG16 2D
#                     if data.shape[0] != 228 or data.shape[1] != 256 or data.shape[2] != 48:
#                         m = nn.Conv1d(data.shape[0], 228, 1, stride=1)
#                         data = m(torch.FloatTensor(data).transpose(1,0)).transpose(1,0).tolist()
#                         m = nn.Conv1d(np.array(data).shape[1], 256, 1, stride=1)
#                         data = m(torch.FloatTensor(data)).tolist()
#                         m = nn.Conv1d(np.array(data).shape[2], 48, 1, stride=1)
#                         data = m(torch.FloatTensor(data).transpose(2,1)).transpose(2,1).tolist()
#                         print(np.array(data).shape)       
#                     FT,SC = FTVGG(np.array(data),ii,'T1')
#                     T1.append(FT)
#                     T1_label.append(SC)
#     # Bold store in lmdb
#     for file in os.listdir(dat_dir+'/Bold'):
#         for ii in names:
#             if ii in file:
#                 print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#                 myFile = os.fsencode(file)
#                 myFile = myFile.decode('utf-8')
#                 myNifti = nib.load((dat_dir+'/Bold/'+myFile))
#                 data = myNifti.get_fdata()
#                 data = data*(185.0/np.percentile(data, 97))
# #                 print(data.shape)
#                 if len(data.shape) == 4:
#                     # get Feature with VGG16 3D      
#                     FT,SC = FTVGG3D(np.array(data),ii,'Bold')
#                     Bold.append(FT)
#                     Bold_label.append(SC) 
#     # DTI store in lmdb
#     for file in os.listdir(dat_dir+'/DTI'):
#         for ii in names:
#             if ii in file:
#                 print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#                 myFile = os.fsencode(file)
#                 myFile = myFile.decode('utf-8')
#                 myNifti = nib.load((dat_dir+'/DTI/'+myFile))
#                 data = myNifti.get_fdata()
#                 data = data*(185.0/np.percentile(data, 97))
#                 if len(data.shape) == 4:
#                     # get Feature with VGG16 3D      
#                     FT,SC = FTVGG3D(np.array(data),ii,'DTI')
#                     DTI.append(FT)
#                     DTI_label.append(SC)
               
    # get all FT and labels
    FTT1_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/FT_VGG2DT1'
    T1,T1_label = getFT(FTT1_dir,names,'T1')
    FTDTI_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/FT_VGG3DDTI'
    DTI,DTI_label = getFT(FTDTI_dir,names,'DTI')
    FTBold_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/FT_VGG3DBold'
    Bold,Bold_label = getFT(FTBold_dir,names,'Bold')    
    
    return T1,T1_label,DTI,DTI_label,Bold,Bold_label

def getFT(FTT1_dir,names,Dty):
    FT = []
    label = []
    lmdb_env = lmdb.open(FTT1_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8') 
            for ii in names:
                if ii in key :                    
                    if Dty in key:
                        if 'T1' in Dty :
                            if  np.frombuffer(value).shape[0] != 4096 :
                                print('++++++++++++++++++++++ ',key,' ++++++++++++++++++++++++++')
                                print('data size:',np.frombuffer(value).shape)
                            else:
                                FT.append(np.frombuffer(value).tolist())
                        if 'DTI' in Dty or 'Bold' in Dty :
                            if np.frombuffer(value).shape[0] != 18432 :
                                print('++++++++++++++++++++++ ',key,' ++++++++++++++++++++++++++')
                                print('data size:',np.frombuffer(value).shape)
                            else:
                                FT.append(np.frombuffer(value).tolist())
                    elif '_score' in key:
                        label.append(np.frombuffer(value).tolist())
    print(Dty+' FT:',np.array(FT, dtype=object).shape,' label:',np.array(label, dtype=object).shape)
    return FT,label

def FTVGG(data,ii,Dty):
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG16(device)
    model = model.cuda()
    model.eval()
    SC_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/MRIScore.txt'
    Scores = readScores(SC_dir)
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat'
    env = lmdb.open(lmdb_dir+"/FT_VGG2D"+Dty, map_size = int(1e12)*2)
    txn = env.begin(write=True)        
    print("++++++++++++++++++++++ The process of "+ii+" ++++++++++++++++++++++")
    print('store score in lmdb:',float(Scores[int(ii)-1]))
    txn.put((ii+'_score').encode(), np.array(float(Scores[int(ii)-1])))      
    FT = []
    inputs = torch.FloatTensor(data.reshape(1,data.shape[0],data.shape[1],data.shape[2])).to(device)
#     print(inputs.size())
    print('input size:',inputs.size())
    FT = model(inputs).tolist()
    txn.put((ii+'_'+Dty).encode(), np.array(FT))    
    print('ouput FT size:',np.array(FT).shape)
    txn.commit() 
    env.close()
    for ii in range(5):
        torch.cuda.empty_cache()
        
    return FT,Scores[int(ii)-1]

def FTVGG3D(data,ii,Dty):
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG3D(device)
    model = model.cuda()
    model.eval()
    SC_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/MRIScore.txt'
    Scores = readScores(SC_dir)
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat'
    env = lmdb.open(lmdb_dir+"/FT_VGG3D"+Dty, map_size = int(1e12)*2)
    txn = env.begin(write=True)        
    print("++++++++++++++++++++++ The process of "+ii+" ++++++++++++++++++++++")
    print('store score in lmdb:',float(Scores[int(ii)-1]))
    txn.put((ii+'_score').encode(), np.array(float(Scores[int(ii)-1])))      
    FT = []
    inputs = torch.FloatTensor(data.reshape(1,data.shape[0],data.shape[1],data.shape[2],data.shape[3])).to(device)
#     print(inputs.size())
    print('input size:',inputs.size())
    FT = model(inputs).tolist()
    txn.put((ii+'_'+Dty).encode(), np.array(FT))    
    print('ouput FT size:',np.array(FT).shape)
    txn.commit() 
    env.close()
    for ii in range(5):
        torch.cuda.empty_cache()
#     print(torch.cuda.memory_summary())
    
    return FT,Scores[int(ii)-1]

def loadDat(model1,model2,model3,dat_dir,names,Dty):
    T1 = []
    T1_label = []
    DTI = []
    DTI_label = []
    Bold = []
    Bold_label = []
    # T1 store in lmdb
#     for file in os.listdir(dat_dir+'/T1'):
#         for ii in names:
#             if ii in file:
#                 print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#                 myFile = os.fsencode(file)
#                 myFile = myFile.decode('utf-8')
#                 myNifti = nib.load((dat_dir+'/T1/'+myFile))
#                 data = myNifti.get_fdata()
#                 data = data*(185.0/np.percentile(data, 97))
#                 if len(data.shape) == 3:
#                     # get Feature with VGG16 2D
#                     if data.shape[0] != 228 or data.shape[1] != 256 or data.shape[2] != 48:
#                         m = nn.Conv1d(data.shape[0], 228, 1, stride=1)
#                         data = m(torch.FloatTensor(data).transpose(1,0)).transpose(1,0).tolist()
#                         m = nn.Conv1d(np.array(data).shape[1], 256, 1, stride=1)
#                         data = m(torch.FloatTensor(data)).tolist()
#                         m = nn.Conv1d(np.array(data).shape[2], 48, 1, stride=1)
#                         data = m(torch.FloatTensor(data).transpose(2,1)).transpose(2,1).tolist()
#                         print(np.array(data).shape)       
#                     FT,SC = FTM1(model1,np.array(data),ii,Dty+'T1')
#                     T1.append(FT)
#                     T1_label.append(SC)
#     # Bold store in lmdb
#     for file in os.listdir(dat_dir+'/Bold'):
#         for ii in names:
#             if ii in file:
#                 print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#                 myFile = os.fsencode(file)
#                 myFile = myFile.decode('utf-8')
#                 myNifti = nib.load((dat_dir+'/Bold/'+myFile))
#                 data = myNifti.get_fdata()
#                 data = data*(185.0/np.percentile(data, 97))
# #                 print(data.shape)
#                 if len(data.shape) == 4:
#                     # get Feature with VGG16 3D      
#                     FT,SC = FTM2(model3,np.array(data),ii,Dty+'Bold')
#                     Bold.append(FT)
#                     Bold_label.append(SC) 
#     # DTI store in lmdb
#     for file in os.listdir(dat_dir+'/DTI'):
#         for ii in names:
#             if ii in file:
#                 print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#                 myFile = os.fsencode(file)
#                 myFile = myFile.decode('utf-8')
#                 myNifti = nib.load((dat_dir+'/DTI/'+myFile))
#                 data = myNifti.get_fdata()
#                 data = data*(185.0/np.percentile(data, 97))
#                 if len(data.shape) == 4:
#                     # get Feature with VGG16 3D      
#                     FT,SC = FTM2(model2,np.array(data),ii,Dty+'DTI')
#                     DTI.append(FT)
#                     DTI_label.append(SC)
               
    # get all FT and labels
    numFT = [1000,1000]
    FTT1_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/FT_2D'+Dty+'T1'
    T1,T1_label = getFTM(FTT1_dir,names,'T1',numFT)
    FTDTI_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/FT_3D'+Dty+'DTI'
    DTI,DTI_label = getFTM(FTDTI_dir,names,'DTI',numFT)
    FTBold_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/FT_3D'+Dty+'Bold'
    Bold,Bold_label = getFTM(FTBold_dir,names,'Bold',numFT)    
    
    return T1,T1_label,DTI,DTI_label,Bold,Bold_label

def FTM1(model1,data,ii,Dty):
    ## model
    model =model1
    SC_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/MRIScore.txt'
    Scores = readScores(SC_dir)
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat'
    env = lmdb.open(lmdb_dir+"/FT_2D"+Dty, map_size = int(1e12)*2)
    txn = env.begin(write=True)        
    print("++++++++++++++++++++++ The process of "+ii+" ++++++++++++++++++++++")
    print('store score in lmdb:',float(Scores[int(ii)-1]))
    txn.put((ii+'_score').encode(), np.array(float(Scores[int(ii)-1])))      
    FT = []
    inputs = torch.FloatTensor(data.reshape(1,data.shape[0],data.shape[1],data.shape[2])).to(device)
#     print(inputs.size())
    print('input size:',inputs.size())
    FT = model(inputs).tolist()
    txn.put((ii+'_'+Dty).encode(), np.array(FT))    
    print('ouput FT size:',np.array(FT).shape)
    txn.commit() 
    env.close()
    for ii in range(5):
        torch.cuda.empty_cache()
        
    return FT,Scores[int(ii)-1]

def FTM2(model2,data,ii,Dty):
    model = model2
    SC_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/MRIScore.txt'
    Scores = readScores(SC_dir)
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat'
    env = lmdb.open(lmdb_dir+"/FT_3D"+Dty, map_size = int(1e12)*2)
    txn = env.begin(write=True)        
    print("++++++++++++++++++++++ The process of "+ii+" ++++++++++++++++++++++")
    print('store score in lmdb:',float(Scores[int(ii)-1]))
    txn.put((ii+'_score').encode(), np.array(float(Scores[int(ii)-1])))      
    FT = []
    inputs = torch.FloatTensor(data.reshape(1,data.shape[0],data.shape[1],data.shape[2],data.shape[3])).to(device)
#     print(inputs.size())
    print('input size:',inputs.size())
    FT = model(inputs).tolist()
    txn.put((ii+'_'+Dty).encode(), np.array(FT))    
    print('ouput FT size:',np.array(FT).shape)
    txn.commit() 
    env.close()
    for ii in range(5):
        torch.cuda.empty_cache()
#     print(torch.cuda.memory_summary())
    
    return FT,Scores[int(ii)-1]

def getFTM(FT_dir,names,Dty,numFT):
    FT = []
    label = []
    lmdb_env = lmdb.open(FT_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8') 
            for ii in names:
                if ii in key :                    
                    if Dty in key:
                        if 'T1' in Dty :
                            if  np.frombuffer(value).shape[0] != numFT[0] :
                                print('++++++++++++++++++++++ ',key,' ++++++++++++++++++++++++++')
                                print('data size:',np.frombuffer(value).shape)
                            else:
                                FT.append(np.frombuffer(value).tolist())
                        if 'DTI' in Dty or 'Bold' in Dty :
                            if np.frombuffer(value).shape[0] != numFT[1] :
                                print('++++++++++++++++++++++ ',key,' ++++++++++++++++++++++++++')
                                print('data size:',np.frombuffer(value).shape)
                            else:
                                FT.append(np.frombuffer(value).tolist())
                    elif '_score' in key:
                        label.append(np.frombuffer(value).tolist())
    print(Dty+' FT:',np.array(FT, dtype=object).shape,' label:',np.array(label, dtype=object).shape)
    return FT,label

##load data from array
class ERP_matrix_datasets(Dataset):
    ##build a new class for own dataset
    import numpy as np
    def __init__(self, fmri_data_matrix, label_matrix,
                 isTrain='train', transform=False):
        super(ERP_matrix_datasets, self).__init__()

        if not isinstance(fmri_data_matrix, np.ndarray):
            self.fmri_data_matrix = np.array(fmri_data_matrix)
        else:
            self.fmri_data_matrix = fmri_data_matrix
        
        self.Subject_Num = self.fmri_data_matrix.shape[0]
        self.Region_Num = self.fmri_data_matrix[0].shape[-1]

        if isinstance(label_matrix, pd.DataFrame):
            self.label_matrix = label_matrix
        elif isinstance(label_matrix, np.ndarray):
            self.label_matrix = pd.DataFrame(data=np.array(label_matrix))

        self.data_type = isTrain
        self.transform = transform

    def __len__(self):
        return self.Subject_Num

    def __getitem__(self, idx):
        #step1: get one subject data
        fmri_trial_data = self.fmri_data_matrix[idx]
        fmri_trial_data = fmri_trial_data.reshape(1,fmri_trial_data.shape[0])
        label_trial_data = np.array(self.label_matrix.iloc[idx])
#         print('fmri_trial_data\n{}\n======\nlabel_trial_data\n{}\n'.format(fmri_trial_data.shape,label_trial_data.shape))
        tensor_x = torch.stack([torch.FloatTensor(fmri_trial_data[ii]) for ii in range(len(fmri_trial_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.LongTensor([label_trial_data[ii]]) for ii in range(len(label_trial_data))])
#         print('tensor_x\n{}\n=======\ntensor_y\n{}\n'.format(tensor_x.size(),tensor_y.size()))
        return tensor_x, tensor_y

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.hidden(x)))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

def predLinDL(FT,scores,modName):
    import csv
    index = 2
    nameLabel = ['age', 'UPDRS', 'MoCA']
    model_test = Net(n_feature=np.array(FT).shape[1], n_hidden=200, n_output=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_test = model_test.to(device)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    loss_func = nn.L1Loss(reduction='mean')
    num_epochs = 10
    trainData,trainLabels,testData,testLabels = crossDataLabel(FT,scores)
    print('******************************************'+nameLabel[index-1]+'************************************************')
    model_history = model_fit_evaluate(model_test,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs)
    Res_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/Log/'
    with open(Res_dir+'/his'+modName+'.csv','w') as file:
        filedNames = ['model','train_MAE','train_CS2','train_CS5','test_MAE','test_CS2','test_CS5']
        writer = csv.DictWriter(file,fieldnames=filedNames)
        writer.writeheader()
        for ii in range(num_epochs):
            writer.writerow({'model':modName,'train_MAE':model_history['train_MAE'][ii].tolist(),'train_CS2':model_history['train_CS2'][ii],'train_CS5':model_history['train_CS5'][ii],'test_MAE':model_history['test_MAE'][ii].tolist(),'test_CS2':model_history['test_CS2'][ii],'test_CS5':model_history['test_CS5'][ii]})

def crossDataLabel(FT,labels):
    ## 5 cross valition
    test_size = 0.2
    randomseed=1234
    test_sub_num = len(FT)
    print('test_sub_num: ',test_sub_num)
    rs = np.random.RandomState(randomseed)
    train_sid, test_sid = train_test_split(range(test_sub_num), test_size=test_size, random_state=rs, shuffle=True)
    print('training on %d subjects, validating on %d subjects' % (len(train_sid), len(test_sid)))
    ####train set 
    fmri_data_train = [FT[i] for i in train_sid]
    trainLabels = pd.DataFrame(np.array([labels[i] for i in train_sid]))
#     print(type(trainLabels),'\n',trainLabels)
    ERP_train_dataset = ERP_matrix_datasets(fmri_data_train, trainLabels, isTrain='train')
    trainData = DataLoader(ERP_train_dataset)

    ####test set
    fmri_data_test = [FT[i] for i in test_sid]
    testLabels = pd.DataFrame(np.array([labels[i] for i in test_sid]))
#     print(type(testLabels),'\n',testLabels)
    ERP_test_dataset = ERP_matrix_datasets(fmri_data_test, testLabels, isTrain='test')
    testData = DataLoader(ERP_test_dataset)
    
    return trainData,trainLabels,testData,testLabels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
## train and test model
def model_fit_evaluate(model,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    model_history={}
    model_history['train_MAE']=[];
    model_history['train_CS2']=[];
    model_history['train_CS5']=[];
    model_history['test_MAE']=[];
    model_history['test_CS2']=[];
    model_history['test_CS5']=[];

    for epoch in range(num_epochs):
        train_MAE,train_CS2,train_CS5 =train(model, device, trainData, trainLabels, optimizer,loss_func, epoch)
        model_history['train_MAE'].append(train_MAE)
        model_history['train_CS2'].append(train_CS2)
        model_history['train_CS5'].append(train_CS5)

        test_MAE,test_CS2,test_CS5 = test(model, device, testData, testLabels, loss_func)
        model_history['test_MAE'].append(test_MAE)
        model_history['test_CS2'].append(test_CS2)
        model_history['test_CS5'].append(test_CS5)
        if test_CS5 > best_acc:
            best_acc = test_CS5
            print("Model updated: Best-Acc = {:4f}".format(best_acc))

    print("best testing accuarcy:",best_acc)
    
    return model_history
    
##training the model
def train(model, device,train_loader, trainLabels, optimizer,loss_func, epoch):
    model.train()

    MAE = 0.0
    CS2 = 0.0
    CS5 = 0.0
    t0 = time.time()
    Predict_Scores = []
    True_Scores = []
    L1_MAE = []
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.squeeze(0)
        target = target.view(-1).float()
#         print('inputs ',data.size(),'labels:',target.size())
#         print('inputs ',data,'labels:',target)
        out = model(data)
        Predict_Scores.append(out),True_Scores.append(target)
        loss = loss_func(out,target)
        L1_MAE.append(loss)
        
        loss.backward()
        optimizer.step()
    num2,CS2 = LowerCount(L1_MAE,2)
    num5,CS5 = LowerCount(L1_MAE,5)
    MAE = sum(L1_MAE)/len(L1_MAE)   
    print("\nEpoch {}: \nTime Usage:{:4f} | Training MAE {:4f} | CS2 {:4f} | CS5 {:4f}".format(epoch,time.time()-t0,MAE,CS2,CS5))
    return MAE,CS2,CS5

def test(model, device, test_loader, testLabels, loss_func):
    model.eval()
    MAE = 0.0
    CS2 = 0.0
    CS5 = 0.0 
    ##no gradient desend for testing
    with torch.no_grad():
        Predict_Scores = []
        True_Scores = []
        L1_MAE = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
#             print('input:',target,'predict:',out)
            Predict_Scores.append(out.tolist()[0][0][0]+np.random.uniform(-2,2)),True_Scores.append(target.tolist()[0][0][0]+np.random.uniform(-2,2))
            
            loss = loss_func(out,target)
            L1_MAE.append(loss)
#             print('MAE:',loss)
        num2,CS2 = LowerCount(L1_MAE,2)
        num5,CS5 = LowerCount(L1_MAE,5)
        MAE = sum(L1_MAE)/len(L1_MAE)
#         plotScatter(Predict_Scores,True_Scores) 
        
    return MAE,CS2,CS5    

def LowerCount(a,b):
    num = 0
    for i in a:
        if i<b:
            num+=1
    percent = num/len(a)
    return num,percent
    
## load dataset from PPMI
dat_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI'
name_dir = dat_dir+'/name_Bold.txt'
names = readScores(name_dir)
# one subject of T1,DTI and Bold
# T1,T1_label,DTI,DTI_label,Bold,Bold_label = loadDatVGG(dat_dir,names)

## Deep learning Model with 2D or 3D convolution
# VGG16 with 2D convolution    
# predLinDL(T1,T1_label,'VGG2DT1')
# VGG16 with 3D convolution
# predLinDL(DTI,DTI_label,'VGG3DDTI')
# predLinDL(Bold,Bold_label,'VGG3DBold')

# AlexNet with 2D convolution
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model1 = AlexNet(device)
# model1 = model1.cuda()
# model1.eval()
# model2 = AlexNet3D(device)
# model2 = model2.cuda()
# model2.eval()
# model3 = AlexNet3D1(device)
# model3 = model3.cuda()
# model3.eval()
# T1,T1_label,DTI,DTI_label,Bold,Bold_label = loadDat(model1,model2,model3,dat_dir,names,'AlexNet')
# predLinDL(T1,T1_label,'AlexNet2DT1')
# # AlexNet with 3D convolution
# predLinDL(DTI,DTI_label,'AlexNet3DDTI')
# predLinDL(Bold,Bold_label,'AlexNet3DBold')

# ResNet50 with 2D convolution
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model1=resnet50().to(device)
summary(model1,(228,256,48))
model1 = model1.cuda()
model1.eval()
model2 = resnet503D().to(device)
summary(model2,(116,116,72,65))
model2 = model2.cuda()
model2.eval()
model3 = resnet503D1().to(device)
summary(model3,(68,66,40,210))
model3 = model3.cuda()
model3.eval()
T1,T1_label,DTI,DTI_label,Bold,Bold_label = loadDat(model1,model2,model3,dat_dir,names,'ResNet50')
predLinDL(T1,T1_label,'ResNet502DT1')
# ResNet50 with 3D convolution
predLinDL(DTI,DTI_label,'ResNet503DDTI')
predLinDL(Bold,Bold_label,'ResNet503DBold')

# DeseNet121 with 2D convolution

# DeseNet121 with 3D convolution