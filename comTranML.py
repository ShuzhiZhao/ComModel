import scipy.io as scio
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model

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

def stringToFloat(sfc):
    sfc1 = []
    for ii in range(len(sfc)):
        sfc1.append(float(sfc[ii]))
    
    return sfc1

def subLabels(labels,number):
    subLabels = []
    for i in range(len(labels)):
        subLabels.append(labels[i][number-1])
#     print(np.array(subLabels).shape)
    return subLabels

def loadFC(FC_dir):
    name_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/name.txt'
    names = readScores(name_dir)
    UPDRS_dir = '/media/lhj/Momery/PD_predictDL/Data/UPDRS.txt'
    UPDRS = readScores(UPDRS_dir)
    MMSE_dir = '/media/lhj/Momery/PD_predictDL/Data/MMSE.txt'
    MMSE = readScores(UPDRS_dir)
    VoReMa_dir = '/media/lhj/Momery/PD_predictDL/Data/VoReMa.txt'
    VoReMa = readScores(VoReMa_dir)
#     print('UPDRS len ',len(UPDRS),'MMSE len ',len(MMSE),'VoReMa len ',len(VoReMa))
    DFCMs = []
    Dlabels = []
    DFCMs_dir = FC_dir+'/GretnaDFCMatrixR'
    SFCMs = []
    Slabels = []
    SFCMs_dir = FC_dir+'/GretnaSFCMatrixR'
    print('Load FC feature and scores ......')
    for file in os.listdir(DFCMs_dir):
        for ii in names:
            if ii in file:
                dfc = scio.loadmat(DFCMs_dir+'/'+file)  ## data type dict
#                 print('*********** the process '+file+' DFCM **********************')
                if len(dfc['DRStruct']) > 0:
                    if len(dfc['DRStruct'][0][0]) > 0:                        
#                         print('****** exist ******* length '+str(len(dfc['DRStruct'][0][0])))
                        for jj in range(len(dfc['DRStruct'][0][0])):
                            temp = []
                            DFCMs.append(dfc['DRStruct'][0][0][jj])
                            temp.append(UPDRS[int(ii)-1])
                            temp.append(MMSE[int(ii)-1])
                            temp.append(VoReMa[int(ii)-1])
                            Dlabels.append(temp)
    
    for file in os.listdir(SFCMs_dir):
        for ii in names:
            temp = []
            if ii in file:
                sfc = readScores(SFCMs_dir+'/'+file)  ## data type dict
                sfc = stringToFloat(sfc)  ## string of list convert to float of list
#                 print('*********** the process '+file+' DFCM **********************')
#                 print(type(np.array(sfc).reshape(116,116).tolist()[0][0]))
                SFCMs.append(np.array(sfc).reshape(116,116).tolist())
                temp.append(UPDRS[int(ii)-1])
                temp.append(MMSE[int(ii)-1])
                temp.append(VoReMa[int(ii)-1])
                Slabels.append(temp)
                            
    print('DFCMs shape:',np.array(DFCMs).shape,' Dlabels shape:',np.array(Dlabels).shape,'SFCMs shape:',np.array(SFCMs).shape,' Slabels shape:',np.array(Slabels).shape)
    
    return DFCMs,Dlabels,SFCMs,Slabels    

def loadSC(SC_dir):
    SCFAs=[]
    FaLab=[]
    SCFNs=[]
    FnLab=[]
    SClen=[]
    LenLab=[]
    SCSurf=[]
    SurfLab=[]
    SCVox=[]
    VoxLab=[]
    name_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/name.txt'
    names = readScores(name_dir)
    UPDRS_dir = '/media/lhj/Momery/PD_predictDL/Data/UPDRS.txt'
    UPDRS = readScores(UPDRS_dir)
    MMSE_dir = '/media/lhj/Momery/PD_predictDL/Data/MMSE.txt'
    MMSE = readScores(UPDRS_dir)
    VoReMa_dir = '/media/lhj/Momery/PD_predictDL/Data/VoReMa.txt'
    VoReMa = readScores(VoReMa_dir)
    print('Load SC feature and scores ......')
    for file in os.listdir(SC_dir):
        for ii in names:
            temp = []
            if ii in file:
                sfc = readScores(SC_dir+'/'+file)  ## data type dict
                sfc = stringToFloat(sfc)  ## string of list convert to float of list
                if '_FA_AAL_' in file:
#                     print(file,np.array(sfc).shape)
                    SCFAs.append(np.array(sfc).reshape(116,116).tolist())
                    temp.append(UPDRS[int(ii)-1])
                    temp.append(MMSE[int(ii)-1])
                    temp.append(VoReMa[int(ii)-1])
                    FaLab.append(temp)
                elif '_FN_AAL_' in file:
#                     print(file,np.array(sfc).shape)
                    SCFNs.append(np.array(sfc).reshape(116,116).tolist())
                    temp.append(UPDRS[int(ii)-1])
                    temp.append(MMSE[int(ii)-1])
                    temp.append(VoReMa[int(ii)-1])
                    FnLab.append(temp)
                elif '_Length_AAL_' in file:
#                     print(file,np.array(sfc).shape)
                    SClen.append(np.array(sfc).reshape(116,116).tolist())
                    temp.append(UPDRS[int(ii)-1])
                    temp.append(MMSE[int(ii)-1])
                    temp.append(VoReMa[int(ii)-1])
                    LenLab.append(temp)
                elif '_ROISurfaceSize_AAL_' in file:
#                     print(file,np.array(sfc).shape)
                    SCSurf.append(np.array(sfc).tolist())
                    temp.append(UPDRS[int(ii)-1])
                    temp.append(MMSE[int(ii)-1])
                    temp.append(VoReMa[int(ii)-1])
                    SurfLab.append(temp)
                elif '_ROIVoxelSize_AAL_' in file:
#                     print(file,np.array(sfc).shape)
                    SCVox.append(np.array(sfc).tolist())
                    temp.append(UPDRS[int(ii)-1])
                    temp.append(MMSE[int(ii)-1])
                    temp.append(VoReMa[int(ii)-1])
                    VoxLab.append(temp)
    
    print('SCFAs shape:',np.array(SCFAs).shape,' FaLab shape:',np.array(FaLab).shape,'SCFNs shape:',np.array(SCFNs).shape,' FnLab shape:',np.array(FnLab).shape,'SClen shape:',np.array(SClen).shape,' LenLab shape:',np.array(LenLab).shape,'SCSurf shape:',np.array(SCSurf).shape,' SurfLab shape:',np.array(SurfLab).shape,'SCVox shape:',np.array(SCVox).shape,' VoxLab shape:',np.array(VoxLab).shape)
    
    return SCFAs,FaLab,SCFNs,FnLab,SClen,LenLab,SCSurf,SurfLab,SCVox,VoxLab

def evalPred(pred,y_test):
    sumMAE = 0.0
    numCS2 = 0
    numCS5 = 0
    if len(pred) == len(y_test):
        for ii in range(len(pred)):
            sumMAE = sumMAE+abs(float(pred[ii])-float(y_test[ii]))
            if abs(float(pred[ii])-float(y_test[ii])) <= 2.0:
                numCS2 = numCS2+1
            if abs(float(pred[ii])-float(y_test[ii])) <= 5.0:
                numCS5 = numCS5+1
    MAE = sumMAE/len(pred)
    CS2 = numCS2/len(pred)
    CS5 = numCS5/len(pred)
    print('test dataset MAE:',MAE,' CS2:',CS2,' CS5:',CS5)
    return MAE,CS2,CS5

def model_evaluate(X_train , X_test , y_train, y_test,FTname):
    ## construct model
    # SVR
    regSVR = svm.SVR()
    regSVR.fit(X_train, y_train)
    pred = regSVR.predict(X_test )
    print('************ the model SVR result *****************')
    MAE_SVR,CS2_SVR,CS5_SVR = evalPred(pred,y_test)

    # Ridge regression
    regRidge = linear_model.Ridge(alpha=.5)
    regRidge.fit(X_train, y_train)
    pred = regRidge.predict(X_test )
    print('************ the model Ridge result *****************')
    MAE_Ridge,CS2_Ridge,CS5_Ridge = evalPred(pred,y_test)

    # Lasso
    regLasso = linear_model.Lasso(alpha=0.1)
    regLasso.fit(X_train, y_train)
    pred = regLasso.predict(X_test )
    print('************ the model Lasso result *****************')
    MAE_Lasso,CS2_Lasso,CS5_Lasso = evalPred(pred,y_test)

    # Elastic Net
    regElastic = linear_model.ElasticNet(random_state=0)
    regElastic.fit(X_train, y_train)
    pred = regElastic.predict(X_test )
    print('************ the model Elastic Net result *****************')
    MAE_Elastic,CS2_Elastic,CS5_Elastic = evalPred(pred,y_test)

    # Bayesian Regression
    regBay = linear_model.BayesianRidge()
    regBay.fit(X_train, y_train)
    pred = regBay.predict(X_test )
    print('************ the model Bayesian result *****************')
    MAE_Bay,CS2_Bay,CS5_Bay = evalPred(pred,y_test)

    # # Logistic regression
    regLogistic = linear_model.LogisticRegression(random_state=0)
    regLogistic.fit(X_train, y_train)
    pred = regLogistic.predict(X_test )
    print('************ the model Logistic result *****************')
    MAE_Logistic,CS2_Logistic,CS5_Logistic = evalPred(pred,y_test)

    # Decision Trees Regression
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test )
    print('************ the model Decision Trees result *****************')
    MAE_clf,CS2_clf,CS5_clf = evalPred(pred,y_test)

    # Forests of randomized trees
    from sklearn.ensemble import RandomForestRegressor
    regRanFor = RandomForestRegressor(random_state=1)
    regRanFor.fit(X_train, y_train)
    pred = regRanFor.predict(X_test )
    print('************ the model RandomForest result *****************')
    MAE_RanFor,CS2_RanFor,CS5_RanFor = evalPred(pred,y_test)
    
    ## save result
    import csv
    save_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat/Log'
    with open(save_dir+'/his'+FTname+'TranML.csv','w') as file:
        filedNames = ['model','test_MAE','test_CS2','test_CS5']
        writer = csv.DictWriter(file,fieldnames=filedNames)
        writer.writeheader()
        writer.writerow({'model':'SVR','test_MAE':MAE_SVR,'test_CS2':CS2_SVR,'test_CS5':CS5_SVR})
        writer.writerow({'model':'Ridge','test_MAE':MAE_Ridge,'test_CS2':CS2_Ridge,'test_CS5':CS5_Ridge})
        writer.writerow({'model':'Lasso','test_MAE':MAE_Lasso,'test_CS2':CS2_Lasso,'test_CS5':CS5_Lasso})
        writer.writerow({'model':'Elastic','test_MAE':MAE_Elastic,'test_CS2':CS2_Elastic,'test_CS5':CS5_Elastic})
        writer.writerow({'model':'Bayesian','test_MAE':MAE_Bay,'test_CS2':CS2_Bay,'test_CS5':CS5_Bay})
        writer.writerow({'model':'Logistic','test_MAE':MAE_Logistic,'test_CS2':CS2_Logistic,'test_CS5':CS5_Logistic})
        writer.writerow({'model':'DTree','test_MAE':MAE_clf,'test_CS2':CS2_clf,'test_CS5':CS5_clf})
        writer.writerow({'model':'RandomForest','test_MAE':MAE_RanFor,'test_CS2':CS2_RanFor,'test_CS5':CS5_RanFor})
            

## load data from sMRI and fMRI CN
cM_dir = '/media/lhj/Momery/PD_predictDL/Data/CmMDat'
# load FC from mat
FC_dir = cM_dir+'/FC/PD/total'
DFCMs,Dlabels,SFCMs,Slabels = loadFC(FC_dir)
# load SC from txt
SC_dir = cM_dir+'/SC/total/PD'
SCFAs,FaLab,SCFNs,FnLab,SClen,LenLab,SCSurf,SurfLab,SCVox,VoxLab = loadSC(SC_dir)

## split Train and test dataset with randomise
index = 3
task = ['UPDRS','MMSE','VoReMa']
print('++++++++++++++++++++++++++++++++++++++++++ AAL Surface with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(SCSurf),np.array(subLabels(SurfLab,index)), random_state=0)
## train and test with different models, and save result
model_evaluate(X_train , X_test , y_train, y_test,'Surface'+task[index-1])
print('++++++++++++++++++++++++++++++++++++++++++ AAL Voxel with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(SCVox),np.array(subLabels(VoxLab,index)), random_state=0)
model_evaluate(X_train , X_test , y_train, y_test,'ROIVoxel'+task[index-1])
print('++++++++++++++++++++++++++++++++++++++++++ AAL SCFA with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(SCFAs).reshape(152,13456),np.array(subLabels(FaLab,index)), random_state=0)
model_evaluate(X_train , X_test , y_train, y_test,'SCFA'+task[index-1])
print('++++++++++++++++++++++++++++++++++++++++++ AAL SCFN with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(SCFNs).reshape(152,13456),np.array(subLabels(FnLab,index)), random_state=0)
model_evaluate(X_train , X_test , y_train, y_test,'SCFN'+task[index-1])
print('++++++++++++++++++++++++++++++++++++++++++ AAL SCLength with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(SClen).reshape(152,13456),np.array(subLabels(LenLab,index)), random_state=0)
model_evaluate(X_train , X_test , y_train, y_test,'SCFN'+task[index-1])
print('++++++++++++++++++++++++++++++++++++++++++ AAL DFCMs with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(DFCMs).reshape(5465,13456),np.array(subLabels(Dlabels,index)), random_state=0)
model_evaluate(X_train , X_test , y_train, y_test,'DFCMs'+task[index-1])
print('++++++++++++++++++++++++++++++++++++++++++ AAL SFCMs with '+task[index-1]+' ++++++++++++++++++++++++++++++++++++++')
X_train , X_test , y_train, y_test = train_test_split(np.array(SFCMs).reshape(80,13456),np.array(subLabels(Slabels,index)), random_state=0)
model_evaluate(X_train , X_test , y_train, y_test,'SFCMs'+task[index-1])

