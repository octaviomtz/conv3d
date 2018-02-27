#############
import sys
import numpy as np
import h5py
import copy
import time
import math
import random
import pickle
import pandas as pd
import multiprocessing as mp
##
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import keras.backend as K
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
##
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score
##

##############

# Variables obtained from the grid
k=int(sys.argv[-1])


archNN='A057'
cubesNumber=21
    
###########

# MAIN PROGRAM

t0 = time.time()

#getScansFold_args=[foldX,cubesNumber]


def CNNPET(archNN, cubesNumber, k, foldX):
    
    ################
    
    grid=pd.DataFrame.from_csv('theGrid3.csv')

    gridDropout1=grid['gridDropout1'][k]
    gridDropout2=grid['gridDropout2'][k]
    gridLR=grid['gridLR'][k]
    gridConvL2penal1=grid['gridConvL2penal1'][k]
    #gridConvL2penal2=grid['gridConvL2penal2'][k]
    #gridOptimizer=grid['gridOptimizer'][k]
    gridConvActivations=grid['gridConvActivations'][k]
    gridneuronsLayer1=grid['gridneuronsLayer1'][k]
    gridneuronsLayer2=grid['gridneuronsLayer2'][k]
    gridneuronsLayer3=grid['gridneuronsLayer3'][k]
    gridMiniBatch=grid['gridMiniBatch'][k]
    gridkernelsLayer1=grid['gridkernelsLayer1'][k]
    gridkernelsLayer2=grid['gridkernelsLayer2'][k]
    gridRatioBetweenKernels=grid['gridRatioBetweenKernels'][k]
    gridRatioBetweenNeurons=grid['gridRatioBetweenNeurons'][k]
    gridFullyL2penal1=grid['gridFullyL2penal1'][k]
    #gridFullyL2penal2=grid['gridFullyL2penal2'][k]
    gridMergeL2penal1=grid['gridMergeL2penal1'][k]
    gridneuronsMerge=grid['gridneuronsMerge'][k]
    gridFullyActivations=grid['gridFullyActivations'][k]
    gridNeuronsInit=grid['gridNeuronsInit'][k]
    gridConvInit=grid['gridConvInit'][k]
    gridConvBorder=grid['gridConvBorder'][k]
    gridOutL2penal=grid['gridOutL2penal'][k]
    gridOptimizers=grid['gridOptimizers'][k]

 
    optim1=Adam(lr=gridLR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-04)
    optim2=RMSprop(lr=gridLR, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizerOpts=[optim1,optim2]
    optimizer=optimizerOpts[gridOptimizers]

    gridConvL2penal2=gridConvL2penal1
    gridFullyL2penal2=gridFullyL2penal1
    #### 
    modelName='{0}_{1}'.format(archNN,k)
    

    outAct='softmax'
    Loss='categorical_crossentropy'
    
    defineModel_args=[optimizer,gridConvL2penal1,gridkernelsLayer1,gridConvInit,gridConvBorder,gridConvActivations,gridFullyL2penal1,gridneuronsLayer1,gridkernelsLayer2,gridNeuronsInit,gridDropout1,gridFullyActivations,outAct,gridOutL2penal,Loss]
    
    ############ FUNCTIONS

    def readScan(scan):
        # We read the file saved in Matlab. There is only one variable in the file called scansMini
        data = h5py.File(scan, 'r')
        Xscans=data.get('scansMini')
        # We have to get the values into the right format (subjects, dim1, dim2, dim3, channels)
        X=copy.copy(Xscans.value)
        X=np.expand_dims(X,4)
        X1=np.rollaxis(X,3)
        return X1
        
    ############

    def readLabels(labels):
        data = h5py.File(labels, 'r')
        Xscans=data.get('labels')
        X=copy.copy(Xscans.value)
        X2=np.squeeze(X).astype(int)
        X3= np.zeros((len(X2), 2))
        X3[np.arange(len(X2)), X2] = 1
        return X3
        
    ############

    def getScansFold(getScansFold_args):
        foldX,cubesNumber=getScansFold_args
        foldName='fold{0}'.format(foldX)
        XTrain=[]
        XTest=[]
        for i in (np.arange(cubesNumber)+1):
            pathNameTrain=foldName+'/train/scansMiniTrain{0}.mat'.format(i)
            pathNameTest=foldName+'/test/scansMiniTest{0}.mat'.format(i)
            xtrain=readScan(pathNameTrain)
            xtest=readScan(pathNameTest)
            XTrain.append(xtrain)
            XTest.append(xtest)
        pathNameLabelTrain=foldName+'/train/labelsTrain.mat'
        pathNameLabelTest=foldName+'/test/labelsTest.mat'
        y=readLabels(pathNameLabelTrain)
        y_true=readLabels(pathNameLabelTest)
        return XTrain, XTest, y, y_true

    #############

    def restAllModels(final_model, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21):
        final_model.reset_states()
        model1.reset_states()
        model2.reset_states()
        model3.reset_states()
        model4.reset_states()
        model5.reset_states()
        model6.reset_states()
        model7.reset_states()
        model8.reset_states()
        model9.reset_states()
        model10.reset_states()
        model11.reset_states()
        model12.reset_states()
        model13.reset_states()
        model14.reset_states()
        model15.reset_states()
        model16.reset_states()
        model17.reset_states()
        model18.reset_states()
        model19.reset_states()
        model20.reset_states()
        model21.reset_states()

    # def inputArch_4conv():
    #     modelX = Sequential()
    #     modelX.add(BatchNormalization(input_shape=(10, 10, 10 ,1)))
    #     modelX.add(Convolution3D(32,3,3,3, activation='relu'))
    #     modelX.add(Convolution3D(32,3,3,3, activation='relu'))
    #     modelX.add(MaxPooling3D())
    #     modelX.add(Convolution3D(64,3,3,3, activation='relu'))
    #     modelX.add(Convolution3D(64,3,3,3, activation='relu'))
    #     modelX.add(MaxPooling3D())
    #     modelX.add(Flatten())
    #     modelX.add(Dense(512, activation='relu'))
    #     return modelX



    def inputArch_4conv_batch():
        modelX=Sequential()
        modelX.add(BatchNormalization(input_shape=(10, 10, 10 ,1)))
        modelX.add(Convolution3D(32, 3, 3, 3, border_mode='valid', init='he_normal',W_regularizer=l2(.05)))
        modelX.add(BatchNormalization())
        modelX.add(Convolution3D(32, 3, 3, 3, border_mode='same', init='he_normal', W_regularizer=l2(.05)))
        modelX.add(MaxPooling3D())
        modelX.add(BatchNormalization())
        modelX.add(Convolution3D(64, 3, 3, 3, border_mode='same', init='he_normal', W_regularizer=l2(.05)))
        modelX.add(BatchNormalization())
        modelX.add(Convolution3D(64, 3, 3, 3, border_mode='same', init='he_normal', W_regularizer=l2(.05)))
        modelX.add(MaxPooling3D())
        modelX.add(Activation('relu'))
        modelX.add(Flatten())
        modelX.add(BatchNormalization())
        modelX.add(Dropout(0.2))
        modelX.add(Dense(512, init='glorot_normal', activation='relu', W_regularizer=l2(.05)))
        return modelX

    def inputArch_2conv_batch(defineModel_args,size):
        modelX=Sequential()
        modelX.add(BatchNormalization(input_shape=(size,size,size,1)))
        modelX.add(Convolution3D(gridkernelsLayer1,3,3,3, init=gridConvInit))
        modelX.add(BatchNormalization())
        modelX.add(Activation(gridConvActivations))
        #modelX.add(MaxPooling3D((2,2,2)))
        modelX.add(Convolution3D(gridkernelsLayer1,3,3,3, init=gridConvInit))
        modelX.add(BatchNormalization())
        modelX.add(Activation(gridConvActivations))
        #modelX.add(MaxPooling3D((2,2,2)))
        modelX.add(Flatten())
        modelX.add(Dense(gridneuronsLayer1, activation='relu'))
        modelX.add(BatchNormalization())
        return modelX

    def inputArch_3conv_batch_drop(defineModel_args,size):
        modelX=Sequential()
        modelX.add(BatchNormalization(input_shape=(size,size,size,1)))
        modelX.add(Convolution3D(gridkernelsLayer1,3,3,3, init=gridConvInit))
        modelX.add(BatchNormalization())
        modelX.add(Activation(gridConvActivations))
        #modelX.add(MaxPooling3D())
        modelX.add(Convolution3D(gridkernelsLayer1,3,3,3, init=gridConvInit))
        modelX.add(BatchNormalization())
        modelX.add(Activation(gridConvActivations))
        #modelX.add(MaxPooling3D())
        modelX.add(Convolution3D(gridkernelsLayer1,3,3,3, init=gridConvInit))
        modelX.add(BatchNormalization())
        modelX.add(Activation(gridConvActivations))
        modelX.add(MaxPooling3D())
        modelX.add(Flatten())
        modelX.add(Dense(gridneuronsLayer1, init=gridNeuronsInit, activation=gridFullyActivations, W_regularizer=l2(gridFullyL2penal1)))
        #modelX.add(Dense(gridneuronsLayer1*2))
        #modelX.add(BatchNormalization())
        #modelX.add(Activation(gridFullyActivations))
        #modelX.add(Dropout(gridDropout1))
        #modelX.add(Dense(gridneuronsLayer1*2))
        #modelX.add(BatchNormalization())
        #modelX.add(Activation(gridFullyActivations))
        #modelX.add(Dropout(gridDropout1))
        return modelX

    #def inputArch_fastai_


    def inputArch_simple(defineModel_args,size):

        #optimizer,gridConvL2penal1,gridkernelsLayer1,gridConvInit,gridConvBorder,gridConvActivations,gridFullyL2penal1,gridneuronsLayer1,gridkernelsLayer2,gridNeuronsInit,gridDropout1,gridFullyActivations,outAct,gridOutL2penal,Loss=defineModel_args

        ###########

        modelX=Sequential()
        modelX.add(Convolution3D(gridkernelsLayer1, 3, 3, 3, input_shape=(size, size, size, 1), border_mode=gridConvBorder, init=gridConvInit, W_regularizer=l2(gridConvL2penal1)))
        modelX.add(BatchNormalization())
        modelX.add(Activation(gridConvActivations))
        modelX.add(Flatten())
        modelX.add(Dropout(gridDropout1))
        modelX.add(Dense(gridneuronsLayer1, init=gridNeuronsInit, activation=gridFullyActivations, W_regularizer=l2(gridFullyL2penal1))) 
        return modelX
        
    ###########

    def defineModel(defineModel_args):
        
        #optimizer,gridConvL2penal1,gridkernelsLayer1,gridConvInit,gridConvBorder,gridConvActivations,gridFullyL2penal1,gridneuronsLayer1,gridkernelsLayer2,gridNeuronsInit,gridDropout1,gridFullyActivations,outAct,gridOutL2penal,Loss=defineModel_args

        ###########

        model1=inputArch_2conv_batch(defineModel_args,16)
        model2=inputArch_2conv_batch(defineModel_args,12)
        model3=inputArch_2conv_batch(defineModel_args,10)
        model4=inputArch_2conv_batch(defineModel_args,10)
        model5=inputArch_2conv_batch(defineModel_args,10)
        model6=inputArch_2conv_batch(defineModel_args,10)
        model7=inputArch_2conv_batch(defineModel_args,8)
        model8=inputArch_2conv_batch(defineModel_args,8)
        model9=inputArch_2conv_batch(defineModel_args,8)
        model10=inputArch_2conv_batch(defineModel_args,8)
        model11=inputArch_2conv_batch(defineModel_args,8)
        model12=inputArch_2conv_batch(defineModel_args,8)
        model13=inputArch_2conv_batch(defineModel_args,8)
        model14=inputArch_2conv_batch(defineModel_args,8)
        model15=inputArch_2conv_batch(defineModel_args,8)
        model16=inputArch_2conv_batch(defineModel_args,8)
        model17=inputArch_2conv_batch(defineModel_args,8)
        model18=inputArch_2conv_batch(defineModel_args,8)
        model19=inputArch_2conv_batch(defineModel_args,8)
        model20=inputArch_2conv_batch(defineModel_args,8)
        model21=inputArch_2conv_batch(defineModel_args,8)

        merged = Merge([model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21], mode='concat')
        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Dense(gridneuronsMerge, init=gridNeuronsInit, activation=gridFullyActivations, W_regularizer=l2(gridMergeL2penal1)))
        final_model.add(Dense(2, init=gridNeuronsInit, activation=outAct, W_regularizer=l2(gridOutL2penal)))
        final_model.compile(loss=Loss, optimizer=optimizer, metrics=['accuracy'])
        
        return final_model, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21

    def step_decay(epoch):
            initial_lrate = gridLR
            drop = 0.5
            epochs_drop = 5
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate

    ###########
        
    def trainAndTestNetwork(trainAndTestNetwork_args):
        modelName,foldX, XTrain, XTest, y, y_true, final_model, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21,gridMiniBatchTuple1,gridMiniBatchTuple2=trainAndTestNetwork_args
        #acc_train=[]
        #loss_train=[]
        #acc_val=[]
        #loss_val=[]
        #all_lr=[]
        evaluateTheBestModel=0
        
        maxAcc=0
        maxAccVal=0
        countNoIncrease=0
        minValLoss=10
        #minTrainLoss=10
        checkLossEveryXepochs=0

        #save_check_point=ModelCheckpoint('m14_weights.{epoch:02d}-{val_acc:.2f}.hdf5',monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        save_check_point=ModelCheckpoint('m_{0}_fold{1}_loss.h5'.format(modelName,foldX),monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate, save_check_point]
        
        restAllModels(final_model, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21)
        number_parameters=final_model.count_params()
        print(number_parameters)
        iteration=0
        #total_n_iterations=1
        #while iteration<=total_n_iterations:   

        history=final_model.fit(XTrain, y, validation_split=0.2593, nb_epoch=50, batch_size=gridMiniBatchTuple1,verbose=0, callbacks=callbacks_list)
        # Append values
        acc_train_=history.history['acc']
        acc_val_=history.history['val_acc']
        loss_train_=history.history['loss']
        loss_val_=history.history['val_loss']
        acc_train=np.expand_dims(acc_train_,1)
        acc_val=np.expand_dims(acc_val_,1)
        loss_train=np.expand_dims(loss_train_,1)
        loss_val=np.expand_dims(loss_val_,1)


        all_lr_=K.get_value(final_model.optimizer.lr)
        all_lr=np.repeat(all_lr_,3)


        
        minValLoss=min(loss_val)[0]
        
        #final_model.save('model_{0}_fold{1}_loss.h5'.format(modelName,foldX))
        loss_val_win=min(loss_val)[0]
        acc_val_win=max(acc_val)[0]
        nameVal_winner='{0}_fold{1}_val.pickle'.format(modelName,foldX)
        iterationVal_winner=iteration
        all_lr_win=all_lr
        countNoIncrease_win=99
        maxAcc_win=maxAcc
        iterationVal=iteration

        #iteration+=1
              
        
        ###########
        # test best model according to val_acc
        #if evaluateTheBestModel==1:
        best_model=load_model('m_{0}_fold{1}_loss.h5'.format(modelName,foldX))
        y_pred=best_model.predict_classes(XTest)
        y_predC=np.squeeze(y_pred)  
        y_predProb=final_model.predict(XTest)
        y_predP=np.squeeze(y_predProb)
        test_info=best_model.evaluate(XTest,y_true)
        test_loss=test_info[0]
        testAcc=test_info[1]
        testWrong=np.where(y_true[:,1].astype(bool)!=y_predC.astype(bool))[0]
        testMatrix=confusion_matrix(y_true[:,1],y_predC)

        ##########      
        #return outResults, wrongPredictions, confusionMatrix, predictions
        outp=[]
        outp.append(np.squeeze(acc_train))
        outp.append(np.squeeze(acc_val))
        outp.append(np.squeeze(loss_train))
        outp.append(np.squeeze(loss_val))
        outp.append(np.squeeze(all_lr))
        outp.append(iteration)
        outp.append(iterationVal)
        outp.append(loss_val_win)
        outp.append(acc_val_win)
        outp.append(testAcc)
        outp.append(test_loss)

        outp.append(foldX)
        
        outp.append(testWrong)
        outp.append(testMatrix)
        outp.append(y_predC)
        outp.append(y_predP[:,1])
        
        #return np.squeeze(acc_train), np.squeeze(acc_val), np.squeeze(loss_train), np.squeeze(loss_val)
        return outp 
    
    ################
    
    
    getScansFold_args=foldX, cubesNumber

    inputData = getScansFold(getScansFold_args) 
    modelArchitectures=defineModel(defineModel_args)
    modelInfo=tuple((modelName, foldX)) 
    gridMiniBatchTuple=tuple((gridMiniBatch,gridMiniBatch))
    trainAndTestNetwork_args=modelInfo+inputData+modelArchitectures+gridMiniBatchTuple
    NNOutput =trainAndTestNetwork(trainAndTestNetwork_args)
    return NNOutput 

pool = mp.Pool(processes=5)
outPut = [pool.apply_async(CNNPET, args=(archNN, cubesNumber, k, foldX, )) for foldX in range(1,6)]
out = [p.get() for p in outPut]
#print np.shape(out)

#outResults=[]
outResults=pd.DataFrame()
predictions=pd.DataFrame()
wrongPredictions=pd.DataFrame()

for i in range(0,5):
    
    results=out
    acc_train=results[i][0]
    acc_val=results[i][1]
    loss_train=results[i][2]
    loss_val=results[i][3]
    all_lr=results[i][4]
    iteration=results[i][5]
    iterationVal=results[i][6]
    loss_val_win=results[i][7]
    acc_val_win=results[i][8]

    testAcc=results[i][9]
    testLoss=results[i][10]
    theFold=results[i][11]
    
    testWrong=results[i][12]
    testMatrix=results[i][13]
    y_predC=results[i][14]
    y_predP=results[i][15]
    print(np.shape(results))
    df1=pd.DataFrame(acc_train,columns=['acc_train_{0}'.format(theFold)])
    df2=pd.DataFrame(acc_val,columns=['acc_val_{0}'.format(theFold)])
    df3=pd.DataFrame(loss_train,columns=['loss_train_{0}'.format(theFold)])
    df4=pd.DataFrame(loss_val,columns=['loss_val_{0}'.format(theFold)])
    df5=pd.DataFrame(all_lr,columns=['learningRate_{0}'.format(theFold)])
    df6=pd.DataFrame([iteration],columns=['total_iterations_{0}'.format(theFold)])
    df7=pd.DataFrame([iterationVal],columns=['iterationVal_{0}'.format(theFold)])
    df8=pd.DataFrame([loss_val_win],columns=['loss_val_win_{0}'.format(theFold)])
    df9=pd.DataFrame([acc_val_win],columns=['acc_val_win_{0}'.format(theFold)])
    df10=pd.DataFrame([testLoss],columns=['testLoss_{0}'.format(theFold)])
    
    df11=pd.DataFrame([testAcc],columns=['testAcc_{0}'.format(theFold)])
    
    
    wrongPredictions_=pd.DataFrame(testWrong,columns=['wrongPredictions_{0}'.format(theFold)])
    confusionMatrix=pd.DataFrame(testMatrix,columns=['0','1'])
    predictionsC=pd.DataFrame(y_predC,columns=['predClasses_{0}'.format(theFold)])
    predictionsP=pd.DataFrame(y_predP,columns=['predProba_{0}'.format(theFold)])
    
    outResults_=pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10, df11],axis=1) 
    outResults=pd.concat([outResults,outResults_],axis=1)
    
    predictions_=pd.concat([predictionsC,predictionsP],axis=1) 
    predictions=pd.concat([predictions,predictions_],axis=1)
    
    wrongPredictions=pd.concat([wrongPredictions, wrongPredictions_],axis=1)
    
modelNameOut='{0}_{1}'.format(archNN,k)   
outResults.to_csv('{0}_outResults.csv'.format(modelNameOut), index=False)
predictions.to_csv('{0}_predictions.csv'.format(modelNameOut), index=False)
wrongPredictions.to_csv('{0}_wrongPredictions.csv'.format(modelNameOut), index=False)

t1 = time.time()
print(t1-t0)
