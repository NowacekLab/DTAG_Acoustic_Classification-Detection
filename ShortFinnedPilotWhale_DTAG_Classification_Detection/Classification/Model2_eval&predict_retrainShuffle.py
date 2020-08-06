#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import things to use
import os
import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Activation, Masking, Dense 
from keras.layers import Convolution2D as Conv2D 
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D 
from keras.layers import Softmax, ReLU, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn import metrics 
from keras.optimizers import SGD
print('finished imports!')


# In[2]:


#?
from keras.backend.tensorflow_backend import set_session
config = tensorflow.ConfigProto()
config.intra_op_parallelism_threads = 1
print(config.intra_op_parallelism_threads)
set_session(tensorflow.Session(config=config))


# In[3]:


def find_max(list_path, list_name, files_path): 
    #change to the directory where list of STFT files is 
    os.chdir(list_path)
    #loop through to find maxTime, Don't need to save anything else though
    maxTime = 0
    print('made variable maxTime')

    #open file (lists all train .npy STFT file names) 
    file = open(list_name, 'r') 
    data = file.read().split('\n')
    file.close()
    print('got data files in a list')
    #print(data)

    os.chdir(files_path)
    print('changed to files folder')

    for array in data:
        #find cols (number of time steps) of each STFT and save longest one
        curSTFT = np.load(array)
        rows, cols = curSTFT.shape
        if (cols>maxTime):
            maxTime = cols
        if (rows!=1025):
            print('Error, not 1025 STFT coefficients') 
    
    return maxTime 

trainingMax = find_max('/home/ec2-user/SageMaker','newTrainListLesser.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_08_10_11_trainData')
print(trainingMax)
print('found trainingMax')


# In[4]:


def get_labels_and_samples(path, file, datapath, maxTime):
    #change to the directory where list of STFT files is 
    os.chdir(path)
    
    labels = []
    paddedSTFTs = []
    print('made variables')

    #open file (lists all train .npy STFT file names) 
    file = open(file, 'r') #note, ran with 169 in TopList, could not handle all 676
    data = file.read().split('\n')
    file.close()
    print('got data files in a list')
    print(data)

    os.chdir(datapath)
    print('changed to files folder')

    for array in data:
        #loop through to find out if file is a buzz or minibuzz, and add label accordingly 
        nameParse = array.split("_",-1)
        #print(nameParse)
        typeParse = nameParse[4].split("u", -1)
        #print(typeParse)
        if(typeParse[0] == 'b'):
            #buzz = 1 
            labels.append(1)
        elif(typeParse[0] == 'minib'):
            #minibuzz = 0
            labels.append(0)
        else: 
            print('Error, not a buzz or minibuzz!')

    print(labels) 
    print('label loop done')

    for array in data: 
        #loop though STFTs again to zero pad, transpose, and reshape soo that there is one channel  
        #NOTE: must be done after we definitvely know the max number of time steps
        curSTFT = np.load(array)
        rows, cols = curSTFT.shape

        zeroPad = np.zeros((rows,maxTime-cols))
        paddedSTFT = np.append(curSTFT, zeroPad, axis = 1)

        paddedSTFT = np.transpose(paddedSTFT)
        paddedSTFT = np.reshape(paddedSTFT, paddedSTFT.shape + (1,))

        paddedSTFTs.append(paddedSTFT)

    print('sample loop done')

    labels = np.array(labels)
    #one hot encoding for labels
    hotlabels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    samples = np.array(paddedSTFTs)
    print('hotlabels and samples in np arrays')
    return hotlabels, samples

print('done with get_labels_get_samples')


# In[ ]:





# In[5]:


def data_generator(path, file, datapath, bs, maxTime, mode='train'):
    # open the text file for reading
    os.chdir(path)
    f = open(file, 'r')
    while True: 
        
        # initialize our batches of images and labels
        samples = []
        labels = []
    
        # keep looping until we reach our batch size
        while len(samples) < bs:
            # attempt to read the next line of the text file
            line = f.readline()
            
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()
                # if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
                if mode == "eval":
                    break
    
            # construct list of labels: find out if file is a buzz or minibuzz, and add label accordingly 
            nameParse = line.split('_',-1)
            #print(nameParse)
            typeParse = nameParse[4].split('u', -1)
            #print(typeParse)
            if(typeParse[0] == 'b'):
                #buzz = 1 
                labels.append(1)
            elif(typeParse[0] == 'minib'):
                #minibuzz = 0
                labels.append(0)
            else: 
                print('Error, not a buzz or minibuzz!')
    
            #switch to aws_pilotwhales2 folder
            os.chdir(datapath)
            #print('changed to aws_pilotwhales2 folder')
            
            #construct list of samples
            lineParse = line.split("\n",-1)
            #print(lineParse)
            #print(os.getcwd())
            curSTFT = np.load(lineParse[0])
            rows, cols = curSTFT.shape
    
            zeroPad = np.zeros((rows,maxTime-cols))
            paddedSTFT = np.append(curSTFT, zeroPad, axis = 1)
    
            paddedSTFT = np.transpose(paddedSTFT)
            paddedSTFT = np.reshape(paddedSTFT, paddedSTFT.shape + (1,))
    
            samples.append(paddedSTFT)
        
        #convert from lists to numpy arrays
        labels = np.array(labels)
        #one hot encoding for labels
        hotlabels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
        samples = np.array(samples)
        #print('hotlabels and samples in np arrays')
    
        # yield the batch to the calling function
        if mode == "predict":
            yield(samples)
        else: 
            yield (samples, hotlabels)
        
# initialize both the training and validation generators
trainGen = data_generator('/home/ec2-user/SageMaker', 'newTrainOnlyListLesser.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_08_10_11_trainData', 13, trainingMax, mode = 'train')
print('generated training set')
validGen = data_generator('/home/ec2-user/SageMaker', 'newValidListLesser.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_08_10_11_trainData', 13, trainingMax, mode = 'train')
print('generated vaildation set')


# In[6]:


#MAKE MODEL AND COMPILE IT

#layer 0: input
#labels[], samples[] (maxTime rows, 1025 cols) each in trainGen and validGen
print('ready to start model')

# build model
model = Sequential()
print('made model')

#NOTE, CHANGED PADDING ON 2D CONVOLUTIONS from 'valid'=no paddinng to 'same'=padding so input and output are same dimensions

#layer 1: 2D convolution between input and 256 filters with 1 row and 1025 cols
print(trainingMax)
model.add(Conv2D(256, input_shape = [trainingMax,1025,1], kernel_size = [1,1025], strides=(1, 1), padding='valid', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#batch normalization- add in layer? don't understand parameters well
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 1 done')

##layer 2: 2D convolution between output of layer 1 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,1], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 2 done')

#layer 3: 2D convolution between output of layer 2 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,1], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 3 done')

#layer 4: 2D convolution between output of layer 3 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,1], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 4 done')

#layer 5: Global max pooling
model.add(GlobalMaxPooling2D(data_format="channels_last"))
print('layer 5 done')

#layer 6: fully connected layer
model.add(Dense(2, activation='softmax', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
print('layer 6 done')

#Compile model [COMPILE]
#OLD COMPILE (for fit, not fit_generator)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
opt = SGD(lr=0.02) #note, can play with leraning rate and other parameters here
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=["accuracy"])
print('compiled')

print(model.summary())


# In[7]:


#Now let us train our model [FIT]
ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=True)

#with fit_generator
print("[INFO] training w/ generator...")
model.fit_generator(trainGen, steps_per_epoch=42, epochs=5, verbose=2, callbacks=[ES], validation_data=validGen, validation_steps=10, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
print('ran fit_generator')

#with fit
#model.fit(x=samples, y=hotlabels, batch_size=13, epochs=5, verbose=2, callbacks=[ES], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
#print('ran fit')


# In[8]:


#Prepare test data for evaluation

#NOTE NEED TO USE TRAINING MAX TO BE CONSISTENT WITH THE NEURAL NETWORK
#find max length (time) in testing data
#testingMax = find_max('/home/ec2-user/SageMaker','testList.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_testData')
#maxTime = testingMax
#print(testingMax)
#print('found testingMax')

# initialize testing generator
testGen = data_generator('/home/ec2-user/SageMaker', 'newTestListLesser.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_08_10_11_testData', 13, trainingMax, mode = "eval")
print('generated testing set')


# In[9]:


#Now let us evaluate our model [EVALUATE]

#with evaluate_generator
results = model.evaluate_generator(testGen, steps = 13, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
print(results)
print(model.metrics_names)

#with evauluate
#results = model.evaluate(x=testSamples, y=testHotlabels, batch_size=26, verbose=1, sample_weight=None, steps=None)


# In[10]:


#HOLD OFF ON 
#[PREDICT] (w/TestData I kept aside as well)

#predictGen = data_generator('/home/ec2-user/SageMaker', 'testList.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_testData', 13, trainingMax, mode = "eval")
#print('generated predict set')

#with predict_generator
#need a different generator(just grab batches of data but do not know answer)
#predictions = model.predict_generator(testGen, steps=13, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
#print(predictions)

#with predict
#preprocess STFT data in TestData folder!
#see getLabels_stackedData3.py, but don't give labels
hotlabels, samples = get_labels_and_samples('/home/ec2-user/SageMaker', 'testList.txt', '/home/ec2-user/SageMaker/aws_pilotwhales2_testData', trainingMax)
hotlabels_pred = model.predict(samples, batch_size=13, verbose=1, steps=None)


# In[ ]:


#HOLD OFF ON 
#see if prediction results are right (compare hot labels I generate with what the model guesses)
print(hotlabels)
hotlabels = keras.utils.to_categorical(hotlabels, num_classes=2, dtype='float32')
print(hotlabels)
print(hotlabels_pred)
print((np.max(abs(hotlabels-hotlabels_pred),axis=1)>.5))


# In[23]:


# save the model
# uncomment the lines below when you have a model you're happy with
import h5py
os.chdir('/home/ec2-user/SageMaker/')
# model.save('Model2_eval+predict_retrainShuffle_2019-05-03.h5')
# model.save_weights('Model2_eval+predict_retrainShuffle_weights_2019-05-03.h5')


# In[18]:





# In[19]:





# In[ ]:




