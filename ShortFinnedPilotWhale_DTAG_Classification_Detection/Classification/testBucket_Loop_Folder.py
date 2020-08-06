#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get STFT .npy files of buzzes and minibuzzes from AWS S3 storage bucket
from sagemaker import get_execution_role
import io
import boto3, botocore
print('finished imports')


# In[2]:


bucket = 'pilotwhales2' # Use the name of your s3 bucket here
#open file (lists all train .npy STFT file names) 
file = open('newTestList.txt', 'r')
data = file.read().split('\n')
file.close()
print(data)


# In[3]:


pilotwhales = boto3.resource('s3').Bucket(bucket) # S3 bucket object
savefolder = 'aws_pilotwhales2_08_10_11_testData/' # folder in SageMaker that things get saved to
print('folders good')


# In[4]:


#for loop through all .npy STFT files listed in data
for array in data:
    data_key = 'TestList/'+array # name of file
    print('able to find file')
    # try/catch block is for file-not-found exception
    #lots of errors with try catch? Anyways, all 676 files are now in folder
    #not an error with try/catch, still get errors when running this, but still have all files 
    #try:
    pilotwhales.download_file(data_key, savefolder+array)
    #except botocore.exceptions.ClientError as e:
        #if e.response['Error']['Code'] == "404":
            #print("The object does not exist.")
        #else:
            #raise


# In[ ]:




