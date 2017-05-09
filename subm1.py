# Imports

import os, sys, pickle, math
import numpy as np
import pandas as pd
from glob import glob
from shutil import copyfile
from PIL import ImageFile


from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import log_loss
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3

# Paths

cur_dir=os.getcwd()

data_dir=cur_dir+'/data/'
#data_dir=cur_dir+'/data/sample/'
test_dir=cur_dir+'/data/test/'
local_model_path=os.path.dirname(cur_dir)+'/models/'

train_dir=data_dir+'train/'
valid_dir=data_dir+'valid/'
save_dir=data_dir+'save/'

type1_dir=train_dir+'Type_1/'
type2_dir=train_dir+'Type_2/'
type3_dir=train_dir+'Type_3/'


# Configuration

conf = dict()
conf['sample_size'] = 0.33
conf['val_split'] = 0.2
conf['batch_size'] = 64
conf['epoch'] = 15
conf['patience'] = 3
conf['image_size'] = (64, 64, 3)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Utils

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=64, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)



# Load Data

batches_test=get_batches(test_dir, target_size=conf['image_size'][0:2], shuffle=False)

def create_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(4, (3, 3), activation='relu',input_shape=input_shape)) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax')) 
    return model

model=create_model(conf['image_size'])
model.load_weights(save_dir+'ms4_best.h5')
def get_name(str):
    return str[str.index('/')+1:]
test_ids=[get_name(name) for name in batches_test.filenames]
pred=model.predict_generator(batches_test, math.ceil(batches_test.n/conf['batch_size']), workers=4, verbose=1)
df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_ids
df.to_csv('submission.csv', index=False)

