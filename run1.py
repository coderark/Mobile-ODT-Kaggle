import os, sys, pickle, math
import numpy as np
from glob import glob
from shutil import copyfile
from PIL import ImageFile

from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import log_loss
from keras import backend as K

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

conf = dict()
conf['sample_size'] = 0.33
conf['val_split'] = 0.2
conf['batch_size'] = 64
conf['nb_epoch'] = 1
conf['patience'] = 3
conf['image_size'] = (224, 224, 3)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def vgg_preprocess(x):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
    x = x - vgg_mean
    return x[:, ::-1]

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=64, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

batches_train=get_batches(train_dir, target_size=conf['image_size'][0:2])
batches_valid=get_batches(valid_dir, target_size=conf['image_size'][0:2])

train_labels=to_categorical(batches_train.classes)
valid_labels=to_categorical(batches_valid.classes)

def conv_block(model, num_layers, num_filters):
    for i in range(num_layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(num_filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    return model
    
def FCB_block(model, units=4096, dropout=False, bn=False):
    model.add(Dense(units))
    if bn: model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout: model.add(Dropout(0.5))
    return model

def create_full_vgg16(input_shape):
    model=Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=input_shape, output_shape=input_shape))
    model=conv_block(model, 2, 64)
    model=conv_block(model, 2, 128)    
    model=conv_block(model, 3, 256)    
    model=conv_block(model, 3, 512)    
    model=conv_block(model, 3, 512)
    
    model.add(Flatten())
    model=FCB_block(model, dropout=True)
    model=FCB_block(model, dropout=True)
    model.add(Dense(1000, activation='softmax'))
    fname='vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    local_path=local_model_path+fname
    down_path='https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    if os.path.exists(local_path):
        model.load_weights(local_path)
    else:
        model.load_weights(get_file(fname, down_path, cache_subdir='models'))
    return model

def create_conv_vgg16(input_shape):
    model=Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model=conv_block(model, 1, 64)
    model=conv_block(model, 2, 128)    
    model=conv_block(model, 3, 256)    
    model=conv_block(model, 3, 512)    
    model=conv_block(model, 3, 512)
    fname='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    local_path=local_model_path+fname
    down_path='https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if os.path.exists(local_path):
        model.load_weights(local_path)
    else:
        model.load_weights(get_file(fname, down_path, cache_subdir='models'))
    return model

def finetune(model, num_class, level=2):
    layers=model.layers
    layers_idx=[idx for idx, layer in enumerate(layers) if type(layer) is Convolution2D or type(layer) is Dense]
    for i in range(layers_idx[level*-1]): layers[i].trainable=False
    model.pop()
    model.add(Dense(num_class, activation='softmax'))
    return model

def get_base_model(model, level=2):
    layers=model.layers
    layers_idx=[idx for idx, layer in enumerate(layers) if type(layer) is Convolution2D or type(layer) is Dense]
    for i in range(len(layers)): 
        if i>=layers_idx[level*-1]:
            model.pop()
    return model

vgg16_full=create_full_vgg16(conf['image_size'])

vgg16_ft1=finetune(vgg16_full, 4)

vgg16_base1=get_base_model(vgg16_ft1, level=3)

train_base=vgg16_base1.predict_generator(batches_train, math.ceil(batches_train.n/conf['batch_size']), workers=4, verbose=1)
valid_base=vgg16_base1.predict_generator(batches_valid, math.ceil(batches_valid.n/conf['batch_size']), workers=4, verbose=1)

np.save(save_dir+'X1_train_conv_224', train_base)
np.save(save_dir+'X1_valid_conv_224', valid_base)
