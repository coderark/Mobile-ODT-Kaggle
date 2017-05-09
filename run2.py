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
from keras.layers import Activation, Input, GlobalAveragePooling2D
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
from keras.applications.inception_v3 import InceptionV3

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
conf['image_size'] = (299, 299, 3)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=64, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

batches_train=get_batches(train_dir, target_size=conf['image_size'][0:2])
batches_valid=get_batches(valid_dir, target_size=conf['image_size'][0:2])

train_labels=to_categorical(batches_train.classes)
valid_labels=to_categorical(batches_valid.classes)

base_model = InceptionV3(weights='imagenet', include_top=False)

train_base=base_model.predict_generator(batches_train, math.ceil(batches_train.n/conf['batch_size']), workers=4, verbose=1)
valid_base=base_model.predict_generator(batches_valid, math.ceil(batches_valid.n/conf['batch_size']), workers=4, verbose=1)

print(train_base.shape, valid_base.shape)

np.save(save_dir+'X2_train_iv3_299', train_base)
np.save(save_dir+'X2_valid_iv3_299', valid_base)
