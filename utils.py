import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, merge
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

batch_size = 32

""" Augmentations """


def gen_flow_for_two_inputs(X1, X2, y, gen):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrasy are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]
            

def white_noise(X_train,X_angle_train,y_train, number_syn = 2):
    
    X_train_final = X_train.copy()
    X_angle_train_final = X_angle_train.copy()
    y_train_final = y_train.copy()
    for syn in range(1,number_syn):
        X_train_temp =  X_train.copy()
        X_angle_train_temp = X_angle_train.copy()
        y_train_temp = y_train.copy()
        X_train_temp = [np.array([band * np.random.normal(1,np.std(band),size=(75,75)) for band in image.reshape(3,75,75)]).reshape(75,75,3) for image in  X_train_temp]
        X_train_final = np.concatenate([X_train_final,X_train_temp])
        X_angle_train_final = np.concatenate([X_angle_train_final, X_angle_train_temp])
        y_train_final = np.concatenate([y_train_final, y_train_temp])
    
    return X_train_final, X_angle_train_final, y_train_final
        
        
        
        
        
            


""" Models """

def keras_baselilne():
    
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")
    
    img_1 = Conv2D(16,3,3, activation=p_activation) (input_1)
    img_1 = Conv2D(16,3,3, activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(32, 3,3, activation=p_activation) (img_1)
    img_1 = Conv2D(32, 3,3, activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64,3,3, activation=p_activation) (img_1)
    img_1 = Conv2D(64,3,3, activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, 3,3, activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D() (img_1)
    
    
    img_2 = Conv2D(128, 3,3, activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D() (img_2)
    
    img_3 = BatchNormalization(momentum=bn_model)(input_2)
    
    img_concat =  (Concatenate()([img_1, img_2, img_3]))
    
    dense_ayer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(256, activation=p_activation)(img_concat) ))
    dense_ayer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(64, activation=p_activation)(dense_ayer) ))
    output = Dense(1, activation="sigmoid")(dense_ayer)
    
    model = Model([input_1,input_2],  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def resnet():
    
    
    kernel_size = (5,5)
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    
    ## input CNN
    input_CNN  = BatchNormalization(momentum = 0.99)(input_1 )
    input_CNN  = Conv2D(32,kernel_size, activation=p_activation, padding='same') (input_CNN )
    input_CNN  = BatchNormalization(momentum = 0.99)(input_CNN )
    input_CNN  = MaxPooling2D((2,2)) (input_CNN )
    input_CNN  = Dropout(0.25)(input_CNN )
    input_CNN  = Conv2D(64,kernel_size, activation=p_activation, padding='same') (input_CNN )
    input_CNN  = BatchNormalization(momentum = 0.99)(input_CNN )
    input_CNN  = MaxPooling2D((2,2)) (input_CNN )
    input_CNN  = Dropout(0.25)(input_CNN )
    

    ## first residual
    
    input_CNN_residual  = BatchNormalization(momentum = 0.99)(input_CNN)
    input_CNN_residual  = Conv2D(128,kernel_size, activation=p_activation, padding='same') (input_CNN_residual )
    input_CNN_residual  = BatchNormalization(momentum = 0.99)(input_CNN_residual)
    input_CNN_residual  = Dropout(0.25)(input_CNN_residual )
    input_CNN_residual  = Conv2D(64,kernel_size, activation=p_activation, padding='same') (input_CNN_residual )
    input_CNN_residual  = BatchNormalization(momentum = 0.99)(input_CNN_residual)
    
    input_CNN_residual = merge([input_CNN_residual,input_CNN], mode = 'sum')
    
    ## final CNN
    
    top_CNN  = Conv2D(128,kernel_size, activation=p_activation, padding='same') (input_CNN_residual )
    top_CNN  = BatchNormalization(momentum = 0.99)(top_CNN)
    top_CNN  = MaxPooling2D((2,2)) (top_CNN )
    top_CNN  = Conv2D(256,kernel_size, activation=p_activation, padding='same') (top_CNN )
    top_CNN  = BatchNormalization(momentum = 0.99)(top_CNN)
    top_CNN  = Dropout(0.25)(top_CNN )
    top_CNN  = MaxPooling2D((2,2)) (top_CNN )
    top_CNN  = Conv2D(512,kernel_size, activation=p_activation, padding='same') (top_CNN )
    top_CNN  = BatchNormalization(momentum = 0.99)(top_CNN)
    top_CNN  = Dropout(0.25)(top_CNN )
    top_CNN  = MaxPooling2D((2,2)) (top_CNN )
    top_CNN  = GlobalMaxPooling2D() (top_CNN)
    

    
    layer_dense = Dense(512)(top_CNN)
    layer_dense  = BatchNormalization(momentum = 0.99)(layer_dense)
    layer_dense  = Dropout(0.5)(layer_dense )
    layer_dense = Dense(256)(layer_dense)
    layer_dense  = BatchNormalization(momentum = 0.99)(layer_dense)
    layer_dense  = Dropout(0.5)(layer_dense )
    output = Dense(1, activation="sigmoid")(layer_dense)
    
    
    model = Model(input_1,  output)
    optimizer = Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model




"""visualization function"""



def true_positive(valid):
    valid_positive = valid[valid['is_iceberg']==1]
    valid_positive.sort_values('prediction',ascending=False,inplace=True)
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in valid_positive["band_1"]])
    for i in range(5):
        plt.imshow(np.array(x_band1[i]).astype(np.float32).reshape(75, 75)) 
        plt.title(valid_positive.ix[valid_positive.index[i], 'prediction'])
        plt.show()
        
def true_negetive(valid):
    valid_negetive = valid[valid['is_iceberg']==0]
    valid_negetive.sort_values('prediction',ascending=True,inplace=True)
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in valid_negetive["band_1"]])
    for i in range(5):
        plt.imshow(np.array(x_band1[i]).astype(np.float32).reshape(75, 75))
        plt.title(valid_negetive.ix[valid_negetive.index[i], 'prediction'])
        plt.show()
        
def false_negative(valid):
    valid_positive = valid[valid['is_iceberg']==1]
    valid_positive.sort_values('prediction',ascending=True,inplace=True)
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in valid_positive["band_1"]])
    for i in range(5):
        plt.imshow(np.array(x_band1[i]).astype(np.float32).reshape(75, 75)) 
        plt.title(valid_positive.ix[valid_positive.index[i], 'prediction'])
        plt.show()

def false_positive(valid):
    valid_negetive = valid[valid['is_iceberg']==0]
    valid_negetive.sort_values('prediction',ascending=False,inplace=True)
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in valid_negetive["band_1"]])
    for i in range(5):
        plt.imshow(np.array(x_band1[i]).astype(np.float32).reshape(75, 75))
        plt.title(valid_negetive.ix[valid_negetive.index[i], 'prediction'])
        plt.show()
        
        

        

