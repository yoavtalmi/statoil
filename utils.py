import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, merge
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD
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
        X_train_temp = [np.array([band * np.random.normal(1,np.std(band)/np.abs(np.mean(band)),size=(75,75)) for band in image.reshape(3,75,75)]).reshape(75,75,3) for image in  X_train_temp]
        for image in X_train_temp:
            image.reshape(3,75,75)[2][0][0] = np.mean([image.reshape(3,75,75)[0][0][0],image.reshape(3,75,75)[1][0][0]])
        X_train_final = np.concatenate([X_train_final,X_train_temp])
        X_angle_train_final = np.concatenate([X_angle_train_final, X_angle_train_temp])
        y_train_final = np.concatenate([y_train_final, y_train_temp])
    
    return X_train_final, X_angle_train_final, y_train_final

def flip(X_train,X_angle_train,y_train, number_syn = 2):
    
    X_train_final = X_train.copy()
    X_angle_train_final = X_angle_train.copy()
    y_train_final = y_train.copy()
    
    
    fun_dict = {'fliplr' : [np.fliplr], 'flipud' : [np.flipud], 'both' : [np.fliplr,np.flipud]}
    for key in fun_dict.keys():
        X_train_temp =  X_train.copy()
        X_angle_train_temp =  X_angle_train.copy()
        y_train_temp =  y_train.copy()
        for fun in fun_dict[key]:
            X_train_temp = [np.array([fun(band) for band in image_1.reshape(3,75,75)]).reshape(75,75,3) for image_1 in  X_train_temp]
        X_train_final = np.concatenate([X_train_final,X_train_temp])
        X_angle_train_final = np.concatenate([X_angle_train_final, X_angle_train_temp])
        y_train_final = np.concatenate([y_train_final, y_train_temp])
    
    return X_train_final, X_angle_train_final, y_train_final
         


def adding_noise(X_train,X_angle_train,y_train, mean, std, number_syn = 2):
    
    X_train_final = X_train.copy()
    X_angle_train_final = X_angle_train.copy()
    y_train_final = y_train.copy()
    for syn in range(1,number_syn):
        X_train_temp =  X_train.copy()
        X_angle_train_temp = X_angle_train.copy()
        y_train_temp = y_train.copy()
        X_train_temp = [np.array([(band + np.random.normal(mean,std/2,size=(75,75)))/2 for band in image.reshape(3,75,75)]).reshape(75,75,3) for image in  X_train_temp]
        for image in X_train_temp:
            image.reshape(3,75,75)[2][0][0] = np.mean([image.reshape(3,75,75)[0][0][0],image.reshape(3,75,75)[1][0][0]])
        X_train_final = np.concatenate([X_train_final,X_train_temp])
        X_angle_train_final = np.concatenate([X_angle_train_final, X_angle_train_temp])
        y_train_final = np.concatenate([y_train_final, y_train_temp])
    
    return X_train_final, X_angle_train_final, y_train_final


def black_box(X_train,X_angle_train,y_train, number_syn = 2, box_size = 5):
    
    X_train_final = X_train.copy()
    X_angle_train_final = X_angle_train.copy()
    y_train_final = y_train.copy()
    for syn in range(1,number_syn):
        X_train_temp =  []
        X_angle_train_temp = X_angle_train.copy()
        y_train_temp = y_train.copy()
        for image in X_train:
            box = np.zeros((75,75))
            num_box = np.random.choice(range(1,2))            
            for i in range(num_box):
                box_size_temp = np.random.choice(range(10,11),size=2)
                top_left = np.random.choice(range(75 - box_size_temp[0]))
                top_left_2 = np.random.choice(range(75 - box_size_temp[1]))
                box[top_left:top_left +  box_size_temp[0],top_left_2:top_left_2 + box_size_temp[1]] = -99
            X_train_temp.append((np.array([(band + box)  for band in image.reshape(3,75,75)]).reshape(75,75,3)))
        X_train_final = np.concatenate([X_train_final,X_train_temp])
        X_angle_train_final = np.concatenate([X_angle_train_final, X_angle_train_temp])
        y_train_final = np.concatenate([y_train_final, y_train_temp])
    
    return X_train_final, X_angle_train_final, y_train_final

def noise_box(X_train,X_angle_train,y_train, number_syn = 2, box_size = 5):
    
    X_train_final = X_train.copy()
    X_angle_train_final = X_angle_train.copy()
    y_train_final = y_train.copy()
    for syn in range(1,number_syn):
        X_train_temp =  []
        X_angle_train_temp = X_angle_train.copy()
        y_train_temp = y_train.copy()
        for image in X_train:
            box = np.ones((75,75))
            num_box = np.random.choice(range(1,2))            
            for i in range(num_box):
                box_size_temp = np.random.choice(range(10,11),size=2)
                top_left = np.random.choice(range(75 - box_size_temp[0]))
                top_left_2 = np.random.choice(range(75 - box_size_temp[1]))
                box[top_left:top_left +  box_size_temp[0],top_left_2:top_left_2 + box_size_temp[1]] = np.random.normal(1,np.std(image),size=(10,10))
            X_train_temp.append((np.array([(band * box)  for band in image.reshape(3,75,75)]).reshape(75,75,3)))
        X_train_final = np.concatenate([X_train_final,X_train_temp])
        X_angle_train_final = np.concatenate([X_angle_train_final, X_angle_train_temp])
        y_train_final = np.concatenate([y_train_final, y_train_temp])
    
    return X_train_final, X_angle_train_final, y_train_final


def self_generator(features, features_angle, labels, batch_size, method='white_noise_gen', box_size=10):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 75, 75, 3))
 batch_angles = np.zeros((batch_size,1))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= np.random.choice(len(features),1)
     if method == 'white_noise_gen':
         batch_features[i] = white_noise_gen(features[index])
     if method == 'black_box_gen':
         batch_features[i] = black_box_gen(features[index],box_size)
     batch_angles[i] = features_angle[index]
     batch_labels[i] = labels[index]
   yield [batch_features, batch_angles], batch_labels
   
def white_noise_gen(image):
    X_train_temp = np.array([band * np.random.normal(1,np.std(band)/np.abs(np.mean(band)),size=(75,75)) for band in image.reshape(3,75,75)]).reshape(75,75,3)
    X_train_temp.reshape(3,75,75)[2][0][0] = np.mean([X_train_temp.reshape(3,75,75)[0][0][0],X_train_temp.reshape(3,75,75)[1][0][0]])
    return X_train_temp
        
def black_box_gen(image,box_size):
    num_box = np.random.choice(range(3),size=1)
    box = np.zeros((75,75))
    for num in range(num_box):
        box_size_temp = np.random.choice(range(8,11),size=2)
        top_left = np.random.choice(range(75 - box_size_temp[0]))
        top_left_2 = np.random.choice(range(75 - box_size_temp[1]))
        box[top_left:top_left +  box_size_temp[0],top_left_2:top_left_2 + box_size_temp[1]] = -99
    return np.array([(band + box)  for band in image.reshape(3,75,75)]).reshape(75,75,3)
    
    
        


""" Models """

def keras_baselilne(input_x = 75, input_y = 75):
    
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(input_x, input_y, 3), name="X_1")
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


def keras_baselilne_batch(input_x = 75, input_y = 75):
    
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(input_x, input_y, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")
    
    img_1 = BatchNormalization(momentum = 0.99) (input_1)
    img_1 = Conv2D(16,3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = Conv2D(16,3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = Conv2D(32, 3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = Conv2D(32, 3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = Conv2D(64,3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = Conv2D(64,3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, 3,3, activation=p_activation) (img_1)
    img_1 = BatchNormalization(momentum = 0.99) (img_1)
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
    
    model = Model([input_1,input_2],  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def resnet(input_x = 75, input_y = 75):
    
    
    kernel_size = (5,5)
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(input_x, input_y, 3), name="X_1")
    
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
        
        

        

