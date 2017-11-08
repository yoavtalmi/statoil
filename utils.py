from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, merge
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

batch_size = 32


def gen_flow_for_two_inputs(X1, X2, y, gen):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrasy are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]
            

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
            
        

