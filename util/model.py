from keras.layers import Dense, Activation, CuDNNLSTM, GlobalAveragePooling1D
from keras.layers import add, concatenate, BatchNormalization, Conv1D
from keras.layers import Reshape, Input, LSTM, Dropout,Bidirectional
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend
from keras.optimizers import Adam, Adadelta

import numpy as np
import os
def custom_loss(layer, lay2, T):
    def loss(y_true,y_pred):
        from keras.losses import categorical_crossentropy as logloss
        from keras.losses import kullback_leibler_divergence as kld
        ce = logloss(y_true, y_pred)
        if T>= 1:
            y_pred_soft = backend.softmax(lay2/T)
            layer_soft = backend.softmax(layer/T)
            kl = kld(layer_soft, y_pred_soft)
            return ce + (T**2)*kl
        elif T==0:
            return ce
        else:
            return ce*0
    return loss


def lstm_block(size, units, inp):
    """
    LSTM Network composed of 2 LSTM layers
    """

    t = Reshape((size))(inp)
    t = LSTM(units[0], recurrent_dropout=0.5, dropout=0.2, return_sequences=True)(t)
    t = LSTM(units[1], recurrent_dropout=0.5, dropout=0.5, return_sequences=False)(t)
    return t

def bilstm_block(size, units, inp):
    """
    LSTM Network composed of 2 LSTM layers
    """

    t = Reshape((size))(inp)
    t = Bidirectional(LSTM(units[0], recurrent_dropout=0.5, dropout=0.2, return_sequences=True))(t)
    t = Bidirectional(LSTM(units[1], recurrent_dropout=0.5, dropout=0.5, return_sequences=False))(t)
    return t

def cudnnlstm_block(size, units, inp):
    """
    LSTM Network composed of 2 CuDNNLSTM layers

    """

    t = Reshape((size))(inp)
    t = CuDNNLSTM(units[0], return_sequences=True)(t)
    t = CuDNNLSTM(units[1], return_sequences=False)(t)
    return t

def TCN_Block(inp, activation_custom, vals, jump=True, length=8):
    """
    TCN Network
    """

    t = Conv1D(vals[0], length, padding='same')(inp)

    def sub_block(activation_custom, fc_units, stride, inp, length):
        t1 = Conv1D(fc_units, 1, strides=stride, padding='same')(inp)
        t = BatchNormalization(axis=-1)(inp)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(stride), dilation_rate=1, padding='causal')(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same')(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=2, padding='causal')(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same')(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=4, padding='causal')(t)
        t = add([t1, t2])
        return t

    tout1 = sub_block(activation_custom, vals[0],1,t, length)
    tout2 = sub_block(activation_custom, vals[1],jump+1,tout1, length)
    tout3 = sub_block(activation_custom, vals[2],jump+1,tout2, length)
    tout4 = sub_block(activation_custom, vals[3],jump+1,tout3, length)

    return tout1, tout2, tout3, tout4

class model(object):
    """
    Machine Learning Model
    """
    def __init__(self,
        num_classes=1,          # Number of classes for the model
        model_type=(0,''),      # Model Type, 0 for TCN, 1 for LSTM, 2 for TCN+LSTM
        lr=1e-3,                # Learning Rate
        num_frame=60,           # Number of frames
        feature_size=(6,)):    # Feature dimensions

        self.model = None
        self.model_type = model_type
        self.num_frame = num_frame
        self.optimizer = Adam(lr=lr)
#        self.optimizer = Adadelta(lr=lr)

        if model_type[0] == "TCN": #TCN
            self.create_model_0(num_frame, num_classes, feature_size)
        elif model_type[0] == "LSTM": #LSTM
            self.create_model_1(num_frame, num_classes, feature_size)
        elif model_type[0] == "BiLSTM": #BiLSTM
            self.create_model_2(num_frame, num_classes, feature_size)
        elif model_type[0] == "GridLSTM": #BiLSTM
            self.create_model_2(num_frame, num_classes, feature_size)
        elif model_type[0] == "BGLSTM": #BiLSTM
            self.create_model_2(num_frame, num_classes, feature_size)

#        elif model_type[0] == 2: #TCN + LSTM
#            self.create_model_2(num_frame, num_classes, feature_size)

    def create_model_0(self, num_frame, num_classes, feature_size,activation_custom="relu"):
        main_input = Input(shape=(num_frame, feature_size[0]))
        t = Reshape((num_frame, feature_size[0]))(main_input)

        vals = [8, 16, 32, 64]
        t1, t2, t3, t4 = TCN_Block(t, activation_custom, vals, jump=True, length=6)
        t = BatchNormalization(axis=-1)(t4)
        t = GlobalAveragePooling1D()(t)

        t = Dense(20)(t)
        tout = Dense(num_classes,activation="sigmoid")(t)

        self.model = Model(inputs=main_input, output=[tout])
        self.model.summary()

        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=[])


    def create_model_1(self, num_frame, num_classes, feature_size,activation_custom="relu"):
        main_input = Input(shape=(num_frame, feature_size[0]))
        t = lstm_block((num_frame, feature_size[0]), (60,30) , main_input)
        t = BatchNormalization(axis=-1)(t)
        t = Dense(20)(t)
        tout = Dense(num_classes,activation="sigmoid")(t)

        self.model = Model(inputs=main_input, output=[tout])
        self.model.summary()

        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=[])


    def create_model_2(self, num_frame, num_classes, feature_size,activation_custom="relu"):
        main_input = Input(shape=(num_frame, feature_size[0]))
        t = bilstm_block((num_frame, feature_size[0]), (60,30) , main_input)
        t = BatchNormalization(axis=-1)(t)
        t = Dense(20)(t)
        tout = Dense(num_classes,activation="sigmoid")(t)

        self.model = Model(inputs=main_input, output=[tout])
        self.model.summary()

        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=[])
#
#    def create_model_2(self, num_frame, num_classes, feature_size,activation_custom="relu"):
#        main_input = Input(shape=(num_frame, feature_size[0]))
#        t_lstm_feature = cudnnlstm_block((num_frame, feature_size[0]), (60,30), main_input)
#        t_lstm_logit = Dense(num_classes)(t_lstm_feature)
##        t_lstm_out = Activation('softmax', name='t1')(t_lstm_logit)
#
#        t = Reshape((num_frame, feature_size[0]))(main_input)
#        vals = [8, 16, 32, 64]
#        t1, t2, t3, t4 = TCN_Block(t, activation_custom, vals, jump=True, length=6)
#        t = BatchNormalization(axis=-1)(t4)
#        t = Activation(activation_custom)(t)
#        t_tcn_feature = GlobalAveragePooling1D()(t)
#        t_tcn_logit = Dense(num_classes)(t_tcn_feature)
##        t_tcn_out = Activation('softmax', name='t2')(t_tcn_logit)
#
#        logit = concatenate([t_lstm_feature, t_tcn_feature])
#        logit = Dense(num_classes)(logit)
#        tout = Activation('softmax', name='out')(logit)
#
#        self.model = Model(inputs=main_input, output=[tout, t_lstm_out, t_tcn_out])
#        self.model.summary()
#        losses = {"out": 'mse',
#          "t1": custom_loss(logit, t_lstm_logit, 1),
#          "t2": custom_loss(logit, t_tcn_logit, 1)
#          }
#        self.model.compile(loss=losses, optimizer=self.optimizer, metrics=['mae'])


    def train(self,
              train_data, train_label,
              val_data=[], val_label=[],
              bs=1,
              base_epoch=10,
              path=''
              ):

#        train_label = to_categorical(train_label, num_classes=None)
#        val_label = to_categorical(val_label, num_classes=None)

        os.makedirs(f'{path}//{self.model_type[0]}//', exist_ok=True)

        filepath = f'{path}//{self.model_type[0]}//{self.model_type[1]}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='min')

        early=EarlyStopping(monitor='val_loss', patience=100,verbose=0,mode='auto')


        callbacks_list = [checkpoint, early]

        self.model.fit(train_data, train_label, batch_size=bs,
                       epochs=base_epoch, shuffle=True,
                       validation_data=(val_data, val_label),
                       verbose=1, callbacks=callbacks_list,
                       )

        return filepath

    def load(self, path):
        return self.model.load_weights(path)

    def predict(self, data):
        label = self.model.predict(data)
        return label