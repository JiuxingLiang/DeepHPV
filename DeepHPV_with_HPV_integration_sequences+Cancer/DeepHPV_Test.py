from __future__ import print_function, division
import numpy as np

from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras.layers.core as core
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, multiply, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec
from keras.utils import CustomObjectScope
from keras.layers.merge import concatenate
import keras
from keras.callbacks import TensorBoard
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # choice GPU


class Attention(Layer):
    def __init__(self, hidden, init='glorot_uniform', activation='linear', W_regularizer=None, b_regularizer=None,
                 W_constraint=None, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.hidden = hidden
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W0 = self.add_weight(name='{}_W1'.format(self.name), shape=(input_dim, self.hidden),
                                  initializer='glorot_uniform', trainable=True)  # Keras 2 API
        self.W = self.add_weight(name='{}_W'.format(self.name), shape=(self.hidden, 1), initializer='glorot_uniform',
                                 trainable=True)
        self.b0 = K.zeros((self.hidden,), name='{}_b0'.format(self.name))
        self.b = K.zeros((1,), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W0, self.W, self.b, self.b0]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W0] = self.W_constraint
            self.constraints[self.W] = self.W_constraint

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attmap = self.activation(K.dot(x, self.W0) + self.b0)
        attmap = K.dot(attmap, self.W) + self.b
        attmap = K.reshape(attmap, (-1, self.input_length))  # Softmax needs one dimension
        attmap = K.softmax(attmap)
        dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
        out = K.concatenate([dense_representation,
                             attmap])  # Output the attention maps but do not pass it to the next layer by DIY flatten layer
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] + input_shape[1])

    def get_config(self):
        config = {'init': 'glorot_uniform',
                  'activation': self.activation.__name__,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'hidden': self.hidden if self.hidden else None}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class attention_flatten(Layer):  # Based on the source code of Keras flatten
    def __init__(self, keep_dim, **kwargs):
        self.keep_dim = keep_dim
        super(attention_flatten, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" '
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. '
                                                             'Make sure to pass a complete "input_shape" '
                                                             'or "batch_input_shape" argument to the first '
                                                             'layer in your model.')
        return (input_shape[0], self.keep_dim)  # Remove the attention map

    def call(self, x, mask=None):
        x = x[:, :self.keep_dim]
        return K.batch_flatten(x)


def set_up_model_up():
    print('building model')

    seq_input_shape = (2000, 4)
    nb_filter = 128
    nb_filter1 = 256
    filter_length = 8
    filter_length1 = 6
    attentionhidden = 256

    seq_input = Input(shape=seq_input_shape, name='seq_input')
    convul1 = Convolution1D(filters=nb_filter, kernel_size=filter_length, padding='valid', activation='relu',
                            kernel_constraint=maxnorm(3), subsample_length=1)
    convul11 = Convolution1D(filters=nb_filter1, kernel_size=filter_length1, padding='valid', activation='relu',
                             kernel_constraint=maxnorm(3), subsample_length=1)

    pool_ma1 = MaxPooling1D(pool_size=3)
    pool_ma11 = MaxPooling1D(pool_size=3)
    dropout1 = Dropout(0.55)
    dropout2 = Dropout(0.50)
    decoder = Attention(hidden=attentionhidden, activation='linear')  # attention_module
    dense1 = Dense(1)
    dense2 = Dense(1)

    output_1 = pool_ma1(convul1(seq_input))
    output_12 = pool_ma11(convul11(output_1))

    output_2 = dropout1(output_12)
    att_decoder = decoder(output_2)
    output_3 = attention_flatten(output_2._keras_shape[2])(att_decoder)
    output_4 = dense1(dropout2(Flatten()(output_2)))

    # all_outp = merge([output_3, output_4], mode='concat')
    # all_outp = merge([output_3, output_4], mode='Sum')
    all_outp = concatenate([output_3, output_4])
    output_5 = dense2(all_outp)

    output_f = Activation('sigmoid')(output_5)

    model = Model(inputs=seq_input, outputs=output_f)  # Determine the input and output layers
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model


def test_model():
    dsVIS_test_Data = np.load('Data/dsVIS_Test_Data.npy')  # test_data
    dsVIS_test_Label = np.load('Data/dsVIS_Test_Label.npy')  # test_label
    dsVIS_test_Label.astype(int)  # Change the data format
    VISDB_test_Data = np.load('Data/VISDB_Test_Data.npy')  # test_data
    VISDB_test_Label = np.load('Data/VISDB_Test_Label.npy')  # test_label
    VISDB_test_Label.astype(int)  # Change the data format

    print('testing_model')
    model = set_up_model_up()
    model.load_weights('Model/DeepHPV_with_HPV_integration_sequences+Cancer.hdf5')

    model.summary()  # Show model structure

    print('Predicting...')
    dsVIS_test_pred = model.predict(dsVIS_test_Data)
    VISDB_test_pred = model.predict(VISDB_test_Data)
    # Labels_pred = np.round(Label_pred)
    # Labels_pred = Labels_pred.astype(int)

    # print('loss_and_acc...')
    dsVIS_Acc = model.evaluate(dsVIS_test_Data, dsVIS_test_Label, batch_size=128)  # Test
    VISDB_Acc = model.evaluate(VISDB_test_Data, VISDB_test_Label, batch_size=128)  # test
    dsVIS_rocauc_score = roc_auc_score(dsVIS_test_Label, dsVIS_test_pred)
    dsVIS_AveragePrecision_score = average_precision_score(dsVIS_test_Label, dsVIS_test_pred)
    VISDB_rocauc_score = roc_auc_score(VISDB_test_Label, VISDB_test_pred)  # ROC
    VISDB_AveragePrecision_score = average_precision_score(VISDB_test_Label, VISDB_test_pred)  # AvPR
    dsVIS_Test_Quantity = pd.value_counts(dsVIS_test_Label)
    # print('Test_neg_Quantity and Test_pos_Quantity', Test_Quantity._ndarray_values)
    # dsVIS_Test_neg_Quantity = dsVIS_Test_Quantity._ndarray_values[0]
    # dsVIS_Test_pos_Quantity = dsVIS_Test_Quantity._ndarray_values[1]
    VISDB_Test_Quantity = pd.value_counts(VISDB_test_Label)
    # print('Test_neg_Quantity and Test_pos_Quantity', Test_Quantity._ndarray_values)
    # VISDB_Test_neg_Quantity = VISDB_Test_Quantity._ndarray_values[0]
    # VISDB_Test_pos_Quantity = VISDB_Test_Quantity._ndarray_values[1]

    print('-------dsVIS_test_result------------------')
    print('dsVIS_Test_pos_Quantity:', dsVIS_Test_Quantity._ndarray_values[1])
    print('dsVIS_Test_neg_Quantity:', dsVIS_Test_Quantity._ndarray_values[0])
    print('Test acc:', dsVIS_Acc[1])
    print('auroc:', dsVIS_rocauc_score)
    print('aupr:', dsVIS_AveragePrecision_score)
    print('-------VISDB_test_result------------------')
    print('VISDB_Test_pos_Quantity:', VISDB_Test_Quantity._ndarray_values[1])
    print('VISDB_Test_neg_Quantity:', VISDB_Test_Quantity._ndarray_values[0])
    print('Test acc:', VISDB_Acc[1])
    print('auroc:', VISDB_rocauc_score)
    print('aupr:', VISDB_AveragePrecision_score)

    # test_pred = Label_pred
    # test_label = Label_test
    np.save('Pred_Result/dsVIS_Result/Test_label', dsVIS_test_Label)
    np.save('Pred_Result/dsVIS_Result/Test_pred', dsVIS_test_pred)
    np.save('Pred_Result/VISDB_Result/Test_label', VISDB_test_Label)
    np.save('Pred_Result/VISDB_Result/Test_pred', VISDB_test_pred)

    # Save the excel
    output_directory = 'Pred_Result/dsVIS_Result/'
    model_test_results = pd.DataFrame(data=np.zeros((1, 5), dtype=np.float), index=[0],
                                      columns=['dsVIS_Test_pos_Quantity', 'dsVIS_Test_neg_Quantity', 'Test_acc',
                                               'rocauc_score', 'AveragePrecision_score'])
    model_test_results['dsVIS_Test_pos_Quantity'] = dsVIS_Test_Quantity._ndarray_values[1]
    model_test_results['dsVIS_Test_neg_Quantity'] = dsVIS_Test_Quantity._ndarray_values[0]
    model_test_results['Test_acc'] = dsVIS_Acc[1]
    model_test_results['rocauc_score'] = dsVIS_rocauc_score
    model_test_results['AveragePrecision_score'] = dsVIS_AveragePrecision_score
    model_test_results.to_csv(output_directory + 'model_test_results.csv', index=False)

    output_directory = 'Pred_Result/VISDB_Result/'
    model_test_results = pd.DataFrame(data=np.zeros((1, 5), dtype=np.float), index=[0],
                                      columns=['VISDB_Test_pos_Quantity', 'VISDB_Test_neg_Quantity', 'Test_acc',
                                               'rocauc_score', 'AveragePrecision_score'])
    model_test_results['VISDB_Test_pos_Quantity'] = VISDB_Test_Quantity._ndarray_values[1]
    model_test_results['VISDB_Test_neg_Quantity'] = VISDB_Test_Quantity._ndarray_values[0]
    model_test_results['Test_acc'] = VISDB_Acc[1]
    model_test_results['rocauc_score'] = VISDB_rocauc_score
    model_test_results['AveragePrecision_score'] = VISDB_AveragePrecision_score
    model_test_results.to_csv(output_directory + 'model_test_results.csv', index=False)


if __name__ == '__main__':
    test_model()
