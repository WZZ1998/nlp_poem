import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Input
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
import my_model
tr_poems = open('train.txt', 'r').readlines()
list_7 = []
list_14 = []
list_be = []
list_af = []
list_21_in = []
list_21_tar = []
poem_view_list = []
for line in tr_poems:
    poem = line.split(' ')
    poem_view_list.append(poem)
    list_7.append(poem[0:7])
    list_14.append(poem[8:15])
    list_be.append(poem[0:7] + poem[8:15])
    list_af.append(poem[16:23] + poem[24:31])
    list_21_in.append(['@'] + poem[8:15] + poem[16:23] + poem[24:31])
    list_21_tar.append(poem[8:15] + poem[16:23] + poem[24:31] + ['#'])
char2id = []
Char_set = set()
for i in range(len(list_7)):
    char2id.extend(list_7[i] + list_21_in[i] + list_21_tar[i])
    Char_set.update(list_7[i] + list_21_in[i] + list_21_tar[i])
char2id = pd.Series(char2id).value_counts()
char2id[:] = range(len(char2id))
id2char = {}
for i in Char_set:
    id2char[char2id[i]] = i


batch_size = 64
epochs = 60
lstm_hidden_dim = 512
word_size = 400
encoder_seq_length = 14
decoder_seq_length = 14
dic_size = len(char2id)


def poem2num(poem_list):
    npoem_list = []
    for line in poem_list:
        one_line = []
        for one_char in line:
            one_line.append(char2id[one_char])
        npoem_list.append(one_line)
    return npoem_list


def poem2one_hot(poem_list, dic_size, line_length):
    plist = np.array(poem_list)
    res = np.zeros(shape=(len(plist), line_length, dic_size), dtype='bool')
    for i in range(len(plist)):
        for j in range(line_length):
            res[i, j, char2id[plist[i, j]]] = 1
    return res


encoder_input_data = np.array(poem2num(list_be))
#decoder_input_data = np.array(poem2num(list_21_in))
decoder_target_data = poem2one_hot(list_af, dic_size, decoder_seq_length)
new_model = my_model.create_super_model(len(char2id))
checkpointer = ModelCheckpoint(
    filepath="my_trained_model/fin_poem_s2s_weights.h5", verbose=1, save_best_only=False, period=1, save_weights_only=True)
new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
                  'accuracy'])
new_model.fit(encoder_input_data, decoder_target_data,
              batch_size=batch_size, epochs=epochs, callbacks=[checkpointer])

new_model.save('my_trained_model/fin_poem_s2s_model.h5')


exit(0)
'''
train_model = my_model.create_train_model(len(char2id))
train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
checkpointer = ModelCheckpoint(
    filepath="my_trained_model/poem_s2s_weights.h5", verbose=1, save_best_only=False, period=1, save_weights_only=True)
train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size, epochs=epochs, callbacks=[checkpointer])
train_model.save('my_trained_model/poem_s2s_model.h5')
'''
