import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Input, multiply
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Permute
import seq2seq
from seq2seq.models import AttentionSeq2Seq
lstm_hidden_dim = 512
word_size = 400
encoder_seq_length = 14
decoder_seq_length = 14


def attention_3d_block(inputs):
    print(np.shape(inputs))
    a = Permute((2, 1))(inputs)

    print(np.shape(a))
    a = Dense(decoder_seq_length, activation='softmax')(a)
    print(np.shape(a))
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    print(np.shape(output_attention_mul))
    return output_attention_mul


def create_super_model(dic_size):
    encoder_inputs = Input(shape=(None, ))
    shared_char_embedding = Embedding(
        input_dim=dic_size, output_dim=word_size, input_length=encoder_seq_length)
    encoder = Bidirectional(LSTM(lstm_hidden_dim, return_sequences=True))

    en_embedding_res = shared_char_embedding(encoder_inputs)
    encoder_outputs = encoder(en_embedding_res)
    att_outputs = attention_3d_block(encoder_outputs)

    decoder_lstm = Bidirectional(LSTM(
        lstm_hidden_dim, return_sequences=True))
    decoder_dense = Dense(dic_size, activation='softmax')

    decoder_outputs = decoder_lstm(att_outputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    model.summary()
    return model


def create_att_model(dic_size):
    model = AttentionSeq2Seq(input_dim=word_size, input_length=encoder_seq_length,
                             hidden_dim=lstm_hidden_dim, output_length=decoder_seq_length, output_dim=word_size)
    f_model = Sequential()
    f_model.add(Embedding(input_dim=dic_size, output_dim=word_size))
    f_model.add(model)
    f_model.add(Dense(dic_size, activation='softmax'))
    f_model.summary()
    return f_model


def create_train_model(dic_size):
    encoder_inputs = Input(shape=(None,))
    shared_char_embedding = Embedding(
        input_dim=dic_size, output_dim=word_size)
    en_embedding_res = shared_char_embedding(encoder_inputs)
    encoder = Bidirectional(LSTM(lstm_hidden_dim, return_state=True))
    encoder_outputs, state_h, state_c = encoder(en_embedding_res)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    de_embedding_res = shared_char_embedding(decoder_inputs)
    decoder_lstm = LSTM(
        lstm_hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        de_embedding_res, initial_state=encoder_states)
    decoder_dense = Dense(dic_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model(inputs=[encoder_inputs,
                          decoder_inputs], outputs=decoder_outputs)
    model.summary()
    return model


def create_infer_encoder_model(dic_size):
    encoder_inputs = Input(shape=(None,))
    shared_char_embedding = Embedding(
        input_dim=dic_size, output_dim=word_size)
    en_embedding_res = shared_char_embedding(encoder_inputs)
    encoder = Bidirectional(LSTM(lstm_hidden_dim, return_state=True))
    encoder_outputs, state_h, state_c = encoder(en_embedding_res)
    encoder_states = [state_h, state_c]
    model = Model(encoder_inputs, encoder_states)
    model.summary()
    return model


def create_infer_decoder_model(dic_size):
    decoder_inputs = Input(shape=(None,))
    shared_char_embedding = Embedding(
        input_dim=dic_size, output_dim=word_size)
    de_embedding_res = shared_char_embedding(decoder_inputs)
    decoder_lstm = LSTM(
        lstm_hidden_dim, return_sequences=False, return_state=True)
    decoder_state_input_h = Input(shape=(lstm_hidden_dim,))
    decoder_state_input_c = Input(shape=(lstm_hidden_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        de_embedding_res, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(dic_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([decoder_inputs] +
                  decoder_states_inputs, [decoder_outputs] + decoder_states)
    model.summary()
    return model
