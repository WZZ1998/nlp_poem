import my_model
import numpy as np
import pandas as pd
import random
from keras.models import load_model
import pickle
with open('my_trained_model/char2id.pkl', 'rb') as inp:
    char2id = pickle.load(inp)
with open('my_trained_model/id2char.pkl', 'rb') as inp:
    id2char = pickle.load(inp)

va_poems = open('val_tmp_res.txt', 'r').readlines()
list_be = []


def poem2num(poem_list):
    npoem_list = []
    for line in poem_list:
        one_line = []
        for one_char in line:
            if one_char not in char2id:
                one_line.append(random.randint(1, 1000))
            else:
                one_line.append(char2id[one_char])

        npoem_list.append(one_line)
    return npoem_list


for line in va_poems:
    poem = line.split(' ')
    list_be.append(poem[0:7] + poem[8:15])

encoder_input_data = np.array(poem2num(list_be))

att_mod = my_model.create_super_model(len(char2id))
#att_mod = load_model("my_trained_model/new_poem_s2s_model.h5")
att_mod.load_weights("my_trained_model/fin_poem_s2s_weights.h5", by_name=True)

res_file = open('val_res.txt', 'w')


def decode_seq2(input_seq):

    out = att_mod.predict(np.array([input_seq]))
    s = []

    for i in range(len(input_seq)):
        id_c = np.argmax(out[0, i, :])
        s.append(id2char[id_c])
    return s


res_l = []
for i in range(len(encoder_input_data)):
    index = i
    input_seq = encoder_input_data[index]

    decoded_sentence = decode_seq2(input_seq)
    l = list_be[index]
    l1 = l[0:7]
    l2 = l[7:14]
    res = ' '.join(l1)
    res = res + ' , '
    res = res + ' '.join(l2)
    res = res + ' . '
    k = decoded_sentence
    k1 = k[0:7]
    k2 = k[7:14]
    res = res + ' '.join(k1)
    res = res + ' , '
    res = res + ' '.join(k2)
    res = res + ' .\n'

    print(res)
    res_l.append(res)

res_file.writelines(res_l)
res_file.close()
exit(0)
