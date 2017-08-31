import pandas as pd

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, Lambda, merge
from keras.layers import Embedding, TimeDistributed
import numpy as np
import tensorflow as tf
import re
from keras import backend as K
import keras.callbacks
import sys
import os
import json
import ujson

basedir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))

date = "11-28-16"

if not date in os.listdir(os.path.join(basedir, "modeling")):
    os.mkdir(os.path.join(basedir, "modeling", date))

model_dir = os.path.join(basedir, "modeling", date)
preprocessing_dir = os.path.join(basedir, "preprocessing", "fall-2016")
data_dir = os.path.join(basedir, "raw-data", "fall-2016")

checkpoint = "memex-char-lstm-fall.00-0.72.hdf5"

test_clusters = None

with open(os.path.join(model_dir, "cluster_splits"), "r") as f:
    cluster_splits = eval(f.read())
    test_clusters = cluster_splits["test"]

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_char_encodings():
    with open(os.path.join(preprocessing_dir, "encodings"), "r") as f:
        return eval(f.read())

char_indices = load_char_encodings()



def encode_and_trim(docs, X, maxlen, max_sentences):
    """
    Replace -1's in vector representation of chars with encodings in reverse order (-1s toward beginning indicate 
    character length was less than max allowed)
    """
    for i, doc in enumerate(docs):
        for j, sentence in enumerate(doc):
            if j < max_sentences:
                for t, char in enumerate(sentence[-maxlen:]):
                    X[i, j, (maxlen-1-t)] = char_indices[char]
    return X

def get_cluster_and_text_and_id(json_file, outfile):
    """
    Get all ads associated with a set of clusters (train or test set), write to file
    """
    with open(os.path.join(model_dir, outfile), "w") as out:
        with open(os.path.join(data_dir, json_file), "r") as f:

            for idx, ad in enumerate(f.readlines()):
                ad = ujson.loads(ad)
                if ad["cluster_id"] in test_clusters: 
                    if "doc_id" in ad:
                        if "class" in ad:
                            out.write(str([ ad["doc_id"], ad["cluster_id"], ad["class"] ]) + "\n")
                        else:
                            out.write(str([ ad["doc_id"], ad["cluster_id"] ]) + "\n")
                    else:
                        # print("missing doc_id")
                        pass


get_cluster_and_text_and_id("CP1_train_ads.json", "test_eval")

maxlen = 200
max_sentences = 25

def build_lists(file, eval_docs, eval_classes):
    with open(os.path.join(data_dir, file)) as f:
        ads = f.readlines()

        for ad in ads:
            ad = ujson.loads(ad)
            if ad["cluster_id"] in test_clusters:
                eval_sentences = re.split('(\?+|\n+|\.+)', str(ad["extracted_text"]))
                eval_sentences = [re.sub("(\r|\n|\t)", " ", clean_str(sent)).lower() for sent in eval_sentences if len(sent) > 4]
                eval_docs.append(eval_sentences)
                eval_classes.append(ad["class"])

eval_docs = []
eval_classes = []

build_lists("CP1_train_ads.json", eval_docs, eval_classes)
X_eval = np.ones((len(eval_docs), max_sentences, maxlen), dtype=np.int64) * -1
X_eval = encode_and_trim(eval_docs, X_eval, maxlen, max_sentences)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


def binarize(x, sz=46):
    """
    Used in creation of Lambda layer to create a one hot encoding of sentence characters on the fly. 
    x : tensor of dimensions (maximum sentence length, ) TODO
    sz : number of unique characters in the corpus
    tf.to_float casts a tensor to type "float32"
    """
    one_hot = tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1)
    return tf.to_float(one_hot)


def binarize_outshape(in_shape):
    """
    
    """
    return in_shape[0], in_shape[1], 46


def max_1d(x):
    """
    
    """
    return K.max(x, axis=1)


with tf.device("/gpu:0"): 
    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    pool_length = 2

    # document input -> 15 x 512
    document = Input(shape=(max_sentences, maxlen), dtype='int64')

    # sentence input -> 512,
    in_sentence = Input(shape=(maxlen,), dtype='int64')

    # binarize function creates a onehot encoding of each character index
    embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)


    for i in range(len(nb_filter)):
        embedded = Convolution1D(nb_filter=nb_filter[i],
                                filter_length=filter_length[i],
                                border_mode='valid',
                                activation='relu',
                                init='glorot_normal',
                                subsample_length=1)(embedded)

        embedded = Dropout(0.1)(embedded)
        embedded = MaxPooling1D(pool_length=pool_length)(embedded)

with tf.device("/gpu:1"): 
    forward_sent = LSTM(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(embedded)
    backward_sent = LSTM(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(embedded)



    sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
    sent_encode = Dropout(0.1)(sent_encode)

    encoder = Model(input=in_sentence, output=sent_encode)
    encoded = TimeDistributed(encoder)(document)

with tf.device("/gpu:2"): 
    forwards = LSTM(80, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(encoded)
    backwards = LSTM(80, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)


with tf.device("/gpu:3"): 
    merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
    output = Dropout(0.2)(merged)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.2)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(input=document, output=output)

    if checkpoint:
        print("STARTING FROM CHECKPOINT")
        model.load_weights(os.path.join(model_dir, "checkpoints", checkpoint))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

proba = model.predict(X_eval, batch_size=128)

def join_cluster_to_probability(probs):
    """
    After probability assignments are made, rejoin them with a cluster id, allowing for prediction at the cluster level
    """
    cluster_probs = {}
    with open(os.path.join(model_dir, "test_eval"), "r") as f:
        for i, doc in enumerate(f.readlines()):
            doc = eval(doc)
            cluster_id = doc[1]
            if cluster_id in cluster_probs:
                cluster_probs[cluster_id]["scores"].append(float(probs[i][0]))
            else:
                cluster_probs[cluster_id] = {} 
                cluster_probs[cluster_id]["scores"] = [float(probs[i][0])]

            if len(doc) == 3:
                cluster_probs[cluster_id]["label"] = doc[2]
#                 except ValueError("don't have labels...")

    with open(os.path.join(model_dir, "test_ad_predictions_by_cluster"), "w") as out:
        out.write(json.dumps(cluster_probs, indent=4))  

join_cluster_to_probability(proba)

