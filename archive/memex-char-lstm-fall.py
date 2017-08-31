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
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
# from pylab import rcParams

basedir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))

date = "11-28-16"

if not date in os.listdir(os.path.join(basedir, "modeling")):
	os.mkdir(os.path.join(basedir, "modeling", date))

model_dir = os.path.join(basedir, "modeling", date)
preprocessing_dir = os.path.join(basedir, "preprocessing", "fall-2016")
data_dir = os.path.join(basedir, "raw-data", "fall-2016")

# checkpoint = "memex-char-lstm-fall.00-0.72.hdf5"
checkpoint = None

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


def build_id_lookup(infile, outfile, limit=10000000):
    """
    Output dictionary of cluster, doc_id assignments
    :param infile: raw data (each line is json doc representing an ad)
    :param outfile: {"<cluster_id1>" : ["<doc_id1>", "<doc_id2>", ...], "<cluster_id2>" : ["<doc_id3>", "<doc_id4>", ...]}
    """
    clusters = {}
    with open(os.path.join(preprocessing_dir, outfile), "w") as out:
        with open(os.path.join(data_dir,infile)) as f:
            for idx,ad in enumerate(f):
                if idx < limit:
                    ad = json.loads(ad)
                    if "doc_id" in ad:
                        if ad["cluster_id"] in clusters:
                            clusters[ad["cluster_id"]].append(ad["doc_id"])
                        else:
                            clusters[ad["cluster_id"]] = [ad["doc_id"]]
                else:
                    break
            out.write(json.dumps(clusters, sort_keys=True, indent=4))

# build_id_lookup("CP1_train_ads.json", "training_ids")


def read_clusters(file):
    """
    Read in clusters and doc_ids generated from build_id_lookup function
    """  
    with open(os.path.join(preprocessing_dir, file)) as f:
        clusters = eval(f.read())
        return clusters

# clusters_lookup = read_clusters("training_ids")

# indices = np.arange(len(clusters_lookup.keys()))

# np.random.shuffle(indices)

# clusters = np.array(list(clusters_lookup.keys()))[indices]

# train_clusters = clusters[:int(round(len(clusters)*.8, 0))]
# test_clusters = clusters[int(round(len(clusters)*.8, 0)):]

# with open(os.path.join(model_dir, "cluster_splits"), "w") as out:
#     splits = {}
#     splits["train"] = list(train_clusters)
#     splits["test"] = list(test_clusters)
#     out.write(json.dumps(splits, indent=4))


########################################################################

train_clusters = None
test_clusters = None

with open(os.path.join(model_dir, "cluster_splits"), "r") as f:
    cluster_splits = eval(f.read())
    train_clusters = cluster_splits["train"]
    test_clusters = cluster_splits["test"]

train_docs, test_docs = [], []
train_classes, test_classes = [], []

def build_lists(file, train_docs, test_docs, train_classes, test_classes):
    num_sentences = {}
    len_sentences = {}
    with open(os.path.join(data_dir, file)) as f:
        ads = f.readlines()

        for idx, ad in enumerate(ads):
            ad = ujson.loads(ad)
            if ad["cluster_id"] in train_clusters:
                train_sentences = re.split('(\?+|\n+|\.+)', str(ad["extracted_text"]))
                train_sentences = [re.sub("(\r|\n|\t)", " ", clean_str(sent)).lower() for sent in train_sentences if len(sent) > 4]
                train_sentences = [x for x in train_sentences if x is not '']
                train_docs.append(train_sentences)
                train_classes.append(ad["class"])
            elif ad["cluster_id"] in test_clusters:
                test_sentences = re.split('(\?+|\n+|\.+)', str(ad["extracted_text"]))
                test_sentences = [re.sub("(\r|\n|\t)", " ", clean_str(sent)).lower() for sent in test_sentences if len(sent) > 4]
                test_sentences = [x for x in test_sentences if x is not '']
                
                if len(test_sentences) in num_sentences:
                    num_sentences[len(test_sentences)] += 1
                else:
                    num_sentences[len(test_sentences)] = 1

                test_docs.append(test_sentences)
                test_classes.append(ad["class"])

            if idx % 5000 == 0:
                print(idx)

    with open("sentence_lengths", "w") as lengths:
        lengths.write(json.dumps(num_sentences, indent=4))


# build_lists("CP1_train_ads.json", train_docs, test_docs, train_classes, test_classes)

# with open("train_docs", "w") as out:
#     out.write(str(train_docs))

# with open("test_docs", "w") as out:
#     out.write(str(test_docs))

# with open("train_classes", "w") as out:
#     out.write(str(train_classes))

# with open("test_classes", "w") as out:
#     out.write(str(test_classes))

# print('STOP')

############################################################################

with open("train_docs", "r") as _in:
    train_docs = eval(_in.read())

with open("test_docs", "r") as _in:
    test_docs = eval(_in.read())

with open("train_classes", "r") as _in:
    train_classes = eval(_in.read())

with open("test_classes", "r") as _in:
    test_classes = eval(_in.read())


# for idx, x in enumerate(test_docs):
#     if idx < 2:
#         print(x)
#     else:
#         break

# train_docs = train_docs[:60]
# test_docs = test_docs[:60]
# train_classes = train_classes[:60]
# test_classes = test_classes[:60]

txt = ''

def get_chars(docs, txt):
    for doc in docs:
        for s in doc:
            txt += s
    return txt

txt = get_chars(train_docs, txt)
txt = get_chars(test_docs, txt)
        
chars = set(txt)
print('total chars: ', len(chars))
char_indices = dict((c,i) for i,c in enumerate(chars))

print(json.dumps(char_indices, indent=4, sort_keys=True))

with open(os.path.join(preprocessing_dir,"encodings"), "w") as out:
    out.write(json.dumps(char_indices, indent=4, sort_keys=True))

# max number of characters allowed in a sentence, any additional are thrown out
maxlen = 200

# max sentences allowed in a doc, any additional are thrown out
max_sentences = 100

def load_char_encodings():
    with open(os.path.join(preprocessing_dir, "encodings"), "r") as f:
        return eval(f.read())

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

char_indices = load_char_encodings()


# Create array for vector representation of chars (512D). Filled with -1 initially
X_train = np.ones((len(train_docs), max_sentences, maxlen), dtype=np.int64) * -1
X_test = np.ones((len(test_docs), max_sentences, maxlen), dtype=np.int64) * -1

# create array of class labels
y_train = np.array(train_classes)
y_test = np.array(test_classes)

X_train = encode_and_trim(train_docs, X_train, maxlen, max_sentences)
X_test = encode_and_trim(test_docs, X_test, maxlen, max_sentences)

print('Sample chars in X:{}'.format(X_train[20, 2]))
print('y:{}'.format(y_train[12]))


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


# max number of characters allowed in a sentence, any additional are thrown out
maxlen = 200

# max sentences allowed in a doc, any additional are thrown out
max_sentences = 100

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

    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'checkpoints/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                               monitor='val_loss', verbose=0, save_best_only=False, mode='min')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    history = LossHistory()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, 
    	nb_epoch=7, shuffle=True, callbacks=[earlystop_cb,check_cb, history])

    # just showing access to the history object
    print(history.losses)
    print(history.accuracies)