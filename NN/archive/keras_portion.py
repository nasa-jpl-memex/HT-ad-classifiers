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
maxlen = 512

# max sentences allowed in a doc, any additional are thrown out
max_sentences = 30

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

    forward_sent = LSTM(128, return_sequences=False, dropout_W=0.35, dropout_U=0.35, consume_less='gpu')(embedded)
    backward_sent = LSTM(128, return_sequences=False, dropout_W=0.35, dropout_U=0.35, consume_less='gpu', go_backwards=True)(embedded)

    sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
    sent_encode = Dropout(0.3)(sent_encode)

    encoder = Model(input=in_sentence, output=sent_encode)
    encoded = TimeDistributed(encoder)(document)

    forwards = LSTM(80, return_sequences=False, dropout_W=0.35, dropout_U=0.35, consume_less='gpu')(encoded)
    backwards = LSTM(80, return_sequences=False, dropout_W=0.35, dropout_U=0.35, consume_less='gpu', go_backwards=True)(encoded)

    merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
    output = Dropout(0.3)(merged)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(input=document, output=output)

    if checkpoint:
        model.load_weights(os.path.join(model_dir, "checkpoints", checkpoint))

    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'checkpoints/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                               monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    history = LossHistory()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, 
    	nb_epoch=25, shuffle=True, callbacks=[earlystop_cb,check_cb, history])

    # just showing access to the history object