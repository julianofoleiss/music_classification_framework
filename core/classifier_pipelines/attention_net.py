import time

from classifier_pipeline import ClassifierPipeline
from core.utils.encodings import label_matrix_to_list, get_label_array
from core.single_split_gridsearch import SingleSplitGridSearch

from sklearn.model_selection import ShuffleSplit, train_test_split
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import types
import tempfile

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

#This is the Early Stopping from Keras. However, it restores the
#best model if training ends while patience is active!

from keras.callbacks import Callback

class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            if self.verbose > 0:
                print("EarlyStopping: found a better solution: %.4f. Previous best: %.4f. Waited for %d epochs!" % (current, self.best, self.wait))
            
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # if self.restore_best_weights:
                #     if self.verbose > 0:
                #         print('Restoring model weights from the end of '
                #               'the best epoch')
                #     self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

        if self.restore_best_weights:
            print('Restoring model weights from the end '
                'of the best epoch')
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value




def make_keras_pickable():
    print('making keras pickable')
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        import keras
        import tensorflow as tf
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

class AttentionNet(object):

    def __init__(self, n_features, n_classes, n_frames, input_encoder='ae', bottleneck_size=64, 
        input_encoder_neurons=[50], input_encoder_layers=['dense'],
        input_encoder_activations=['relu'], input_encoder_conv_kernel_sizes=[16],    
        pooling_stride=20, 
        classifier_neurons=[50], classifier_layers=['dense'], classifier_activations=['relu'],
        conv_kernel_sizes=[16], 
        epochs=500, batch_size=20, patience=40, label_dict=None, scratch_dir='./', temp_output='temp',
        gpu='0'):

        self.input_encoder=input_encoder
        self.bottleneck_size = bottleneck_size
        self.input_encoder_neurons=input_encoder_neurons
        self.input_encoder_layers=input_encoder_layers
        self.input_encoder_activations=input_encoder_activations
        self.input_encoder_conv_kernel_sizes=input_encoder_conv_kernel_sizes
        self.pooling_stride = pooling_stride
        self.classifier_neurons = classifier_neurons
        self.classifier_layers = classifier_layers
        self.classifier_activations = classifier_activations
        self.conv_kernel_sizes=conv_kernel_sizes
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.label_dict = label_dict
        self.scratch_dir=scratch_dir
        self.temp_output=temp_output
        self.gpu=gpu

        #input layers
        inputs = keras.Input(shape=(None, n_features))
        norm_in = keras.layers.BatchNormalization()(inputs)

        # #input encoder
        # layer = norm_in
        # cid=0
        # for l,n,a in zip(input_encoder_layers,input_encoder_neurons,input_encoder_activations):
        #     if l == 'conv':
        #         k = input_encoder_conv_kernel_sizes[cid]
        #         cid+=1
        #         layer = keras.layers.Conv1D(n, k, activation=a)(layer)
        #     elif l == 'dense':
        #         layer = keras.layers.TimeDistributed(keras.layers.Dense(n, activation=a))(layer)
        #     elif l == 'dropout':
        #         layer = keras.layers.TimeDistributed(keras.layers.Dropout(n))(layer)
        #     elif l == 'maxpool':
        #         layer = keras.layers.MaxPooling1D(pool_size=n)(layer)                
        #     else:
        #         raise ValueError('Only conv, dense, dropout and maxpool layers are supported in the input encoder.')

        # #bottleneck
        # #bottle=layer
        # bottle = keras.layers.TimeDistributed(keras.layers.Dense(bottleneck_size, activation='relu', name='bottleneck'))(layer)

        lb = keras.layers.Lambda(lambda x: x[:,:,0:26])(norm_in)
        mb = keras.layers.Lambda(lambda x: x[:,:,26:78])(norm_in)
        hb = keras.layers.Lambda(lambda x: x[:,:,78:104])(norm_in)

        clb = keras.layers.Conv1D(filters=42, kernel_size=8, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(lb)
        cmb = keras.layers.Conv1D(filters=42, kernel_size=8, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(mb)
        chb = keras.layers.Conv1D(filters=8, kernel_size=8, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(hb)

        layer = keras.layers.Concatenate()([clb,cmb,chb])

        #bottleneck
        bottle=layer
        #bottle = keras.layers.TimeDistributed(keras.layers.Dense(bottleneck_size, activation='relu', name='bottleneck'))(layer)

        def diff_pooling(inputs):
            return inputs[:,1:,:] - inputs[:,:-1,:]

        def slice_time(inputs, n_frames):
            return inputs[:,n_frames:,:]

        #calculating bottle_diff
        bottle_diff = keras.layers.Lambda(diff_pooling, name='diff_pooling')(bottle)
        #calculating bottle_diff_diff
        bottle_diff2 = keras.layers.Lambda(diff_pooling, name='diff2_pooling')(bottle_diff)

        #slicing x and dx to the length (in time) of dxx
        bottle_diff = keras.layers.Lambda(slice_time, arguments={'n_frames': 1}, name='dx_time_slicing')(bottle_diff)
        
        #with DIFF
        bottle = keras.layers.Lambda(slice_time, arguments={'n_frames':2}, name='x_time_slicing')(bottle)
        input_encoding = keras.layers.Concatenate(name='concat_x_dx_dxx')([bottle, bottle_diff, bottle_diff2])

        # # without DIFF
        #input_encoding=bottle

        if input_encoder == 'ae':
            ae_rec = keras.layers.TimeDistributed(keras.layers.Dense(n_features, activation='linear', name='ae_reconstruction'))(input_encoding)

        #Pooling + classifier

        def stdev_pooling(inputs, stride):
            import keras.backend as K

            data = inputs
            
            padding = K.shape(data)[1] % stride
            data = K.switch(padding > 0, K.temporal_padding(data, padding=(0,stride-padding)), data )
            num_windows = K.shape(data)[1] / stride
            idxs = K.arange(num_windows) * stride
            
            windows = K.map_fn(lambda w: data[:, w: (w + stride), :], idxs, dtype=K.floatx())
            windows = K.permute_dimensions(windows, (1,0,2,3))
            
            stds = K.map_fn(lambda w: K.std(w, axis=1), windows)
            
            return stds


        avg_pool = keras.layers.AveragePooling1D(strides=pooling_stride)(input_encoding)
        std_pool = keras.layers.Lambda(stdev_pooling, arguments={'stride':pooling_stride}, name='stdev_pooling')(input_encoding)

        #AVG + STDEV
        bottle_stats = keras.layers.Concatenate()([avg_pool, std_pool])
        
        # #AVG ONLY
        #bottle_stats = avg_pool
        
        features = bottle_stats

        #def diff_pooling(inputs):
        #    return inputs[:,1:,:] - inputs[:,:-1,:]

        #diff_stats = keras.layers.Lambda(diff_pooling, name='diff_pooling')(bottle_stats)

        #slicing bottle_stats to match diff_stats
        #bottle_stats = keras.layers.Lambda(lambda t: t[:,1:,:], name='feat_slicing')(bottle_stats)

        #features = keras.layers.Concatenate(name='feats_diffs_concat')([bottle_stats,diff_stats])

        #bottle_stats = avg_pool

        cid = 0
        layer = features        
        for l,n,a in zip(classifier_layers,classifier_neurons,classifier_activations):
            if l == 'conv':
                k = conv_kernel_sizes[cid]
                cid+=1
                layer = keras.layers.Conv1D(n, k, activation=a)(layer)
            elif l == 'dense':
                layer = keras.layers.TimeDistributed(keras.layers.Dense(n, activation=a))(layer)
            elif l == 'dropout':
                layer = keras.layers.TimeDistributed(keras.layers.Dropout(n))(layer)
            elif l == 'maxpool':
                layer = keras.layers.MaxPooling1D(pool_size=n)(layer)                    
            else:
                raise ValueError('Only conv, dense and dropout layers are supported in the classifier.')
            
        
        out = keras.layers.TimeDistributed(keras.layers.Dense(n_classes, activation='softmax', name='classifier'))(layer)

        # if True:
        #     d1 = keras.layers.TimeDistributed(keras.layers.Dense(50, activation='relu', name='dense1'))(avg_pool)
        #     d3 = keras.layers.TimeDistributed(keras.layers.Dense(n_classes, activation='softmax', name='classifier'))(d1)

        # if False:
        #     avg_pool = keras.layers.AveragePooling1D(strides=20)(bottle)
        #     d1 = keras.layers.Conv1D(64, 16, activation='relu', name='conv1')(avg_pool)
        #     d3 = keras.layers.TimeDistributed(keras.layers.Dense(n_classes, activation='softmax', name='classifier'))(d1)

        def sum_vote(x):
            import keras
            return keras.backend.sum(x,1)

        p = keras.layers.Lambda(sum_vote, name='sum_vote')(out)

        if input_encoder == 'ae':
            m1 = keras.Model(inputs=inputs, outputs=[ae_rec,p])
        elif input_encoder == 'mlp':
            m1 = keras.Model(inputs=inputs, outputs=p)

        self.net = m1
        #self.relevance = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('relevance').output)
        #self.class_mlp = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('class_mlp').output)
        #self.relevance_exp = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('relevance_exp').output)
        #self.Ap = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('Ap').output)

        #self.net.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
        #    metrics=[keras.metrics.categorical_accuracy])

        if input_encoder == 'ae':
            self.net.compile(optimizer=tf.train.AdamOptimizer(0.001), 
                loss=['mean_squared_error','categorical_crossentropy'],
                metrics=[keras.metrics.categorical_accuracy])
        elif input_encoder == 'mlp':
            self.net.compile(optimizer=tf.train.AdamOptimizer(0.001), 
                loss='categorical_crossentropy',
                metrics=[keras.metrics.categorical_accuracy])

        print(self.net.summary())

        from keras.utils import plot_model
        model_filename = self.scratch_dir + '/' + self.temp_output + '_model.png'
        plot_model(self.net, to_file=model_filename, show_shapes=True)        


        # self.net.compile(optimizer=keras.optimizers.Adagrad(0.01), loss='categorical_crossentropy',
        # metrics=[keras.metrics.categorical_accuracy])

    def fit(self, X, Y):
        #rs=42
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=999)

        print('Training network...', X_train.shape, Y_train.shape )

        temp_model = self.scratch_dir + '/current_best.model'

        #rels = self.relevance.predict(X_train, batch_size=self.batch_size)

        #print(rels[:10,:])

        #MC = keras.callbacks.ModelCheckpoint(temp_model, 
        #        monitor='val_loss', save_best_only=True, verbose=True,
        #        save_weights_only=True)

        #limit how much GPU memory is being used by tensorflow (45% of total GPU memory)
        #also tells tensorflow to use GPU 0
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.visible_device_list = self.gpu
        set_session(tf.Session(config=config))

        print('Training parameters: Max Epochs: %d, Batch Size: %d, ES Patience: %d' % 
            (self.epochs, self.batch_size, self.patience))

        ES = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1, min_delta=0.005, restore_best_weights=True)

        if self.input_encoder == 'ae':
            H = self.net.fit(X_train, [X_train,Y_train], epochs=self.epochs, batch_size=self.batch_size,
                callbacks=[ES],validation_data=(X_val, [X_val, Y_val]))
        elif self.input_encoder == 'mlp':
            H = self.net.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                callbacks=[ES],validation_data=(X_val, Y_val))

        #self.net.load_weights(temp_model)

        # Plot training & validation loss values
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        
        if self.input_encoder == 'ae':
            plt.plot(H.history['val_time_distributed_2_loss'])
            plt.plot(H.history['val_sum_vote_loss'])

        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        legend = ['Train (Total)', 'Val (Total)']

        if self.input_encoder == 'ae':
            legend.extend(['Val (AE)', 'Val(Classifier)'])

        plt.legend(legend, loc='upper left')
        history_filename = self.scratch_dir + '/' + self.temp_output + '_history.png'
        plt.savefig(history_filename)

        return self

    def predict(self, X):
        
        #limit how much GPU memory is being used by tensorflow (40% of total GPU memory)
        #also tells tensorflow to use GPU 0
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        # config.gpu_options.visible_device_list = "0"
        # set_session(tf.Session(config=config))        

        if X.ndim != 3:
            X = X.reshape((-1, self.n_frames,self.n_features))

        Y = self.net.predict(X, batch_size=self.batch_size)

        if self.input_encoder == 'ae':
            Y = Y[1]

        #rels = self.relevance.predict(X, batch_size=self.batch_size)
        #print('relevancias', rels.shape)
        #print(rels[0])
        #print(rels[0][:5])
        #print(rels[:10,:])
        #print(rels[-10:-1,:])

        # re = self.relevance_exp.predict(X, batch_size=self.batch_size)
        # print('relevancias_expandidas', re.shape)
        # print(re[0][:5])
        # #print(re[:10,:])
        # #print(re[-10:-1,:])

        #pr = self.class_mlp.predict(X, batch_size=self.batch_size)
        #print('predicoes', pr.shape)
        #print(pr[0][:5])
        #print(pr[:10,:])
        #print(pr[-10:-1,:])        

        #ap = self.Ap.predict(X, batch_size=self.batch_size)
        #print('Ap', ap.shape)
        #print(ap[0][:5])
        #print(ap[:10,:])
        #print(ap[-10:-1,:])           


        #np.savetxt('rels.txt', rels.reshape((-1, self.n_frames)))

        #print (self.net.get_weights())
        print (Y, Y.shape)
        print(np.argmax(Y, axis=1))
        print(np.unique(np.argmax(Y, axis=1), return_counts=True))

        if self.label_dict is not None:
            Y = label_matrix_to_list(Y, self.label_dict)

        return Y


class AttentionNetClassifier(ClassifierPipeline):

    def __init__(self, params):
        super(AttentionNetClassifier, self).__init__(params)

    def _fit(self, train_features, train_labels, track_idxs=None, pf=None):
        
        if track_idxs == None:
            raise ValueError, "track_idxs is needed to compute the train label list correctly!"

        if pf==None:
            raise ValueError, "pf is needed for this to work!"

        tracks_nframes = np.array(pf.track_nframes)[track_idxs]      

        input_encoder = self.params['attention_net']['input_encoder']
        bottleneck_size = self.params['attention_net']['bottleneck_size']

        input_encoder_neurons = self.params['attention_net']['input_encoder_neurons']
        input_encoder_layers = self.params['attention_net']['input_encoder_layers']
        input_encoder_activations = self.params['attention_net']['input_encoder_activations']
        input_encoder_conv_kernel_sizes = self.params['attention_net']['input_encoder_conv_kernel_sizes']

        pooling_stride = self.params['attention_net']['pooling_stride']
        classifier_neurons = self.params['attention_net']['classifier_neurons']
        classifier_layers = self.params['attention_net']['classifier_layers']
        classifier_activations = self.params['attention_net']['classifier_activations']
        conv_kernel_sizes = self.params['attention_net']['conv_kernel_sizes']

        temp_output = self.params['attention_net']['temp_output']

        gpu = self.params['attention_net']['gpu']
        
        assert len(input_encoder_neurons) == len(input_encoder_layers) == len(input_encoder_activations), \
        'Input encoder layers specification is invalid. You must provide the type of layers and activations ' + \
        'for each layer in the input encoder.'        

        assert len(classifier_neurons) == len(classifier_layers) == len(classifier_activations), \
        'Classifier layers specification is invalid. You must provide the type of layers and activations ' + \
        'for each layer in the classifier.'

        if conv_kernel_sizes is not None:
            assert len(input_encoder_conv_kernel_sizes) >= input_encoder_layers.count('conv'), \
            'The kernel size must be provided for each convolutional layer in the input encoder.'

        if conv_kernel_sizes is not None:
            assert len(conv_kernel_sizes) >= classifier_layers.count('conv'), \
            'The kernel size must be provided for each convolutional layer in the classifier.'

        assert input_encoder == 'mlp' or input_encoder == 'ae', 'The input encoder must be either mlp or ae.'

        epochs = self.params['attention_net']['epochs']
        batch_size = self.params['attention_net']['batch_size']
        patience = self.params['attention_net']['patience']
        scratch_dir = self.params['general']['scratch_directory']

        #label_dict = ....
        exec( open(self.params['general']['label_dict_file']).read() )

        #print(train_labels, train_labels.shape)

        #train_labels = label_matrix_to_list(train_labels, label_dict)       
        #train_labels = get_label_array(train_labels, tracks_nframes)

        n_frames = tracks_nframes[0]
        n_classes = train_labels.shape[1]
        n_features = train_features.shape[1]

        #parameter_combos = dict(svm__C=Cs, 
        #    svm__gamma=gammas)

        #estimator = SingleSplitGridSearch(clf, 
        #    parameter_combos, 
        #    self.params['svm_anova']['num_workers'],
        #    self.params['svm_anova']['grid_verbose'], 
        #    refit=True)

        attnet = AttentionNet(n_features, n_classes, n_frames, input_encoder=input_encoder, bottleneck_size=bottleneck_size,
            input_encoder_neurons=input_encoder_neurons, input_encoder_layers=input_encoder_layers,
            input_encoder_activations=input_encoder_activations,input_encoder_conv_kernel_sizes=input_encoder_conv_kernel_sizes,
            pooling_stride=pooling_stride, classifier_neurons=classifier_neurons,
            classifier_layers=classifier_layers, classifier_activations=classifier_activations,
            conv_kernel_sizes=conv_kernel_sizes, epochs=epochs, batch_size=batch_size,
            patience=patience, label_dict=label_dict, scratch_dir=scratch_dir, temp_output=temp_output,
            gpu=gpu)

        print(train_features.shape, train_labels.shape)

        train_features = train_features.reshape((-1, n_frames, n_features))
        #train_labels = np.array([i for i in train_labels[::n_frames]])

        print(train_features.shape, train_labels.shape)

        assert train_features.shape[0] == train_labels.shape[0]

        T0 = time.time()
        attnet.fit(train_features, train_labels)
        T1 = time.time()

        print "model training took %f seconds" % (T1-T0)

        return attnet

 # def __init__(self, n_features, n_classes, n_frames, class_mlp_hidden_neurons=100, 
    #         relevance_mlp_hidden_neurons=100, epochs=500, batch_size=20, patience=40, label_dict=None,
    #         scratch_dir='./'):

    #     self.class_mlp_hidden_neurons = class_mlp_hidden_neurons
    #     self.relevance_mlp_hidden_neurons = relevance_mlp_hidden_neurons
    #     self.n_features = n_features
    #     self.n_classes = n_classes
    #     self.n_frames = n_frames
    #     self.epochs = epochs
    #     self.batch_size = batch_size
    #     self.patience = patience
    #     self.label_dict = label_dict
    #     self.scratch_dir=scratch_dir

    #     #input layers
    #     inputs = keras.Input(shape=(None, n_features))
    #     norm_in = keras.layers.BatchNormalization()(inputs)
        
    #     #predictive model
    #     d1 = keras.layers.Dense(300, activation='relu', kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros')
    #     x = keras.layers.TimeDistributed(d1)(norm_in)
        
    #     d1 = keras.layers.Dense(n_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #         activation='softmax') 
            
    #         #, activity_regularizer=keras.regularizers.l1(0.0001))        

    #     p = keras.layers.TimeDistributed(d1, name='class_mlp')(x)

    #     #relevance network
    #     dr = keras.layers.Dense(50, activation='relu', kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros')
        
    #     y = keras.layers.TimeDistributed(dr)(norm_in)

    #     dr = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros')

    #     a = keras.layers.TimeDistributed(dr)(y)

    #     def to_prob(x):
    #         import keras
    #         n = x / keras.backend.sum(x, 1, keepdims=True)
    #         return n

    #     a = keras.layers.Lambda(to_prob, name='relevance')(a)

    #     def multiply_weighted(x):
    #         import keras
    #         a = x[0]
    #         p = x[1]
    #         #b = keras.layers.multiply(a,p)
    #         b = keras.layers.multiply(x)
    #         b = keras.backend.sum(p, 1) # b?
    #         return b

    #     Ap = keras.layers.Lambda(multiply_weighted, name='Ap')([a, p])
    #     m1 = keras.Model(inputs, Ap)

    #     self.net = keras.Model(inputs, Ap)
    #     self.relevance = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('relevance').output)
    #     self.class_mlp = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('class_mlp').output)
    #     #self.relevance_exp = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('relevance_exp').output)
    #     self.Ap = keras.Model(inputs=self.net.input, outputs=self.net.get_layer('Ap').output)

    #     #self.net.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
    #     #    metrics=[keras.metrics.categorical_accuracy])

    #     self.net.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
    #         metrics=[keras.metrics.categorical_accuracy])

    #     print(self.net.summary())

    #     from keras.utils import plot_model
    #     plot_model(self.net, to_file='model.png', show_shapes=True)        

    #     # self.net.compile(optimizer=keras.optimizers.Adagrad(0.01), loss='categorical_crossentropy',
    #     # metrics=[keras.metrics.categorical_accuracy])

    # def fit(self, X, Y):

    #     X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

    #     print('Training network...', X_train.shape, Y_train.shape )

    #     #self.net.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
    #     #    callbacks=[keras.callbacks.EarlyStopping(patience=self.patience, monitor='val_categorical_accuracy')],
    #     #    validation_data=(X_val, Y_val))

    #     temp_model = self.scratch_dir + '/current_best.model'

    #     rels = self.relevance.predict(X_train, batch_size=self.batch_size)

    #     print(rels[:10,:])

    #     self.net.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
    #         callbacks=[keras.callbacks.ModelCheckpoint(temp_model, 
    #             monitor='val_loss', save_best_only=True, verbose=True,
    #             save_weights_only=True)],validation_data=(X_val, Y_val))
        
    #     self.net.load_weights(temp_model)

    #     return self

