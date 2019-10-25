import lasagne
import theano
from core.neural_nets.neural_network import NeuralNetwork
from core.data.predicted_features import PredictedFeatures
from six import integer_types
from theano import tensor as T
import numpy as np

class AutoEncoder(NeuralNetwork):

    def __init__(self, 
        input_shape, 
        hidden_layer_sizes, 
        symmetrical=True, 
        dropout_rate=None, 
        learning_rate=0.01, 
        momentum=0.9, 
        weight_updates='sgd',
        hidden_nonlinearity=lasagne.nonlinearities.rectify,
        output_nonlinearity=lasagne.nonlinearities.linear):

        super(AutoEncoder, self).__init__()

        ipt_tensor_size = len(input_shape)
        self.input_var = eval("T.tensor%d(\'input\')") if ipt_tensor_size > 2 else T.matrix('input')
        self.target_var = eval("T.tensor%d(\'input\')") if ipt_tensor_size > 2 else T.matrix('input')
        self.symmetrical = symmetrical

        if isinstance(hidden_layer_sizes, integer_types):
            hidden_layer_sizes = [hidden_layer_sizes]

        if symmetrical:
            self.bottleneck_idx = len(hidden_layer_sizes) - 1
            hidden_layer_sizes.extend( [i for i in reversed(hidden_layer_sizes[0:-1])] )
        else:
            assert (len(hidden_layer_sizes) % 2) == 1, 'hidden_layer_sizes must be odd! The bottleneck is assumed to be the center layer.'
            self.bottleneck_idx = len(hidden_layer_sizes) // 2

        self.hidden_layer_sizes = hidden_layer_sizes

        net = lasagne.layers.InputLayer(input_shape, input_var=self.input_var)
        h_layers = []
        l_idx = 0
        for ls in hidden_layer_sizes:
            if (dropout_rate is not None) and (l_idx > 0):
                net = lasagne.layers.DropoutLayer(net, p=dropout_rate)

            net = lasagne.layers.DenseLayer(net,
                num_units=ls,
                nonlinearity=hidden_nonlinearity)
            h_layers.append(net)

            l_idx +=1

        if dropout_rate is not None:
            lasagne.layers.DropoutLayer(net, p=dropout_rate)

        net = lasagne.layers.DenseLayer(net,
            num_units=input_shape[-1],
            nonlinearity=output_nonlinearity)

        self.net = net
        self.hidden_layers = h_layers
        self.bottleneck_layer = self.hidden_layers[self.bottleneck_idx]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.bottleneck_fn = None
        self.weight_updates = weight_updates

    def _unset_for_saving(self):
        self.bottleneck_fn = None

    def compile(self):
        
        preds = lasagne.layers.get_output(self.net)
        train_loss = lasagne.objectives.squared_error(preds, self.target_var)
        train_loss = train_loss.mean()

        params = lasagne.layers.get_all_params(self.net, trainable=True)

        if self.weight_updates == 'nesterov':
            updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=self.learning_rate, momentum=self.momentum)

        if self.weight_updates == 'sgd':
            updates = lasagne.updates.sgd(train_loss, params, learning_rate=self.learning_rate)

        test_pred = lasagne.layers.get_output(self.net, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_pred, self.target_var)
        test_loss = test_loss.mean()

        h_preds = []
        for h in self.hidden_layers:
            h_preds.append( lasagne.layers.get_output(h, deterministic=True) )

        print("Compiling functions...")

        train_fn = theano.function([self.input_var, self.target_var], train_loss, updates=updates)
        val_fn = theano.function([self.input_var, self.target_var], [test_pred, test_loss])
        test_fn = theano.function([self.input_var], [test_pred])

        for p in h_preds:
            self.feature_fn.append( theano.function([self.input_var], [p]) )

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.val_fn = val_fn
        self.bottleneck_fn = self.feature_fn[self.bottleneck_idx]

        print("...Done!")

    def get_bottleneck_features(self, input_data, batch_size, h5file=None):
        """
        This method returns the activations from the hidden_layer for all samples in input_data.
        """
        #print hidden_layer, len(self.feature_fn)
        if self.bottleneck_idx > len(self.feature_fn) -1 :
            raise Exception, "Feature extraction function for layer %d is not compiled!" % hidden_layer

        return self._predict(input_data, self.bottleneck_fn, batch_size, h5file=h5file)    

def simple_test():
    from core.data.data_batch_gen import TensorAutoEncoderBatches
    from sklearn.datasets import load_digits

    X, _ = load_digits(return_X_y=True)

    X = X.astype('float32')
    
    X_train, X_test = train_test_split(X, test_size=0.2)

    X_train, X_val = train_test_split(X_train, test_size=0.2)

    net = AutoEncoder((None, X_train.shape[1]), [32, 16], True)

    train_data = TensorAutoEncoderBatches(X_train)
    val_data = TensorAutoEncoderBatches(X_val)
    test_data = TensorAutoEncoderBatches(X_test)

    net.fit(train_data, val_data, max_epochs=1000, early_stopping=100, batch_size=500)

    #preds, h5 = net.predict(test_data, batch_size=500, h5file='teste.h5')

    preds, h5 = net.get_bottleneck_features(test_data, 500, 'teste.h5')

    #print preds

    h5.close()

def stft_test():
    from core.data.dft_batches import DFTAutoEncoderBatches
    from core.data.filelists import parse_filelist
    from sklearn.preprocessing import StandardScaler
    import dill
    import copy

    X, _ = parse_filelist('gtzan_folds/gtzan44_labels.txt')

    X = np.array(X)

    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

    X_train_orig = copy.deepcopy(X_train)

    X_train, X_val = train_test_split(X_train, test_size=0.2, shuffle=False)

    net = AutoEncoder((None, 128), [64], True, dropout_rate=None, learning_rate=0.01, weight_updates='nesterov')

    train_data_ss = DFTAutoEncoderBatches(X_train, in_db=True, dft_len=2048, window_step=1024, window_len=2048, standardizer=None, spec_directory=None, mel_bins=128)

    # print('fitting standard scaler...')
    # ss = StandardScaler()
    # for X, Y in train_data_ss.next(batch_size=20000):
    #     ss.partial_fit(X,Y)

    # dill.dump(ss, open('scaler.dill', 'w'))
  
    print('loading standard scaler...')
    ss = dill.load(open('scaler.dill'))

    train_data = DFTAutoEncoderBatches(X_train, in_db=True, dft_len=2048, window_step=1024, window_len=2048, standardizer=ss, mel_bins=128)
    validation_data = DFTAutoEncoderBatches(X_val, in_db=True, dft_len=2048, window_step=1024, window_len=2048, standardizer=ss, mel_bins=128)

    print('training...')
    net.fit(train_data, validation_data, max_epochs=1, early_stopping=None, batch_size=20000, debug=True, shuffle_batches=False)

    print('testing...')
    test_data = DFTAutoEncoderBatches(X_test, in_db=True, dft_len=2048, window_step=1024, window_len=2048, standardizer=ss, mel_bins=128)

    test_loss = net.get_loss(test_data, batch_size=20000)

    print('loss on test data: %.2f' % test_loss)

    train_data = DFTAutoEncoderBatches(X_train_orig, in_db=True, dft_len=2048, window_step=1024, window_len=2048, standardizer=ss, mel_bins=128)

    train_feats = net.get_bottleneck_features(train_data, batch_size=20000, h5file='ae_train_features.dill')
    print train_feats
    test_feats = net.get_bottleneck_features(test_data, batch_size=20000, h5file='ae_test_features.dill')
    print test_feats

    params = dill.dump(net.get_params(), open('weights.dill', 'w'))

    test_feats[1].close()
    train_feats[1].close()

    dill.dump(net, open('net.dill', 'w'))

    net = None

    net = dill.load(open('net.dill'))

    print ('final loss: %.3f' % net.get_loss(test_data, batch_size=20000))

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    #simple_test()

    stft_test()

