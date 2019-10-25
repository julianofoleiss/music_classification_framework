import lasagne
import theano
from theano import tensor as T
import numpy as np
import traceback
from core.data.data_batch_gen import TensorBatches, DataBatchGenerator
import time
import sys
import tables

class SigtiaNET(object):

    def __init__(self, input_shape, n_outputs, hidden_neurons=50, hidden_layers=3, dropout_rate=0.25, learning_rate=0.01, momentum=0.9, weight_updates='sgd'):
        ipt_tensor_size = len(input_shape)
        self.input_var = eval("T.tensor%d(\'input\')") if ipt_tensor_size > 2 else T.matrix('input')
        self.target_var = T.matrix('output')

        net = lasagne.layers.InputLayer(input_shape, input_var=self.input_var)
        h_layers = []

        for i in xrange(hidden_layers):
            if (dropout_rate is not None) and (i > 0):
                #print "adding dropout layer to layer %d" % i
                net = lasagne.layers.DropoutLayer(net, p=dropout_rate)

            net = lasagne.layers.DenseLayer( net, 
                num_units=hidden_neurons, 
                nonlinearity=lasagne.nonlinearities.rectify)
            h_layers.append(net)

        if dropout_rate is not None:
            lasagne.layers.DropoutLayer(net, p=dropout_rate)

        net = lasagne.layers.DenseLayer( net, 
            num_units=n_outputs, 
            nonlinearity=lasagne.nonlinearities.softmax)

        self.net = net
        self.hidden_layers = h_layers
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.train_fn = None
        self.val_fn = None
        self.test_fn = None
        self.feature_fn = []
        self.weight_updates = weight_updates

    def compile(self):
        preds = lasagne.layers.get_output(self.net)

        loss = lasagne.objectives.categorical_crossentropy(preds, self.target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.net, trainable=True)

        if self.weight_updates == 'nesterov':
            updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=self.learning_rate, momentum=self.momentum)
        
        if self.weight_updates == 'sgd':
            updates = lasagne.updates.sgd(loss, params, learning_rate=self.learning_rate)
        
        test_pred = lasagne.layers.get_output(self.net, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_pred, self.target_var)
        test_loss = test_loss.mean()

        h_preds = []
        for h in self.hidden_layers:
            h_preds.append(lasagne.layers.get_output(h, deterministic=True))

        print ("Compiling functions...")
        train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
        val_fn =  theano.function([self.input_var, self.target_var], [test_pred, test_loss])
        test_fn  =  theano.function([self.input_var], [test_pred])

        for p in h_preds:
            self.feature_fn.append( theano.function([self.input_var], [p]) )

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.val_fn = val_fn
        print ("... Done!")

    def fit(self, train_data, validation_data, max_epochs=200, early_stopping=None, batch_size=100, debug=False):

        if self.net is None:
            return False

        if type(train_data) is tuple:
            train_data = TensorBatches(train_data[0], train_data[1])
        else:
            if not issubclass(train_data.__class__, DataBatchGenerator):
                raise TypeError, "train_data should be tuple(inputs,targets) or instance derived of DataBatchGenerator"

        if type(validation_data) is tuple:
            validation_data = TensorBatches(validation_data[0], validation_data[1])
        else:
            if not issubclass(validation_data.__class__, DataBatchGenerator):
                raise TypeError, "validation_data should be tuple(inputs,targets) or instance derived of DataBatchGenerator"

        self.compile()

        best_params = self.get_params()
        best_epoch = 1
        best_val_loss = float("inf")
        n_val_loss = 0
        stopped_early = False
        start_time = 0

        for k in xrange(max_epochs):

            start_time = time.time()
	    
            print("Starting Epoch %d..." % (k+1))
            train_loss = 0
            train_batches = 0
            for train_X, train_Y in train_data.next(batch_size=batch_size):
                #print(train_X.shape)
                train_loss += self.train_fn(train_X, train_Y)
                train_batches+=1

            val_loss = 0
            val_batches = 0
            for val_X, val_Y in validation_data.next(batch_size=batch_size):
                #print(val_X.shape)
                predictions, loss = self.val_fn(val_X, val_Y)
                val_loss += loss
                val_batches += 1

            train_loss /= train_batches
            val_loss /= val_batches

            #< ou <= ????
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = k+1
                n_val_loss = 0
                best_params = self.get_params()
            else:
                n_val_loss +=1

            if early_stopping is not None:
                if n_val_loss >= early_stopping:
                    print("STOPPED EARLY! Best validation loss: %.2f in Epoch %d. Last Epoch loss: %.2f" % (best_val_loss, best_epoch, val_loss) )
                    stopped_early = True
                    break

            print("End of epoch %d!\n\tTrain loss: %.2f\n\tValidation loss: %.2f" % (k+1, train_loss, val_loss))

            if debug:
                print("Epoch took %.2f seconds!" % (time.time() - start_time))

            sys.stdout.flush()
        
        if stopped_early:
            print("End of epoch %d!\n\tTrain loss: %.2f\n\tValidation loss: %.2f" % (k+1, train_loss, val_loss))
        else:
            print("EXECUTED EVERY Epoch! Best validation loss: %.2f in Epoch %d. Last Epoch loss: %.2f" % (best_val_loss, best_epoch, val_loss) )

        self.set_params(best_params)

        print("Training done!")

        return True

    def get_params(self):
        return lasagne.layers.get_all_param_values(self.net, trainable=True)

    def set_params(self, params):
        if self.train_fn is None:
            self.compile()
        lasagne.layers.set_all_param_values(self.net, params)

    def _predict(self, test_data, pred_fn, batch_size=100, h5file=None):
        
        if type(test_data) is tuple:
            test_data = TensorBatches(test_data[0], test_data[1])
        else:
            if not issubclass(test_data.__class__, DataBatchGenerator):
                raise TypeError, "test_data should be tuple(inputs,targets) or instance derived of DataBatchGenerator"

        if h5file is not None:
            a = tables.Float32Atom()
            h5 = tables.open_file(h5file, mode="w")
            f = tables.Filters(complevel=1, complib="lzo")
            predictions = h5.create_earray(h5.root, 'features', a, (0, self.hidden_neurons), filters=f)
            predictions.attrs.h5filename = h5.filename
        else:
            predictions = None

        train_loss = 0
        train_batches = 0
  
        for test_X, _ in test_data.next(batch_size=batch_size, shuffle=False):
            #print(test_X.shape)
            preds = pred_fn(test_X)
            if predictions is None:
                predictions = preds[0]
            else:
                if h5file is not None:
                    predictions.append(preds[0])
                else:
                    predictions = np.append(predictions, preds[0], axis=0)

        if h5file is not None:
            h5.flush()

        return (predictions, h5) if h5file is not None else predictions

    def predict(self, test_data, batch_size=100, h5file=None):
        
        if self.test_fn is None:
            raise Exception, "Model not fit yet!"

        return self._predict(test_data, self.test_fn, batch_size, h5file=h5file)

    def get_features(self, test_data, hidden_layer, batch_size, h5file=None):
        #print hidden_layer, len(self.feature_fn)
        if hidden_layer > len(self.feature_fn) -1 :
            raise Exception, "Feature extraction function for layer %d is not compiled!" % hidden_layer

        return self._predict(test_data, self.feature_fn[hidden_layer], batch_size, h5file=h5file)

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    import dill

    X, Y = load_digits(return_X_y=True)

    X = X.astype('float32')

    Y_M = np.zeros((Y.shape[0], 10))
    k = np.arange(Y.shape[0])
    Y_M[k, Y] = 1
    Y = Y_M.astype('float32')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3)

    #print X_train.shape, X_val.shape, X_test.shape

    net = SigtiaNET((None, 64), 10, hidden_neurons=50, hidden_layers=3, dropout_rate=0.0)

    net.fit((X_train, Y_train), (X_val, Y_val), max_epochs=200, early_stopping=50, batch_size=500)

    predictions = net.predict((X_test, None), batch_size=500)

    #print predictions.shape
    #print Y_test[0]

    predictions = predictions.argmax(axis=1)
    #print predictions
    #print Y_test.argmax(axis=1)

    pred_errors = np.count_nonzero(predictions - Y_test.argmax(axis=1))

    print("Test accuracy: %.2f" % ( (Y_test.shape[0] - pred_errors) / float(Y_test.shape[0] )))

    dill.dump(net.get_params(), open("net", mode='w'))

    #create a new net and load parameters from the first one... test and check the same accuracy

    b = SigtiaNET((None, 64), 10, hidden_layers=3, hidden_neurons=50, dropout_rate=0.0)
    b.set_params(net.get_params())
    
    predictions = b.predict((X_test, None), batch_size=500)
    predictions = predictions.argmax(axis=1)
    pred_errors = np.count_nonzero(predictions - Y_test.argmax(axis=1))
    print("Test accuracy: %.2f" % ( (Y_test.shape[0] - pred_errors) / float(Y_test.shape[0] )))
    
