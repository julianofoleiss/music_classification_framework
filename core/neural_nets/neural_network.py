from core.data.data_batch_gen import TensorBatches, DataBatchGenerator
from core.data.pytable import create_earray
import lasagne
import time
import sys
import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        self.train_fn = None
        self.val_fn = None
        self.test_fn = None
        self.feature_fn = []

    def __getstate__(self):
        """
        When pickling a neural network, the functions are not preserved.
        This is important because if the model is passed onto another computer
        it may not be compatible with the compiled code.
    
        Subclasses MUST implement _unset_for_saving, which should unset all vars that are not
        supposed to be pickled. In case there's nothing to unset, a stub must be created.

        Parameters are saved into a temporary attribute '__parameters', which are restored
        upon unpickling. This attribute is deleted upon unpickling.
        """
        __parameters = self.get_params()
        self.__dict__.update({'__parameters' : __parameters})

        self.train_fn = None
        self.test_fn = None
        self.val_fn = None
        self.feature_fn = []
        self._unset_for_saving()

        return self.__dict__

    def __setstate__(self, state):
        """
        See the documentation for __getstate__ above. 
        """
        model_parameters = state['__parameters']
        del state['__parameters']
        self.__dict__.update(state)
        self.set_params(model_parameters)
    
    def _unset_for_saving(self):
       raise NotImplementedError

    def compile(self):
        """
        This function is called to compile the lasagne computation graphs into callable python functions.

        For built-in fit and predict:
            self.train_fn with a function that returns training loss.
            self.val_fn should be a function that returns (predictions,loss)
            self.test_fn should be a function that returns predictions.

        For general feature extraction (get_features):
            self.feature_fn[] with a list of functions that return predictions. Each index points
            to the corresponding hidden layer.
        """
        raise NotImplementedError
    
    def get_params(self):
        """
        This function collects the parameters from the model and returns as a list.
        
        The default implementation simply collects all trainable parameters.
        """
        return lasagne.layers.get_all_param_values(self.net, trainable=True)

    def set_params(self, params, compile_fns=True):
        """
        Sets the model parameters (params) to an instanced neural network.

        The default implementation simply sets all trainable parameters.

        If compile_fns is True, the network is compiled if not already.
        """
        if self.train_fn is None:
            if compile_fns:
                self.compile()

        lasagne.layers.set_all_param_values(self.net, params)

    def fit(self, train_data, validation_data, shuffle_batches=False, max_epochs=200, early_stopping=None, batch_size=100, debug=False):
        """
        This method provides optional early-stopping functionality for fitting neural networks.

        If early_stopping==None, no early stopping is done and max_epochs are run.
        If early_stopping==x, where x is an integer, up to x epochs will be run after the last optimum validation score.

        After training is stopped, the model restores the best parameters found (which best optimzed performance on the validation set).
        """
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
                raise TypeError, "validation_data should be tuple (inputs,targets) or instance derived of DataBatchGenerator"
        
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

            for train_X, train_Y in train_data.next(batch_size=batch_size, shuffle=shuffle_batches):
                loss = self.train_fn(train_X, train_Y)
                print loss
                train_loss += loss
                train_batches += 1

            val_batches = 0
            val_loss = 0

            print('checking validation...')

            for val_X, val_Y in validation_data.next(batch_size=batch_size, shuffle=shuffle_batches):
                _, loss = self.val_fn(val_X, val_Y)
                print loss
                val_loss += loss
                val_batches += 1

            train_loss /= train_batches
            val_loss /= val_batches

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = k + 1
                n_val_loss = 0
                best_params = self.get_params()
            else:
                n_val_loss += 1

            if early_stopping is not None:
                if n_val_loss >= early_stopping:
                    print("STOPPED EARLY! Best validation loss: %.2f in Epoch %d. Last Epoch Loss: %.2f" % (best_val_loss, best_epoch, val_loss))
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

    def _predict(self, test_data, pred_fn, batch_size=100, h5file=None):
        
        if type(test_data) is tuple:
            test_data = TensorBatches(test_data[0], test_data[1])
        else:
            if not issubclass(test_data.__class__, DataBatchGenerator):
                raise TypeError, "test_data should be tuple(inputs,targets) or instance derived of DataBatchGenerator"

        predictions = None

        train_loss = 0
        train_batches = 0
  
        for test_X, _ in test_data.next(batch_size=batch_size, shuffle=False):
            preds = pred_fn(test_X)

            if predictions is None:
                if h5file is not None:
                    _, h5, predictions = create_earray(h5file, fixdim_size=preds[0].shape[-1])
                else:
                    predictions = preds[0]

            if h5file is not None:
                predictions.append(preds[0])
            else:
                predictions = np.append(predictions, preds[0], axis=0)

        if h5file is not None:
            h5.flush()

        return (predictions, h5) if h5file is not None else (predictions, None)

    def predict(self, test_data, batch_size=100, h5file=None):
        """
        This method returns the outputs for all samples in test_data.
        
        The output here is the activations from the last layer.
        """
        
        if self.test_fn is None:
            raise Exception, "Model not fit yet!"

        return self._predict(test_data, self.test_fn, batch_size, h5file=h5file)

    def get_features(self, input_data, hidden_layer, batch_size, h5file=None):
        """
        This method returns the activations from the hidden_layer for all samples in input_data.

        hidden_layer is a hidden_layer index to feature_fn.
        """
        #print hidden_layer, len(self.feature_fn)
        if hidden_layer > len(self.feature_fn) -1 :
            raise Exception, "Feature extraction function for layer %d is not compiled!" % hidden_layer

        return self._predict(input_data, self.feature_fn[hidden_layer], batch_size, h5file=h5file)    

    def get_loss(self, data, batch_size=100):
        loss_sum = 0
        batches = 0

        for data_X, data_Y in data.next(batch_size=batch_size, shuffle=False):
            _, loss = self.val_fn(data_X, data_Y)
            loss_sum += loss
            batches +=1
            print(loss)

        return loss_sum / batches
        
    @staticmethod
    def to_predicted_features(filelist, data_batches, estimator_output, label_dict_filename=None):
        from core.data.filelists import parse_filelist
        from core.data.predicted_features import PredictedFeatures

        X, Y = parse_filelist(filelist)

        if label_dict_filename is not None:
            ld_text = open(label_dict_filename).read()
            #this reads the dictionary into label_dict
            exec(ld_text)
        else:
            label_dict = None

        pf = PredictedFeatures(X, data_batches.track_nframes, estimator_output, label_dict, [None] * len(data_batches.track_nframes), Y)

        return pf
    

