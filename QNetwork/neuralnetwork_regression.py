import numpy as np
from QNetwork import optimizers
import sys  # for sys.float_info.epsilon
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch  # for Arc
import matplotlib.collections as pltcoll
import math

###############################################
# Thanks to Dr. Chuck Anderson for his neural #
# network with optimizers implementation      #
# https://www.cs.colostate.edu/~anderson/wp/  #
###############################################

class NeuralNetwork():


    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, activation_function='tanh'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation_function = activation_function

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))

        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.total_epochs = 0
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

    def setup_standardization(self, Xmeans, Xstds, Tmeans, Tstds):
        self.Xmeans = np.array(Xmeans)
        self.Xstds = np.array(Xstds)
        self.Tmeans = np.array(Tmeans)
        self.Tstds = np.array(Tstds)

    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flatenned matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(-1, 1, size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views


    # Return string that shows how the constructor was called
    def __repr__(self):
        return f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_per_layer}, {self.n_outputs}, \'{self.activation_function}\')'


    # Return string that is more informative to the user about the state of this neural network.
    def __str__(self):
        result = self.__repr__()
        if len(self.error_trace) > 0:
            return self.__repr__() + f' trained for {len(self.error_trace)} epochs, final training error {self.error_trace[-1]:.4f}'


    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):
        '''
train: 
  X: n_samples x n_inputs matrix of input samples, one per row
  T: n_samples x n_outputs matrix of target output values, one sample per row
  n_epochs: number of passes to take through all samples updating weights each pass
  learning_rate: factor controlling the step size of each update
  method: is either 'sgd' or 'adam'
        '''

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = optimizers.Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units, 
        # but only if the network has a single output. Multiplying by self.Tstds for 
        # multiple outputs does not correctly unstandardize the error.
        if len(self.Tstds) == 1:
            error_convert_f = lambda err: (np.sqrt(err) * self.Tstds)[0] # to scalar
        else:
            error_convert_f = lambda err: np.sqrt(err)[0] # to scalar
            

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=verbose,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=verbose,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        # Return neural network object to allow applying other methods after training.
        #  Example:    Y = nnet.train(X, T, 100, 0.01).use(X)
        return self

    def relu(self, s):
        s[s < 0] = 0
        return s

    def grad_relu(self, s):
        return (s > 0).astype(int)
    
    def forward_pass(self, X):
        '''X assumed already standardized. Output returned as standardized.'''
        self.Ys = [X]
        for W in self.Ws[:-1]:
            if self.activation_function == 'relu':
                self.Ys.append(self.relu(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
            else:
                self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        '''Assumes forward_pass just called with layer outputs in self.Ys.'''
        error = T - self.Ys[-1]
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.Xmeans) / self.Xstds
        Ys = self.forward_pass(X)
        Y = Ys[-1]
        # Unstandardize output Y before returning it
        return Y * self.Tstds + self.Tmeans

    def draw(self, input_names=None, output_names=None, scale='by layer', gray=False):        
        plt.title('{} weights'.format(sum([Wi.size for Wi in self.Ws])))

        def isOdd(x):
            return x % 2 != 0

        n_layers = len(self.Ws)

        Wmax_overall = np.max(np.abs(np.hstack([w.reshape(-1) for w in self.Ws])))

        # calculate xlim and ylim for whole network plot
        #  Assume 4 characters fit between each wire
        #  -0.5 is to leave 0.5 spacing before first wire
        xlim = max(map(len, input_names)) / 4.0 if input_names else 1
        ylim = 0

        for li in range(n_layers):
            ni, no = self.Ws[li].shape  #no means number outputs this layer
            if not isOdd(li):
                ylim += ni + 0.5
            else:
                xlim += ni + 0.5

        ni, no = self.Ws[n_layers-1].shape  #no means number outputs this layer
        if isOdd(n_layers):
            xlim += no + 0.5
        else:
            ylim += no + 0.5

        # Add space for output names
        if output_names:
            if isOdd(n_layers):
                ylim += 0.25
            else:
                xlim += round(max(map(len, output_names)) / 4.0)

        ax = plt.gca()

        # changes from Jim Jazwiecki (jim.jazwiecki@gmail.com) CS480 student
        character_width_factor = 0.07
        padding = 2
        if input_names:
            x0 = max([1, max(map(len, input_names)) * (character_width_factor * 3.5)])
        else:
            x0 = 1
        y0 = 0 # to allow for constant input to first layer
        # First Layer
        if input_names:
            y = 0.55
            for n in input_names:
                y += 1
                ax.text(x0 - (character_width_factor * padding), y, n, horizontalalignment="right", fontsize=20)

        patches = []
        for li in range(n_layers):
            thisW = self.Ws[li]
            if scale == 'by layer':
                maxW = np.max(np.abs(thisW))
            else:
                maxW = Wmax_overall
            ni, no = thisW.shape
            if not isOdd(li):
                # Even layer index. Vertical layer. Origin is upper left.
                # Constant input
                ax.text(x0 - 0.2, y0 + 0.5, '1', fontsize=20)
                for i in range(ni):
                    ax.plot((x0, x0 + no - 0.5), (y0 + i + 0.5, y0 + i + 0.5), color='gray')
                # output lines
                for i in range(no):
                    ax.plot((x0 + 1 + i - 0.5, x0 + 1 + i - 0.5), (y0, y0 + ni + 1), color='gray')
                # cell "bodies"
                xs = x0 + np.arange(no) + 0.5
                ys = np.array([y0 + ni + 0.5] * no)
                for x, y in zip(xs, ys):
                    patches.append(pltpatch.RegularPolygon((x, y - 0.4), 3, 0.3, 0, color ='#555555'))
                # weights
                if gray:
                    colors = np.array(['black', 'gray'])[(thisW.flat >= 0) + 0]
                else:
                    colors = np.array(['red', 'green'])[(thisW.flat >= 0) + 0]
                xs = np.arange(no) + x0 + 0.5
                ys = np.arange(ni) + y0 + 0.5
                coords = np.meshgrid(xs, ys)
                for x, y, w, c in zip(coords[0].flat, coords[1].flat, 
                                      np.abs(thisW / maxW).flat, colors):
                    patches.append(pltpatch.Rectangle((x - w / 2, y - w / 2), w, w, color=c))
                y0 += ni + 1
                x0 += -1 ## shift for next layer's constant input
            else:
                # Odd layer index. Horizontal layer. Origin is upper left.
                # Constant input
                ax.text(x0 + 0.5, y0 - 0.2, '1', fontsize=20)
                # input lines
                for i in range(ni):
                    ax.plot((x0 + i + 0.5,  x0 + i + 0.5), (y0, y0 + no - 0.5), color='gray')
                # output lines
                for i in range(no):
                    ax.plot((x0, x0 + ni + 1), (y0 + i+ 0.5, y0 + i + 0.5), color='gray')
                # cell 'bodies'
                xs = np.array([x0 + ni + 0.5] * no)
                ys = y0 + 0.5 + np.arange(no)
                for x, y in zip(xs, ys):
                    patches.append(pltpatch.RegularPolygon((x - 0.4, y), 3, 0.3, -math.pi / 2, color ='#555555'))
                # weights
                if gray:
                    colors = np.array(['black', 'gray'])[(thisW.flat >= 0) + 0]
                else:
                    colors = np.array(['red', 'green'])[(thisW.flat >= 0) + 0]
                xs = np.arange(ni) + x0 + 0.5
                ys = np.arange(no) + y0 + 0.5
                coords = np.meshgrid(xs, ys)
                for x, y, w, c in zip(coords[0].flat, coords[1].flat,
                                   np.abs(thisW / maxW).flat, colors):
                    patches.append(pltpatch.Rectangle((x - w / 2, y - w / 2), w, w, color=c))
                x0 += ni + 1
                y0 -= 1 ##shift to allow for next layer's constant input

        collection = pltcoll.PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

        # Last layer output labels
        if output_names:
            if isOdd(n_layers):
                x = x0 + 1.5
                for n in output_names:
                    x += 1
                    ax.text(x, y0 + 0.5, n, fontsize=20)
            else:
                y = y0 + 0.6
                for n in output_names:
                    y += 1
                    ax.text(x0 + 0.2, y, n, fontsize=20)
        ax.axis([0, xlim, ylim, 0])
        ax.axis('off')
