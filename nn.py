from numpy import random, array, exp, dot, ndarray
import pickle
import math
import numpy as np
import sys


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def to_categorical(y, num_classes=None, dtype="longdouble"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'longdouble'`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def gelu(x):
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * np.power(x, 3))))


def gelu_derivative(x):
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))

    return (
        0.5 * (1.0 + np.tanh(0.797885 * x + 0.035677 * np.power(x, 3)))
        + x
        * (np.exp(-0.5 * np.power(x, 2)) / np.sqrt(2 * np.pi))
        * (0.035677 * np.power(x, 2) + 0.797885)
        * cdf
    )

def bounded_gelu(x):
    c = 1.702
    y = np.minimum(np.maximum(x, -c), c)
    result = 0.5 * x * (1 + np.tanh(c * y))
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print("Input:", x)
        print("Bounded input:", y)
        print("Output:", result)
        sys.exit()
    return result


def bounded_gelu_derivative(x):
    c = 1.702
    fx = bounded_gelu(x)
    alpha = 0.5 * c * (1.0 + np.tanh(c * np.minimum(np.maximum(x, -c), c))) + \
            0.5 * (fx + np.abs(fx)) * (1 - np.tanh(c * np.minimum(np.maximum(x, -c), c)) ** 2)
    return alpha

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # subtract maximum value from each element
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    if np.isnan(softmax_x).any():
        print("Input:", x)
        print("Exponential:", exp_x)
        print("Denominator:", np.sum(exp_x, axis=-1, keepdims=True))
        sys.exit()
    return softmax_x

print(softmax(np.array([[1, 2, 3, 4, 5]])))
def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


def sigmoid(x):
    # applying the sigmoid function
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    # computing derivative to the Sigmoid function
    return x * (1 - x)


def relu(x):
    # applying the ReLu function
    return x * (x > 0)


def relu_derivative(x):
    # computing derivative to the ReLu function
    return 1.0 * (x > 0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def prelu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)


def prelu_derivative(x, alpha=0.2):
    return np.where(x > 0, 1, alpha)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


def linear(x):
    # the output of this function is just the input x
    return x


def linear_derivative(x):
    # the derivative of a linear function is always 1
    return 1


def maxout(x, axis=-1):
    # we split the input tensor along the given axis and apply the max function elementwise
    return np.max(x, axis=axis)


def maxout_derivative(x, axis=-1):
    # the derivative of this function is a binary mask where the max element is 1, and all others are 0
    max_element = np.max(x, axis=axis, keepdims=True)
    return np.where(x == max_element, 1, 0)


def swish(x):
    # applying the swish function
    return x * sigmoid(x)


def swish_derivative(x):
    # computing the derivative to the swish function
    return sigmoid(x) + x * sigmoid_derivative(x)


activation_functions = {
    "sigmoid": [sigmoid, sigmoid_derivative],
    "relu": [relu, relu_derivative],
    "gelu": [gelu, gelu_derivative],
    "softmax": [softmax, softmax_derivative],
    "tanh": [tanh, tanh_derivative],
    "leaky_relu": [leaky_relu, leaky_relu_derivative],
    "prelu": [prelu, prelu_derivative],
    "elu": [elu, elu_derivative],
    "linear": [linear, linear_derivative],
    "maxout": [maxout, maxout_derivative],
    "swish": [swish, swish_derivative],
    "bounded_gelu": [bounded_gelu, bounded_gelu_derivative]
}


def xavier_init(n, m):
    """
    Xavier initialization for a neural network with n input neurons and m output neurons.
    """
    epsilon = np.sqrt(2 / (n + m))
    weights = np.random.randn(n, m) * epsilon
    return weights


class NeuralNetwork:
    def __init__(
        self, input_length=4, architecture=[], synaptic_weights=None, save_funct=None
    ):
        """
        architecture syntax:
        [
            (layer_output_size, layer_inputs, activation_function) #input size is auto calculated. layer_inputs is a list of all the inputs given. In layer inputs, 0 represents the initial model inputs, 1 is the first layer's output, 2 is the second layer's output, etc.
            (32, [1], "sigmoid") # example
        ]
        """
        self.input_length = input_length
        self.save_funct = save_funct
        self.layers = len(architecture) + 1
        self.architecture = architecture
        if synaptic_weights == None:
            # seeding for random number generation
            random.seed(1)

            # creating synaptic weights for each layer
            self.synaptic_weights = []
            if len(architecture) == 0:
                raise ValueError("Architecture must have a length of at least 1.")
            layer_sizes = [input_length] + [layer[0] for layer in architecture]
            i = 0
            for layer_size, inputs, activation in architecture:
                self.synaptic_weights.append(
                    xavier_init(sum([layer_sizes[inp] for inp in inputs]), layer_size)
                )
                i += 1
        else:
            self.synaptic_weights = array(synaptic_weights, dtype=np.longdouble)

    def __del__(self):
        if self.save_funct:
            self.save_funct(self)

    def train(
        self,
        training_inputs,
        training_outputs,
        training_iterations=1,
        batch_size=0,
        learning_rate=0.5,
        log_every=False,
        save_every=False,
        test_on_log=False,
        lambda_val=0.01, # for the L2
        max_adjustment_norm=np.inf
    ):
        """
        Batch Size of 0 means fit all the data into one batch.
        """
        #TODO: add bias term
        if batch_size == 0:
            batch_size = len(training_inputs)
        if len(training_inputs) != len(training_outputs):
            raise ValueError(
                "Training inputs and training outputs must be the same length."
            )
        training_inputs = array(training_inputs, dtype=np.longdouble)
        training_outputs = array(training_outputs, dtype=np.longdouble)
        # training the model to make accurate predictions while adjusting weights continually
        num_batches = math.ceil(len(training_inputs) / batch_size)
        print(f"Starting training, log every {log_every}, save every {save_every}.")
        for iteration in range(training_iterations):
            # Shuffle the input data before each epoch
            shuffle_indices = np.random.permutation(len(training_inputs))
            training_inputs = training_inputs[shuffle_indices]
            training_outputs = training_outputs[shuffle_indices]

            for batch in range(num_batches):
                inputs = training_inputs[batch * batch_size : (batch + 1) * batch_size]
                outputs = training_outputs[
                    batch * batch_size : (batch + 1) * batch_size
                ]
                # siphon the training data via the neuron
                layer_outputs = [inputs]
                for i in range(len(self.synaptic_weights)):
                    layer_inputs = []
                    for j in self.architecture[i][1]:
                        layer_inputs.append(layer_outputs[j])
                    layer_inputs = np.concatenate(layer_inputs, axis=-1)
                    layer_outputs.append(
                        activation_functions[self.architecture[i][2]][0](
                            dot(layer_inputs, self.synaptic_weights[i])
                        )
                    )
                output = layer_outputs[-1]

                # computing error rate for back-propagation
                error = outputs - output

                # performing weight adjustments
                layer_errors = [
                    error
                    * activation_functions[self.architecture[-1][2]][1](
                        layer_outputs[-1]
                    )
                ]
                for i in reversed(range(len(self.synaptic_weights) - 1)):
                    layer_errors.insert(
                        0,
                        dot(layer_errors[0], self.synaptic_weights[i + 1].T)
                        * activation_functions[self.architecture[i][2]][1](
                            layer_outputs[i + 1]
                        ),
                    )
                for i in range(len(self.synaptic_weights)):
                    layer_inputs = []
                    for j in self.architecture[i][1]:
                        layer_inputs.append(layer_outputs[j])
                    layer_inputs = np.concatenate(layer_inputs, axis=-1)
                    adjustments = dot(layer_inputs.T, layer_errors[i])
                    if max_adjustment_norm != np.inf:
                        adjustments_norm = np.linalg.norm(adjustments)
                        if adjustments_norm > max_adjustment_norm:
                            adjustments *= max_adjustment_norm / adjustments_norm
                    # add L2 regularization term
                    l2_reg = learning_rate * lambda_val * self.synaptic_weights[i]
                    self.synaptic_weights[i] += learning_rate * (adjustments - l2_reg)
                if log_every and ((iteration + 1) * (batch + 1)) % log_every == 0:
                    print(
                        f"Iter: {iteration+1}/{training_iterations}; Batch: {batch+1}/{num_batches}; {100 * (iteration+1)/training_iterations}%"
                    )
                    if test_on_log:
                        test_on_log(self)
                if (
                    self.save_funct
                    and save_every
                    and ((iteration + 1) * (batch + 1)) % log_every == 0
                ):
                    self.save_funct(self)

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        layer_outputs = [inputs]
        for i in range(len(self.synaptic_weights)):
            layer_inputs = []
            for j in self.architecture[i][1]:
                layer_inputs.append(layer_outputs[j])
            layer_inputs = np.concatenate(layer_inputs, axis=-1)
            layer_outputs.append(
                activation_functions[self.architecture[i][2]][0](
                    dot(layer_inputs, self.synaptic_weights[i])
                )
            )
        return layer_outputs[-1]


if __name__ == "__main__":
    x = NeuralNetwork(32, 32, 8, 2)
    x.train(
        [process_value("hi"), process_value("hey")],
        [process_value("hey"), process_value("hi")],
        20,
    )
    print(decode(x.think(process_value("hi"))))
