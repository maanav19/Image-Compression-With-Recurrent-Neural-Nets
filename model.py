from __future__ import print_function
import tensorflow as tf

#Defining the dimensions of the different layers used in the neural net
ENCODE_INPUT_DIMS = 32
ENCODE_LAYER1_DIMS = 64
ENCODE_LAYER2_DIMS = 128
ENCODE_LAYER3_DIMS = 128
DECODE_INPUT_DIMS = 128
DECODE_LAYER1_DIMS = 128
DECODE_LAYER2_DIMS = 128
DECODE_LAYER3_DIMS = 64
DECODE_LAYER4_DIMS = 32

def rnn_conv(name, inputs, hiddens, filters, kernel_size, strides):
    #Defining the different filters used for the gates in an LSTM unit
    gates_filters = 4 * filters
    hidden, cell = hiddens

    #Creating layers for the 2D convolutional RNN
    with tf.compat.v1.variable_scope(name, reuse = tf.compat.v1.AUTO_REUSE):
        conv_inputs = tf.compat.v1.layers.Conv2D(name = 'conv_inputs', inputs = inputs, filters = gates_filters,
                                                 kernel_size = kernel_size, padding = 'same', strides = strides)
        conv_hidden = tf.compat.v1.layers.Conv2D(name = 'conv_hidden', inputs = hidden, filters = gates_filters,
                                                 kernel_size = kernel_size, padding = 'same', strides = strides)

    #Defining the different gate units in an LSTM cell
    in_gate, out_gate, cell_gate, forget_gate = tf.split(conv_inputs + conv_hidden, 4, axis = -1)
    in_gate = tf.nn.sigmoid(in_gate)
    out_gate = tf.nn.sigmoid(out_gate)
    forget_gate = tf.nn.sigmoid(forget_gate)
    cell_gate = tf.nn.tanh(cell_gate)

    new_cell = (tf.math.multiply(out_gate, cell_gate) + tf.math.multiply(in_gate, cell_gate))
    new_hidden = tf.math.multiply(out_gate, tf.nn.tanh(new_cell))

    return new_cell, new_hidden


def initial_hidden(input_size, filters, kernel_size, name):
    #Defining the initial hidden and cell states

    hidden_name = name + '_hidden'
    cell_name = name + '_cell'

    shape = [input_size] + kernel_size + [filters]

    hidden = tf.zeros(shape)
    cell = tf.zeros(shape)

    return hidden, cell


def padding(x, stride):

    if x % stride == 0:
        return x // stride
    else:
        return x // stride + 1



class encoder(object):


    def __init__(self, batch_size, is_training = False, height = 32, width = 32):

        self.batch_size = batch_size
        self.is_training = is_training
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):

        height = padding(padding(self.height, 2), 2)
        width = padding(padding(self.width, 2), 2)

        self.hiddens1 = initial_hidden(self.batch_size, ENCODE_LAYER1_DIMS, [height,width], 'encoder1')
        height = padding(height, 2)
        width = padding(width, 2)

        self.hiddens2 = initial_hidden(self.batch_size, ENCODE_LAYER2_DIMS, [height, width], 'encoder2')
        height = padding(height, 2)
        width = padding(width, 2)

        self.hiddens3 = initial_hidden(self.batch_size, ENCODE_LAYER3_DIMS, [height, width], 'encoder3')



    def encode(self, inputs):

        with tf.compat.v1.variable_scope('encoder', reuse = tf.compat.v1.AUTO_REUSE):
            encode_rnn_input = tf.compat.v1.layers.Conv2D(inputs = inputs, filters = ENCODE_INPUT_DIMS, kernel_size= [3,3],
                                                          strides= (2,2), padding = 'same', name = 'encoder_rnn_input' )
            self.hiddens1 = rnn_conv('encode_rnn_conv1', encode_rnn_input, self.hiddens1, ENCODE_LAYER1_DIMS, [3,3], (2,2))
            self.hiddens2 = rnn_conv('encode_rnn_conv2',self.hiddens1[0], self.hiddens2, ENCODE_LAYER2_DIMS, [3,3], (2,2))
            self.hiddens3 = rnn_conv('encode_rnn_conv3',self.hiddens2[00, self.hiddens3, ENCODE_LAYER3_DIMS, [3,3], (2,2))

        code = binarizer(self.hiddens3[0])
        return code

    def binarizer(self, inputs, filters = 32, kernel_size = (1,1)):

        with tf.compat.v1.variable_scope('binarizer', reuse= tf.compat.v1.AUTO_REUSE):
            binarizer_input = tf.compat.v1.layers.Conv2D(inputs=inputs, filters = filters, kernel_size= kernel_size,
                                                         padding= 'same', name= 'binarizer_inputs', activation= tf.math.tanh)

        if self.is_training:
            probs = (1 + binarizer_input) / 2
                dist = tf.compat.v1.distributions.Bernoulli(probs= probs, dtype= tf.float32)
                noise = 2 * dist.sample(name= 'noise') - 1 - binarizer_input
                output = binarizer_input + tf.stop_gradient(noise)

        else:
            output = tf.math.sign(binarizer_input)

        return output


'''IM RUNNING INTO AN ISSUE WITH THE DECODER CLASS WHERE IT SHOWS THAT THE CLASS ISN'T ACCESIBLE'''
class decoder(object):

    def __init__(self, batch_size, height, width):

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):

        height = padding(self.height, 2)
        width = padding(self.width, 2)

        self.hiddens4 = initial_hidden(self.batch_size, DECODE_LAYER4_DIMS, [height, width], 'decoder4')
        height = padding(height, 2)
        width = padding(width, 2)

        self.hiddens3 = initial_hidden(self.batch_size, DECODE_LAYER3_DIMS, [height, width], 'decoder3')
        height = padding(height, 2)
        width = padding(width, 2)

        self.hiddens2 = initial_hidden(self.batch_size, DECODE_LAYER2_DIMS, [height, width], 'decoder2')
        height = padding(height, 2)
        width = padding(width, 2)

        self.hiddens1 = initial_hidden(self.batch_size, DECODE_LAYER1_DIMS, [height, width], 'decoder1')

    def decode(self, inputs):

        with tf.compat.v1.variable_scope('decoder', reuse= tf.compat.v1.AUTO_REUSE):
            decoder_rnn_input = tf.compat.v1.layers.Conv2D(inputs = inputs, filters= DECODE_INPUT_DIMS, kernel_size= [3,3],
                                                       strides= (1,1), padding= 'same', name= 'decoder_rnn_input')

            self.hiddens1 = rnn_conv('decode_rnn_conv1', decoder_rnn_input, self.hiddens1, DECODE_LAYER1_DIMS, [2,2], (1,1))
            d_rnn_h1 = tf.nn.depth_to_space(self.hiddens1[0], 2)

            self.hiddens2 = rnn_conv('decode_rnn_conv2', d_rnn_h1, self.hiddens2, DECODE_LAYER2_DIMS, [3,3], (1,1))
            d_rnn_h2 = tf.nn.depth_to_space(self.hiddens2[0], 2)

            self.hiddens3 = rnn_conv('decode_rnn_conv3', d_rnn_h2, self.hiddens3, DECODE_LAYER3_DIMS, [3,3], (1,1))
            d_rnn_h3 = tf.nn.depth_to_space(self.hiddens3[0], 2)

            self.hiddens4 = rnn_conv('decode_rnn_conv4', d_rnn_h3, self.hiddens4, DECODE_LAYER4_DIMS, [3,3], (1,1))
            d_rnn_h4 = tf.nn.depth_to_space(self.hiddens4[0], 2)

            output = tf.compat.v1.layers.Conv2D(inputs= d_rnn_h4, filters= 3, kernel_size= [3,3], strides= (1,1),
                                                padding= 'same', name= 'output', activation= tf.nn.tanh)

        return output / 2



class residual_rnn(object):

    def __init__(self, batch_size, num_iters):

        self.batch_size = batch_size
        self.num_iters = num_iters

        self.encoder = encoder(batch_size, is_training= True)
        self.decoder = decoder(batch_size)
        self.inputs = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
        self.build_graph()

    def build_graph(self):

        inputs = self.inputs / 255.0 - 0.5
        self.loss = 0
        self.compress = tf.zeros_like(inputs) + 0.5
        self.encoder.init_hidden()
        self.decoder.init_hidden()

        for x in range(self.num_iters):
            code = self.encoder.encode(inputs)
            outputs = self.decoder.decode(code)

            self.compress += outputs
            self.loss += tf.compat.v1.losses.absolute_difference(inputs, outputs)
            inputs = inputs - outputs

        self.loss /= self.batch_size
        self.compress *= 255

    def get_loss(self):
        return self.loss

    def get_compress(self):
        return self.compress

    def debug(self):
        code = self.encoder.encode(self.inputs)
        output = self.decoder.decode(code)

        print(output.get_shape())

if __name__ = '__main__':
    model = residual_rnn(128, 10)
    model.debug()






