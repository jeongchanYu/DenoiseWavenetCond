import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
import os
import numpy as np
import custom_function as cf

class ConvolutionAutoEncoder(tf.keras.Model):
    def __init__(self, frame_size, default_float='float32'):
        super(ConvolutionAutoEncoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        # parameters
        self.frame_size = frame_size

        # seed
        self.seed = tf.concat([tf.ones([1, 1, 64]), tf.zeros([1, self.frame_size-1, 64])], 1)

        # encoder
        self.conv_encoder_layer = [Conv1D(64, 5, padding='same', activation='tanh')]
        self.conv_encoder_layer.append(Conv1D(64, 3, padding='same', activation='tanh'))
        self.conv_encoder_layer.append(Conv1D(64, 3, strides=2, padding='same', activation='tanh'))

        # filter gen
        self.filter_block = [FilterGenBlock(64) for i in range(4)]

        # convolution block
        self.convolution_block = [ConvolutionBlock(64) for i in range(4)]

        # output block
        self.conv_output = [Conv1D(64, 5, padding='same', activation='tanh')]
        self.conv_output.append(Conv1D(32, 3, padding='same', activation='tanh'))
        self.conv_output.append(Conv1D(1, 1, padding='same', activation='tanh'))

    def call(self, x):
        feature = x
        for f in self.conv_encoder_layer:
            feature = f(feature)

        input = tf.tile(self.seed, [x.shape[0], 1, 1])

        filter = tf.zeros([feature.shape[0], feature.shape[1], 0])
        for i in range(len(self.convolution_block)):
            filter = self.filter_block[i](feature, filter)
            input = self.convolution_block[i](input, filter)

        output = input
        for f in self.conv_output:
            output = f(output)

        return output


    def save_optimizer_state(self, optimizer, save_path, save_name):
        cf.createFolder(save_path)
        np.save(os.path.join(save_path, save_name), optimizer.get_weights())


    def load_optimizer_state(self, optimizer, load_path, load_name):
        opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)
        optimizer.set_weights(opt_weights)


class ConvolutionBlock(tf.Module):
    def __init__(self, channel):
        super(ConvolutionBlock, self).__init__()

        self.conv_out = Conv1D(channel, 5, padding='same', activation='tanh')

    def __call__(self, input, filter):
        conv_out = self.Convolution1D(input, filter)
        conv_out = self.conv_out(conv_out)
        return conv_out

    def Convolution1D(self, input, filter):
        if len(input.shape) != 3 or len(filter.shape) != 3:
            raise Exception("Dimension of input must be 3")
        for i in [0, 2]:
            if input.shape[i] != filter.shape[i]:
                raise Exception("Batch and channel size must be same. Input {} / filter {}".format(input.shape, filter.shape))
        if input.shape[1] < filter.shape[1]:
            raise Exception("Dimension 2 of input must larger than filter. Input {} / filter {}".format(input.shape, filter.shape))

        long_input = input
        short_input = filter

        long_input = tf.transpose(long_input, perm=[0, 2, 1])
        short_input = tf.transpose(short_input, perm=[0, 2, 1])

        long_input = tf.expand_dims(long_input, 3)
        short_input = tf.expand_dims(short_input, 3)

        mat_long = long_input
        for i in range(1, long_input.shape[2]):
            mat_long = tf.concat([mat_long, tf.roll(long_input, shift=i, axis=2)], 3)
        mat_short = tf.pad(short_input, [[0, 0], [0, 0], [0, long_input.shape[2] - short_input.shape[2]], [0, 0]])

        result = tf.linalg.matmul(mat_long, mat_short)

        result = tf.squeeze(result, [3])
        result = tf.transpose(result, perm=[0, 2, 1])

        return result


class FilterGenBlock(tf.Module):
    def __init__(self, channel):
        self.conv_layer = [Conv1D(channel, 5, padding='same', activation='tanh') for i in range(5)]

    def __call__(self, input, filter):
        filter = tf.concat([input, filter], 2)
        for f in self.conv_layer:
            filter = f(filter)

        return filter