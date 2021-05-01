import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
import os
import numpy as np
import custom_function as cf

class DenoiseWavenetCondition(tf.keras.Model):
    def __init__(self, dilation, condition=None, input_size=None, default_float='float32'):
        super(DenoiseWavenetCondition, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        # parameters
        self.dilation = dilation
        self.condition = condition
        self.input_size = input_size

        # channel increase
        self.conv_channel = Conv1D(128, 1, padding='same')

        # residual block
        self.residual_block = [ResidualBlock(d, condition=self.condition, input_size=self.input_size) for d in self.dilation[:-1]]
        self.residual_block.append(ResidualBlock(self.dilation[-1], residual=None, condition=self.condition, input_size=self.input_size))

        # skip output
        self.conv_output = [Conv1D(2048, 3, padding='same', activation='relu')]
        self.conv_output.append(Conv1D(256, 3, padding='same', activation='relu'))
        self.conv_output.append(Conv1D(1, 1, activation='tanh'))


    def call(self, x, condition=None):
        input = self.conv_channel(x)

        output, input = self.residual_block[0](input, condition)
        for f in self.residual_block[1:]:
            skip_output, input = f(input, condition)
            output += skip_output

        for f in self.conv_output:
            output = f(output)

        return output


    def save_optimizer_state(self, optimizer, save_path, save_name):
        cf.createFolder(save_path)
        np.save(os.path.join(save_path, save_name), optimizer.get_weights())


    def load_optimizer_state(self, optimizer, load_path, load_name):
        opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)
        optimizer.set_weights(opt_weights)


class ResidualBlock(tf.Module):
    def __init__(self, dilation, residual=True, condition=None, input_size=None):
        self.input_size = input_size
        self. residual = residual
        self. condition = condition

        self.conv_gated_tanh = Conv1D(128, 3, padding='same', dilation_rate=dilation)
        self.conv_gated_sigmoid = Conv1D(128, 3, padding='same', dilation_rate=dilation)
        self.conv_skip = Conv1D(129, 1)
        if self.residual:
            self.conv_residual = Conv1D(128, 1)
        if self.condition:
            if not self.input_size:
                raise Exception("E: In condition mode, you must put the input_size")
            self.dense_condition_tanh = [Dense(self.input_size) for i in range(128)]
            self.dense_condition_sigmoid = [Dense(self.input_size) for i in range(128)]

    def __call__(self, input, condition=None):
        if self.condition:
            trans_condition_tanh = self.dense_condition_tanh[0](condition)
            trans_condition_sigmoid = self.dense_condition_sigmoid[0](condition)
            trans_condition_tanh = tf.reshape(trans_condition_tanh, [-1, self.input_size, 1])
            trans_condition_sigmoid = tf.reshape(trans_condition_sigmoid, [-1, self.input_size, 1])
            for i in range(1, 128):
                trans_condition_tanh = tf.concat([trans_condition_tanh, tf.reshape(self.dense_condition_tanh[i](condition), [-1, self.input_size, 1])], 2)
                trans_condition_sigmoid = tf.concat([trans_condition_sigmoid, tf.reshape(self.dense_condition_sigmoid[i](condition), [-1, self.input_size, 1])], 2)
            trans_condition_tanh += input
            trans_condition_sigmoid += input
            gated_tanh = self.conv_gated_tanh(trans_condition_tanh)
            gated_sigmoid = self.conv_gated_sigmoid(trans_condition_sigmoid)
        else:
            gated_tanh = self.conv_gated_tanh(input)
            gated_sigmoid = self.conv_gated_sigmoid(input)
        gated_tanh = tf.keras.activations.tanh(gated_tanh)
        gated_sigmoid = tf.keras.activations.sigmoid(gated_sigmoid)
        gated_result = gated_tanh * gated_sigmoid
        skip_output = self.conv_skip(gated_result)
        if self.residual:
            residual_output = self.conv_residual(gated_result)
            residual_output += input
        else:
            residual_output = None

        return skip_output, residual_output