import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import Dense
import os
import numpy as np
import custom_function as cf

class DenoiseWavenetCondition(tf.keras.Model):
    def __init__(self, dilation, input_size, max_condition, default_float='float32'):
        super(DenoiseWavenetCondition, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        # parameters
        self.dilation = dilation
        self.input_size = input_size
        self.max_condition = max_condition

        # channel increase
        self.conv_channel = Conv1D(128, 1, padding='same')

        # condition dimension increase
        self.conv_cond_channel_up = Conv1DTranspose(1, self.input_size-self.max_condition+1)

        # residual block
        self.residual_block = [ResidualBlock(d) for d in self.dilation]

        # skip output
        self.conv_output = [Conv1D(2048, 3, padding='same', activation='relu')]
        self.conv_output.append(Conv1D(256, 3, padding='same', activation='relu'))
        self.conv_output.append(Conv1D(1, 1, activation='tanh'))


    def call(self, x, condition):
        input = self.conv_channel(x)

        condition = self.conv_cond_channel_up(condition)
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
    def __init__(self, dilation):
        self.conv_gated_tanh = Conv1D(128, 3, padding='same', dilation_rate=dilation)
        self.conv_gated_sigmoid = Conv1D(128, 3, padding='same', dilation_rate=dilation)
        self.conv_skip = Conv1D(128, 1)
        self.conv_residual = Conv1D(128, 1)

        self.conv_trans_condition_tanh = Conv1D(128, 1, padding='same')
        self.conv_trans_condition_sigmoid = Conv1D(128, 1, padding='same')

    def __call__(self, input, condition):
        trans_condition_tanh = self.conv_trans_condition_tanh(condition)
        trans_condition_sigmoid =  self.conv_trans_condition_sigmoid(condition)

        gated_tanh = tf.keras.activations.tanh(self.conv_gated_tanh(input) + trans_condition_tanh)
        gated_sigmoid = tf.keras.activations.sigmoid(self.conv_gated_sigmoid(input) + trans_condition_sigmoid)
        gated_result = gated_tanh * gated_sigmoid

        skip_output = self.conv_skip(gated_result)
        residual_output = self.conv_residual(gated_result) + input

        return skip_output, residual_output