import tensorflow as tf
import json
import os
import custom_function as cf
import wav
import numpy as np
import denoise_wavenet_condition as DWC
import time
import datetime
import math


# tf version check
tf_version = cf.get_tf_version()

# prevent GPU overflow
cf.tf_gpu_active_alloc()

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

frame_size = config["frame_size"]
shift_size = config["shift_size"]
window_type = config["window_type"]
batch_size = config["batch_size"]
default_float = config["default_float"]
load_check_point_name = config["load_check_point_name"]
test_source_path = config["test_source_path"]

# test_source_path is directory or file?
source_path_isdir = os.path.isdir(test_source_path)
if source_path_isdir:
    test_source_file_list = cf.read_path_list(test_source_path, "wav")
else:
    test_source_file_list = [test_source_path]

# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make model
    model = CAE.ConvolutionAutoEncoder(frame_size)
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # load model
    if load_check_point_name != "":
        model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_directory(), load_check_point_name))
        test_loss.reset_states()
    else:
        raise Exception("E: 'load_check_pint_name' in 'config.json' is empty.")

# test function
@tf.function
def test_step(dist_inputs):
    output = []
    def step_fn(inputs):
        index, input = inputs
        x = tf.reshape(input, [-1, frame_size, 1])

        y_pred = model(x)
        mse = loss_object(x, y_pred)
        if len(mse.shape) == 0:
            mse = tf.reshape(mse, [1])
        loss = tf.reduce_sum(mse) * (1.0/batch_size)

        if y_pred.shape[0] != 0:
            result = tf.split(y_pred, num_or_size_splits=y_pred.shape[0], axis=0)
            for i in range(len(result)):
                output.append([index[i], tf.squeeze(result[i])])

        return mse

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    test_loss(mean_loss/batch_size)

    return output

# trim dataset
sample_rate_check = 0
window = cf.window(window_type, frame_size)
for i in range(len(test_source_file_list)):
    test_source_cut_list = []
    test_source_cut_index = []
    number_of_total_frame = 0

    # read test data file
    source_signal, source_sample_rate = wav.read_wav(test_source_file_list[i])

    # different sample rate detect
    if sample_rate_check == 0:
        sample_rate_check = source_sample_rate
    elif sample_rate_check != source_sample_rate:
        raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, sample_rate_check))

    # padding
    size_of_source = source_signal.size
    padding_size = (shift_size - (size_of_source % shift_size)) % shift_size
    padding_size += frame_size - shift_size
    source_signal = np.pad(source_signal, (shift_size, padding_size)).astype(default_float)
    number_of_frame = (source_signal.size - (frame_size - shift_size))//(shift_size)
    number_of_total_frame += number_of_frame

    # cut by frame
    for j in range(number_of_frame):
        np_source_signal = np.array(source_signal[j*shift_size:j*shift_size+frame_size])
        if window_type != "uniform":
            np_source_signal *= window
        test_source_cut_list.append(np_source_signal.tolist())
        test_source_cut_index.append(j)


    with strategy.scope():
        # make dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((test_source_cut_index, test_source_cut_list)).batch(batch_size)
        dist_dataset = strategy.experimental_distribute_dataset(dataset=test_dataset)

        # test run
        output_dict = {}
        output_list = [0]*source_signal.size
        j = 0
        start = time.time()
        for inputs in dist_dataset:
            print("\rTest : {}, frame {}/{}".format(test_source_file_list[i].replace(test_source_path, ''), j+1, math.ceil(number_of_total_frame / batch_size)), end='')
            output_package = test_step(inputs)
            for pack in output_package:
                output_dict.setdefault(pack[0].numpy(), pack[1].numpy().tolist())
            j += 1
        output_dict = sorted(output_dict.items())
        for index, value in output_dict:
            for k in range(len(value)):
                output_list[shift_size*index+k] += value[k]

        result_path = "{}/test_result/{}/result/{}".format(cf.load_directory(), load_check_point_name, os.path.dirname(test_source_file_list[i].replace(test_source_path, "")))
        file_name = os.path.basename(test_source_file_list[i])
        cf.createFolder(result_path)
        wav.write_wav(output_list[shift_size:-padding_size], "{}/{}".format(result_path, file_name), sample_rate_check)

        print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

        test_loss.reset_states()

