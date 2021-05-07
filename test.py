import tensorflow as tf
import json
import os
import custom_function as cf
import wav
import numpy as np
import denoise_wavenet as DW
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

previous_size = config['previous_size']
frame_size = config["frame_size"]
future_size = config["future_size"]
shift_size = config["shift_size"]
window_type = config["window_type"]
dilation = config["dilation"]

batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
default_float = config["default_float"]

test_source_path = config["test_source_path"]
test_target_path = config["test_target_path"]

load_check_point_name = config["load_check_point_name"]
save_check_point_name = config["save_check_point_name"]
save_check_point_period = config["save_check_point_period"]
plot_file = config["plot_file"]

# test_target_path is path or file?
if test_target_path != "":
    test_target_path = test_source_path
source_path_isdir = os.path.isdir(test_source_path)
target_path_isdir = os.path.isdir(test_target_path)
if target_path_isdir != source_path_isdir:
    raise Exception("E: Target and source path is incorrect")
if target_path_isdir:
    if not cf.compare_path_list(test_target_path, test_source_path, 'wav'):
        raise Exception("E: Target and source file list is not same")
    test_source_file_list = cf.read_path_list(test_source_path, "wav")
    test_target_file_list = cf.read_path_list(test_target_path, "wav")
else:
    test_source_file_list = [test_source_path]
    test_target_file_list = [test_target_path]


# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make model
    model = DW.DenoiseWavenet(dilation)
    loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
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
    output_result = []
    output_noise = []
    def step_fn(inputs):
        index, x, y = inputs
        x = tf.reshape(x, [-1, previous_size + frame_size + future_size, 1])
        y = tf.reshape(y, [-1, previous_size + frame_size + future_size, 1])

        with tf.GradientTape() as tape:
            y_pred = model(x)
            y_pred = tf.slice(y_pred, [0, previous_size, 0], [-1, frame_size, -1])
            n_pred = tf.slice(x, [0, previous_size, 0], [-1, frame_size, -1]) - y_pred
            mae = loss_object(tf.slice(y, [0, previous_size, 0], [-1, frame_size, -1]),
                              y_pred) * 2
            if len(mae.shape) == 0:
                mae = tf.reshape(mae, [1])

            if y_pred.shape[0] != 0:
                result = tf.split(y_pred, num_or_size_splits=y_pred.shape[0], axis=0)
                noise = tf.split(n_pred, num_or_size_splits=n_pred.shape[0], axis=0)
                for i in range(len(result)):
                    output_result.append([index[i], tf.squeeze(result[i])])
                    output_noise.append([index[i], tf.squeeze(noise[i])])
        return mae

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    test_loss(mean_loss/batch_size)

    return output_result, output_noise

# trim dataset
sample_rate_check = 0
window = cf.window(window_type, frame_size)
for i in range(len(test_source_file_list)):
    test_source_cut_index = []
    test_source_cut_list = []
    test_target_cut_list = []
    number_of_total_frame = 0

    # read test data file
    source_signal, source_sample_rate = wav.read_wav(test_source_file_list[i])
    target_signal, target_sample_rate = wav.read_wav(test_target_file_list[i])

    # different sample rate detect
    if source_sample_rate != target_sample_rate:
        raise Exception("E: Different sample rate detected. source({})/target({})".format(source_sample_rate, target_sample_rate))
    if sample_rate_check == 0:
        sample_rate_check = source_sample_rate
    elif sample_rate_check != source_sample_rate:
        raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, sample_rate_check))
    elif sample_rate_check != target_sample_rate:
        raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, target_sample_rate))

    # padding
    size_of_source = source_signal.size
    padding_size = (shift_size - (size_of_source % shift_size)) % shift_size
    padding_size += frame_size - shift_size
    source_signal = np.pad(source_signal, (shift_size+previous_size, padding_size+future_size)).astype(default_float)
    target_signal = np.pad(target_signal, (shift_size+previous_size, padding_size+future_size)).astype(default_float)
    number_of_frame = (source_signal.size - (frame_size - shift_size) - shift_size - future_size)//(shift_size)
    number_of_total_frame += number_of_frame

    # cut by frame
    for j in range(number_of_frame):
        if window_type != "uniform":
            np_source_signal = np.array(source_signal[j * shift_size:j * shift_size + frame_size + previous_size + future_size])
            np_target_signal = np.array(target_signal[j * shift_size:j * shift_size + frame_size + previous_size + future_size])
            np_source_signal *= window
            test_source_cut_index.append(j)
            test_source_cut_list.append(np_source_signal.tolist())
            test_target_cut_list.append(np_target_signal.tolist())
        else:
            np_source_signal = source_signal[j*shift_size:j*shift_size+frame_size+previous_size+future_size]
            np_target_signal = target_signal[j*shift_size:j*shift_size+frame_size+previous_size+future_size]
            test_source_cut_index.append(j)
            test_source_cut_list.append(np_source_signal)
            test_target_cut_list.append(np_target_signal)


    with strategy.scope():
        # make dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((test_source_cut_index, test_source_cut_list, test_target_cut_list)).batch(batch_size)
        dist_dataset = strategy.experimental_distribute_dataset(dataset=test_dataset)

        # test run
        output_dict_result = {}
        output_dict_noise = {}
        output_list_result = [0]*source_signal.size
        output_list_noise = [0]*source_signal.size
        j = 0
        start = time.time()
        for inputs in dist_dataset:
            print("\rTest : {}, frame {}/{}".format(test_source_file_list[i].replace(test_source_path, ''), j+1, math.ceil(number_of_total_frame / batch_size)), end='')
            output_package_result, output_package_noise = test_step(inputs)
            for k in range(len(output_package_result)):
                output_dict_result.setdefault(output_package_result[k][0].numpy(), output_package_result[k][1].numpy().tolist())
                output_dict_noise.setdefault(output_package_noise[k][0].numpy(), output_package_noise[k][1].numpy().tolist())
            j += 1
        output_dict_result = sorted(output_dict_result.items())
        output_dict_noise = sorted(output_dict_noise.items())
        for k in range(len(output_dict_result)):
            result_value = output_dict_result[k][1]
            noise_value = output_dict_noise[k][1]
            for l in range(len(result_value)):
                output_list_result[shift_size*k+l] += result_value[l]
                output_list_noise[shift_size*k+l] += noise_value[l]

        result_path = "{}/test_result/{}/result/{}".format(cf.load_directory(), load_check_point_name, os.path.dirname(test_source_file_list[i].replace(test_source_path, "")))
        noise_path = "{}/test_result/{}/noise/{}".format(cf.load_directory(), load_check_point_name, os.path.dirname(test_source_file_list[i].replace(test_source_path, "")))

        file_name = os.path.basename(test_source_file_list[i])
        cf.createFolder(result_path)
        cf.createFolder(noise_path)
        wav.write_wav(output_list_result[shift_size+previous_size:-(padding_size+future_size)], "{}/{}".format(result_path, file_name), sample_rate_check)
        wav.write_wav(output_list_noise[shift_size+previous_size:-(padding_size+future_size)], "{}/{}".format(noise_path, file_name), sample_rate_check)

        print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

        test_loss.reset_states()

