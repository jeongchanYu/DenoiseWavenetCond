import tensorflow as tf
import json
import os
import custom_function as cf
import wav
import numpy as np
import convolution_autoencoder as CAE
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
dilation = config["dilation"]

batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
default_float = config["default_float"]

train_source_path = config["train_source_path"]
train_target_path = config["train_target_path"]

load_check_point_name = config["load_check_point_name"]
save_check_point_name = config["save_check_point_name"]
save_check_point_period = config["save_check_point_period"]
plot_file = config["plot_file"]


# training_target_path is path or file?
target_path_isdir = os.path.isdir(train_target_path)
source_path_isdir = os.path.isdir(train_source_path)
if target_path_isdir != source_path_isdir:
    raise Exception("E: Target and source path is incorrect")
if target_path_isdir:
    if not cf.compare_path_list(train_target_path, train_source_path, 'wav'):
        raise Exception("E: Target and source file list is not same")
    train_target_file_list = cf.read_path_list(train_target_path, "wav")
    train_source_file_list = cf.read_path_list(train_source_path, "wav")
else:
    train_target_file_list = [train_target_path]
    train_source_file_list = [train_source_path]


# trim dataset
train_source_cut_list = []
number_of_total_frame = 0
sample_rate_check = 0
window = cf.window(window_type, frame_size)
for i in range(len(train_source_file_list)):
    # read train data file
    source_signal, source_sample_rate = wav.read_wav(train_source_file_list[i])

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
        train_source_cut_list.append(np_source_signal.tolist())


# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_source_cut_list).shuffle(number_of_total_frame).batch(batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(dataset=train_dataset)

    # make model
    model = CAE.ConvolutionAutoEncoder(frame_size)
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    train_loss = tf.keras.metrics.Mean(name='train_loss')


# train function
@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        x = tf.reshape(inputs, [-1, frame_size, 1])
        with tf.GradientTape() as tape:
            y_pred = model(x)
            mse = loss_object(x, y_pred)
            if len(mse.shape) == 0:
                mse = tf.reshape(mse, [1])
            loss = tf.reduce_sum(mse) * (1.0/batch_size)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mse
    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    train_loss(mean_loss/batch_size)


# train run
with strategy.scope():
    # load model
    if load_check_point_name != "":
        saved_epoch = int(load_check_point_name.split('_')[-1])
        for inputs in dist_dataset:
            train_step(inputs)
            break
        model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_directory(), load_check_point_name))
        model.load_optimizer_state(optimizer, '{}/checkpoint/{}'.format(cf.load_directory(), load_check_point_name), 'optimizer')
        train_loss.reset_states()
    else:
        cf.clear_plot_file('{}/{}'.format(cf.load_directory(), config['plot_file']))
        saved_epoch = 0

    for epoch in range(saved_epoch, saved_epoch+epochs):
        i = 0
        start = time.time()
        for inputs in dist_dataset:
            print("\rTrain : epoch {}/{}, training {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(number_of_total_frame / batch_size)), end='')
            train_step(inputs)
            i += 1
        print(" | loss : {}".format(train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

        if ((epoch + 1) % config['save_check_point_period'] == 0) or (epoch + 1 == 1):
            cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_directory(), save_check_point_name, epoch+1))
            model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_directory(), save_check_point_name, epoch+1))
            model.save_optimizer_state(optimizer, '{}/checkpoint/{}_{}'.format(cf.load_directory(), save_check_point_name, epoch + 1), 'optimizer')

        # write plot file
        cf.write_plot_file('{}/{}'.format(cf.load_directory(), config['plot_file']), epoch+1, train_loss.result())

        train_loss.reset_states()

