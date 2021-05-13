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
import make_dataset as md


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

train_source_path = config["train_source_path"]
train_target_path = config["train_target_path"]
test_source_path = config["test_source_path"]
test_target_path = config["test_target_path"]

load_check_point_name = config["load_check_point_name"]
save_check_point_name = config["save_check_point_name"]
save_check_point_period = config["save_check_point_period"]
early_stopping = config["early_stopping"]
plot_file = config["plot_file"]


train_dataset, number_of_total_frame_train = md.make_dataset(train_source_path, train_target_path, batch_size, previous_size, frame_size, future_size, shift_size, None, window_type, True)
if early_stopping:
    test_dataset, number_of_total_frame_test = md.make_dataset(test_source_path, test_target_path, batch_size, previous_size, frame_size, future_size, shift_size, None, window_type, True)


# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make dataset
    dist_dataset_train = strategy.experimental_distribute_dataset(dataset=train_dataset)
    if early_stopping:
        dist_dataset_test = strategy.experimental_distribute_dataset(dataset=test_dataset)

    # make model
    model = DW.DenoiseWavenet(dilation)
    loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    if early_stopping:
        test_loss = tf.keras.metrics.Mean(name='test_loss')


# train function
@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        x = tf.reshape(x, [-1, previous_size + frame_size + future_size, 1])
        y = tf.reshape(y, [-1, previous_size + frame_size + future_size, 1])

        with tf.GradientTape() as tape:
            y_pred = model(x)
            mae = loss_object(tf.slice(y, [0, previous_size, 0], [-1, frame_size, -1]), tf.slice(y_pred, [0, previous_size, 0], [-1, frame_size, -1])) * 2
            if len(mae.shape) == 0:
                mae = tf.reshape(mae, [1])
            loss = tf.reduce_sum(mae) * (1.0 / batch_size)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mae
    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    train_loss(mean_loss/batch_size)

# test function
@tf.function
def test_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        x = tf.reshape(x, [-1, previous_size + frame_size + future_size, 1])
        y = tf.reshape(y, [-1, previous_size + frame_size + future_size, 1])
        y_pred = model(x)
        mae = loss_object(tf.slice(y, [0, previous_size, 0], [-1, frame_size, -1]), tf.slice(y_pred, [0, previous_size, 0], [-1, frame_size, -1])) * 2
        if len(mae.shape) == 0:
            mae = tf.reshape(mae, [1])
        loss = tf.reduce_sum(mae) * (1.0 / batch_size)
        return mae
    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    test_loss(mean_loss/batch_size)


# train run
with strategy.scope():
    # load model
    if load_check_point_name != "":
        saved_epoch = int(load_check_point_name.split('_')[-1])
        for inputs in dist_dataset_train:
            train_step(inputs)
            break
        model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_directory(), load_check_point_name))
        model.load_optimizer_state(optimizer, '{}/checkpoint/{}'.format(cf.load_directory(), load_check_point_name), 'optimizer')
        train_loss.reset_states()
        if early_stopping:
            test_loss.reset_sates()
    else:
        cf.clear_plot_file('{}/{}'.format(cf.load_directory(), config['plot_file']))
        cf.clear_csv_file('{}/{}'.format(cf.load_directory(), config['plot_file']).replace(".plot", ".csv"))
        if early_stopping:
            cf.clear_plot_file('{}/{}'.format(cf.load_directory(), config['plot_file'].replace(".plot", "_test.plot")))
            cf.clear_csv_file('{}/{}'.format(cf.load_directory(), config['plot_file']).replace(".plot", "._test.csv"))
        saved_epoch = 0

    for epoch in range(saved_epoch, saved_epoch+epochs):
        i = 0
        start = time.time()
        for inputs in dist_dataset_train:
            print("\rTrain : epoch {}/{}, training {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(number_of_total_frame_train / batch_size)), end='')
            train_step(inputs)
            i += 1
        print(" | loss : {}".format(train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

        if ((epoch + 1) % config['save_check_point_period'] == 0) or (epoch + 1 == 1):
            cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_directory(), save_check_point_name, epoch+1))
            model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_directory(), save_check_point_name, epoch+1))
            model.save_optimizer_state(optimizer, '{}/checkpoint/{}_{}'.format(cf.load_directory(), save_check_point_name, epoch + 1), 'optimizer')

        if early_stopping:
            i = 0
            start = time.time()
            for inputs in dist_dataset_test:
                print("\rTest : epoch {}/{}, training {}/{}".format(epoch + 1, saved_epoch + epochs, i + 1, math.ceil(number_of_total_frame_test / batch_size)), end='')
                test_step(inputs)
                i += 1
            print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

        # write plot file
        cf.write_plot_file('{}/{}'.format(cf.load_directory(), config['plot_file']), epoch+1, train_loss.result())
        cf.write_csv_file('{}/{}'.format(cf.load_directory(), config['plot_file'].replace(".plot", ".csv")), epoch+1, train_loss.result())
        if early_stopping:
            cf.write_plot_file('{}/{}'.format(cf.load_directory(), config['plot_file'].replace(".plot", "_test.plot")), epoch + 1, test_loss.result())
            cf.write_csv_file('{}/{}'.format(cf.load_directory(), config['plot_file'].replace(".plot", "_test.csv")), epoch + 1, test_loss.result())

        train_loss.reset_states()
        if early_stopping:
            test_loss.reset_sates()
