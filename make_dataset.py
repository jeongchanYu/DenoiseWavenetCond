import os
import custom_function as cf
import tensorflow as tf
import numpy as np
import wav

def make_dataset(train_source_path, train_target_path, batch_size, previous_size, frame_size, future_size, shift_size=None, max_condition=None, window_type='uniform', shuffle=True, default_float='float32'):
    if not shift_size:
        shift_size = frame_size

    # train_target_path is path or file?
    source_path_isdir = os.path.isdir(train_source_path)
    target_path_isdir = os.path.isdir(train_target_path)
    if target_path_isdir != source_path_isdir:
        raise Exception("E: Target and source path is incorrect")
    if target_path_isdir:
        if not cf.compare_path_list(train_target_path, train_source_path, 'wav'):
            raise Exception("E: Target and source file list is not same")
        train_source_file_list = cf.read_path_list(train_source_path, "wav")
        train_target_file_list = cf.read_path_list(train_target_path, "wav")
    else:
        train_source_file_list = [train_source_path]
        train_target_file_list = [train_target_path]


    # trim dataset
    train_source_cut_list = []
    train_target_cut_list = []
    train_source_condition_list = []
    number_of_total_frame = 0
    sample_rate_check = 0
    window = cf.window(window_type, frame_size+previous_size+future_size)
    for i in range(len(train_source_file_list)):
        # read train data file
        source_signal, source_sample_rate = wav.read_wav(train_source_file_list[i])
        target_signal, target_sample_rate = wav.read_wav(train_target_file_list[i])
        if max_condition:
            source_condition = int(train_source_file_list[i].split('_')[-1].replace(".wav", ""))
            if source_condition > max_condition:
                raise Exception("E : Source condition number is larger than max condition number")

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
                train_source_cut_list.append(np_source_signal.tolist())
                train_target_cut_list.append(np_target_signal.tolist())
                if max_condition:
                    train_source_condition_list.append(tf.one_hot(source_condition, max_condition).numpy().tolist())
            else:
                np_source_signal = source_signal[j*shift_size:j*shift_size+frame_size+previous_size+future_size]
                np_target_signal = target_signal[j*shift_size:j*shift_size+frame_size+previous_size+future_size]
                train_source_cut_list.append(np_source_signal)
                train_target_cut_list.append(np_target_signal)
                if max_condition:
                    train_source_condition_list.append(tf.one_hot(source_condition, max_condition).numpy().tolist())

    if max_condition:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_source_cut_list, train_source_condition_list, train_target_cut_list)).batch(batch_size)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_source_cut_list, train_target_cut_list)).batch(batch_size)

    if shuffle:
            train_dataset.shuffle(number_of_total_frame)

    return train_dataset, number_of_total_frame