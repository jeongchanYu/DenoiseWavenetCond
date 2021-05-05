import datetime
import time
import os
import tensorflow as tf
import numpy as np


def get_tf_version():
    """
    Get tensorflow version
    """
    tf_version = tf.__version__.split('.')
    return list(map(int, tf_version[:2]))


def tf_gpu_active_alloc(limit_percent=None):
    """
        Prevent GPU overflow, allocate GPU memory incrementally
        If limit_percent is not None, GPU memory will be set to limit_percent percentage.
    """
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    if limit_percent:
        gpu_config.gpu_options.per_process_gpu_memory_fraction = limit_percent
    session = tf.compat.v1.InteractiveSession(config=gpu_config)


def process_time(func):
    """
    Decorator for calculate processing time of function
    """
    def wrapper():
        start = time.time()
        func()
        print("Processing time :", datetime.timedelta(seconds=time.time()-start))
    return wrapper


def createFolder(directory):
    """
    Create folder, if directory is not exist
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('I: Creating directory. ' +  directory)


def load_directory():
    """
    Load directory of current file.
    """
    path = os.path.join(os.path.dirname(__file__))
    if path == "":
        path = "../ConvolutionAE-master"
    return path


def read_path_list(dirname, extention=""):
    """
    Get file path list that exist in directory.
    You can filter files by the extension you enter. ex)"wav"
    """
    try:
        return_list = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                return_list.extend(read_path_list(full_filename, extention))
            else:
                ext = os.path.splitext(full_filename)[-1][1:]
                if extention == "" or ext == extention:
                    return_list.append(full_filename)
        return_list.sort()
        return return_list
    except PermissionError:
        pass


def compare_path_list(dirname1, dirname2, extention=""):
    """
    You can compare file path list whether they are same or not.
    You can filter files by the extension you enter. ex)"wav"
    """
    list1 = read_path_list(dirname1, extention)
    list2 = read_path_list(dirname2, extention)
    for i in range(len(list1)):
        list1[i] = list1[i].replace(dirname1, "")
    for i in range(len(list2)):
        list2[i] = list2[i].replace(dirname2, "")
    list1.sort()
    list2.sort()
    if list1 == list2:
        return True
    else:
        return False


# for WGP Server
def write_plot_file(filename, index, value):
    """
    This function is used for WGP Server.

    You can write plot file. But it will be override.
    You can clear the file by using "clear_plot_file()" function.
    """
    with open(filename, 'a') as f:
        f.write("{{x:{}, y:{}}},".format(index, value))


def clear_plot_file(filename):
    """
    This function is used for WGP Server.

    You can clear the plot file.
    """
    with open(filename, 'w') as f:
        pass


# for WGP Server
def write_csv_file(filename, index, value):
    """
    This function is used for WGP Server.

    You can write plot file. But it will be override.
    You can clear the file by using "clear_plot_file()" function.
    """
    with open(filename, 'a') as f:
        f.write("{},{}\n".format(index, value))


def clear_csv_file(filename):
    """
    This function is used for WGP Server.

    You can clear the plot file.
    """
    with open(filename, 'w') as f:
        pass


def window(window_name, frame_size):
    check = os.path.isfile('{}/window/{}_{}.npy'.format(load_directory(), window_name, frame_size))
    if check:
        sample = np.load('{}/window/{}_{}.npy'.format(load_directory(), window_name, frame_size))
        return sample
    else:
        if window_name == 'hanning':
            sample = tf.signal.hann_window(frame_size)
        elif window_name == 'hamming':
            sample = tf.signal.hamming_window(frame_size)
        elif window_name == 'uniform':
            sample = tf.ones(frame_size)
        else:
            raise Exception('Select hanning or hamming')

        sample = np.array(sample)
        createFolder("{}/window".format(load_directory()))
        np.save('{}/window/{}_{}'.format(load_directory(), window_name, frame_size), sample)
        return sample