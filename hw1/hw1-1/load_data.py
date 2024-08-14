import urllib.request
import os
import gzip
import numpy as np
import pickle

# 下载 MNIST 数据集到 data 目录


# 读取 MNIST 数据集文件并加载到 numpy 数组中
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16).astype(np.float32)
    return data.reshape(-1, 784) #输入数据按照行向量拍平

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return data

def load_data():
    data_dir = 'dataset' #根目录是HW1
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 
            't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    # 下载文件
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f'Downloading {base_url}{file}...')
            urllib.request.urlretrieve(f'{base_url}{file}', file_path)

    # 读入内存
    train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    # 归一化
    train_images /= 255
    test_images /= 255

    return train_images, train_labels, test_images, test_labels

