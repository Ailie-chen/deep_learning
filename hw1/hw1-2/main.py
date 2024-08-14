import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data
from model import NeuralNetwork


# 定义交叉熵损失函数
class SoftmaxWithCrossEntropy:
    def get_loss(self, x, y):
        batch_size = x.shape[0]
        shiftx = x - np.max(x, axis=1).reshape(batch_size, -1)
        exps = np.exp(shiftx)
        sum_exps = np.sum(exps, axis=1).reshape(batch_size, -1)
        probs = exps / sum_exps
        log_probs = -np.log(probs[np.arange(batch_size), y])
        loss = np.sum(log_probs) / batch_size
        self.cache = (probs, y, batch_size)
        return loss, probs

    def get_grad(self):
        probs, y, batch_size = self.cache
        dx = probs
        dx[np.arange(batch_size), y] -= 1
        dx /= batch_size
        return dx

def train(network, train_data, train_labels, test_data, test_labels, 
          batch_size=64, num_epochs=10, learning_rate=0.01):
    num_batches = train_data.shape[0] // batch_size
    train_loss_ls = []
    train_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    criterion = SoftmaxWithCrossEntropy()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        acc_count = 0
        total_count = 0
        if num_batches != 1:        # SGD
            shuffle_idx = np.arange(train_data.shape[0])
            np.random.shuffle(shuffle_idx)
            shuffle_train_data = train_data[shuffle_idx]
            shuffle_train_labels = train_labels[shuffle_idx]
        
        for batch in range(num_batches):
            # 获得batch
            if num_batches != 1:    # SGD
                batch_data = shuffle_train_data[batch*batch_size:(batch+1)*batch_size]
                batch_labels = shuffle_train_labels[batch*batch_size:(batch+1)*batch_size]
            else:                   # BGD
                batch_data = train_data
                batch_labels = train_labels

            # 前向传播得到loss
            logits = network.forward(batch_data)
            loss, log_probs = criterion.get_loss(logits, batch_labels)
            
            epoch_loss += loss
            acc_count += np.sum(np.argmax(log_probs, axis=1) == batch_labels)
            total_count += batch_data.shape[0]

            # 反向传播计算梯度
            dout = criterion.get_grad()
            network.backward(dout)

            # 更新每一层的参数
            for layer in network.layers:
                layer.W -= learning_rate * layer.W_grad
                layer.b -= learning_rate * layer.b_grad

        test_loss, test_acc = test_evaluate(network, test_data, test_labels)

        train_loss = epoch_loss / num_batches
        train_acc = acc_count / total_count
        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        
        test_loss_ls.append(test_loss)
        test_acc_ls.append(test_acc)

        print("Epoch %d Train Loss %.5f Train Acc %.5f Eval Loss %.5f Eval Acc %.5f" % \
              (epoch, train_loss, train_acc, test_loss, test_acc))

    return train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls

def test_evaluate(network, test_data, test_labels, batch_size=30):
    num_batches = test_data.shape[0] // batch_size
    criterion = SoftmaxWithCrossEntropy()
    total_loss = 0
    acc_count = 0
    total_count = 0
    for batch in range(num_batches):
        # 获得batch
        batch_data = test_data[batch*batch_size:(batch+1)*batch_size]
        batch_labels = test_labels[batch*batch_size:(batch+1)*batch_size]

        # 前向传播得到loss
        logits = network.forward(batch_data)
        loss, log_probs = criterion.get_loss(logits, batch_labels)

        total_loss += loss
        acc_count += np.sum(np.argmax(log_probs, axis=1) == batch_labels)
        total_count += batch_data.shape[0]

    return total_loss / num_batches, acc_count / total_count


def plot_learning_curve(loss_train, acc_train, loss_val, acc_val, save_path):
    """
    Plot the learning curve of the model.

    Args:
        loss_train (list): Training loss of each epoch.
        acc_train (list): Training accuracy of each epoch.
        loss_val (list, optional): Validation loss of each epoch. Defaults to None.
        acc_val (list, optional): Validation accuracy of each epoch. Defaults to None.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # plot training and validation loss curve
    axs[0].plot(range(1, len(loss_train)+1), loss_train, label='Training Loss')
    axs[0].plot(range(1, len(loss_val)+1), loss_val, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()

    # plot training and validation accuracy curve
    axs[1].plot(range(1, len(acc_train)+1), acc_train, label='Training Accuracy')
    axs[1].plot(range(1, len(acc_val)+1), acc_val, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].legend()

    # plt.show()
    fig.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_data()
    # BGD训练
    model1 = NeuralNetwork(input_size=784, output_size=10, layer_sizes=[256, 256])
    loss_train1, acc_train1, loss_val1, acc_val1 = train(model1, train_data, train_labels, 
                                                         test_data, test_labels, num_epochs=100, 
                                                         learning_rate=2e-1, batch_size=60000)

    plot_learning_curve(loss_train1, acc_train1, loss_val1, acc_val1, "hw1-2/bgd_curve.png")

    # SGD训练
    model2 = NeuralNetwork(input_size=784, output_size=10, layer_sizes=[256, 256])
    loss_train2, acc_train2, loss_val2, acc_val2 = train(model2, train_data, train_labels, 
                                                         test_data, test_labels, num_epochs=20, 
                                                         learning_rate=1e-1)

    plot_learning_curve(loss_train2, acc_train2, loss_val2, acc_val2, "hw1-2/sgd_curve.png")
