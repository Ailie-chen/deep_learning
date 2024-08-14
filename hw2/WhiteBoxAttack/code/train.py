import torch
import torchvision 
from argparse import ArgumentParser
from model import my_cnn


# 定义参数，学习自github-助教的写法
parser = ArgumentParser()
parser.add_argument("--data_path", type = str, default = "dataset")
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--epochs", type = int, default = 20)
args = parser.parse_args()

# 使用torchvision加载数据集fashionmnist
fashionmnist_train = torchvision.datasets.FashionMNIST(args.data_path, train = True, download = True, transform = torchvision.transforms.ToTensor())

# 划分一个epoch内的数据集
train_iter = torch.utils.data.DataLoader(fashionmnist_train, args.batch_size, shuffle = True)

# 采用cnn进行训练
network = my_cnn()

# 交叉熵损失函数和一个优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), args.lr)

# 训练
for epoch in range(1, args.epochs + 1):
    total_loss = 0.0
    total_acc = 0.0
    for x, y in train_iter:
        network.train()
        out = network(x)
        loss = loss_function(out, y)
        total_loss = total_loss + loss
        total_acc = total_acc + (((out.argmax(dim=1) == y).float().sum())/args.batch_size)
        optimizer.zero_grad() #清空上一次参数
        loss.backward()
        optimizer.step() #更新参数

    # 输出训练中的loss
    print(f"Epoch:{epoch}, loss:{total_loss/len(train_iter)}, acc:{total_acc/len(train_iter)}")
    


# 保存模型
torch.save(network.state_dict(),'WhiteBoxAttack/model_save/my_cnn.pt')






