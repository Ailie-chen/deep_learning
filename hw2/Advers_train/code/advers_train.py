import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.autograd import Variable
from argparse import ArgumentParser
from my_model import my_cnn
####################### 首先训练得到对抗样本 ################
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

loss_function = torch.nn.CrossEntropyLoss()

model = my_cnn()
model.load_state_dict(torch.load("WhiteBoxAttack/model_save/my_cnn.pt"))
model.eval()

correct_id_ls = []
sec_prob_labels = []
for i, (x, y) in enumerate(train_iter):
    model.eval()
    out = model(x)
    torch.nn.functional.softmax(out, dim=-1)
    correct_id = torch.where((out.argmax(dim=1) == y))[0]
    correct_id += i * 64
    correct_id_ls += correct_id.tolist()

    # 取每个图片第二大概率的标签作为目标错误标签
    out[range(out.size(0)), out.argmax(dim=1).tolist()] *= -1
    sec_prob_labels += out.argmax(dim=1).tolist()

select_num = 1000
selected_id = np.random.choice(correct_id_ls, size=select_num, replace=False)
# selected_imgs = fashionmnist_test[selected_id]      # 样本矩阵：1000 * 28 * 28
max_step = 100       # 每个样本最多迭代次数
alpha = 1e-2

attacked_sample = []
adv_imgs= []
adv_labels = []
for i in range(select_num):
    img = Variable(fashionmnist_train[selected_id[i]][0].view(1, 1, 28, 28), requires_grad=True)
    correct_label = fashionmnist_train[selected_id[i]][1]
    incorrect_label = sec_prob_labels[selected_id[i]]

    for s in range(max_step):
        out = model(img)
        if out.argmax(dim=1) == incorrect_label:
            attacked_sample.append((selected_id[i], img[0], incorrect_label,correct_label))
            img.requires_grad_(False)
            adv_imgs.append(torch.clone(img[0]))
            adv_labels.append(correct_label)
            break

        loss = loss_function(out, torch.tensor([incorrect_label]))
        # img.grad = 0 if img.grad is not None else None
        loss.backward()

        with torch.no_grad():
            img -= alpha * img.grad
            img.grad = None

print(f"successfully attacked image ratio: {len(attacked_sample) / select_num}")

################ 将对抗样本掺入原始样本重新进行训练 ############

# self defined dataset class

class AdvDataset(TensorDataset):
    def __getitem__(self, index):
        return self.tensors[0][index], self.tensors[1][index].item()

# 将对抗样本掺入原始数据集
adv_dataset = AdvDataset(torch.stack([torch.tensor(x) for x in adv_imgs],dim = 0), torch.tensor(adv_labels, dtype=torch.long))
fashionmnist_train_mixed = ConcatDataset([fashionmnist_train, adv_dataset])

# 划分一个epoch内的数据集
train_iter_adver = torch.utils.data.DataLoader(fashionmnist_train_mixed, args.batch_size, shuffle = True)

# 采用cnn进行训练
network = my_cnn()

# 交叉熵损失函数和一个优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), args.lr)

""" # 将attack样本分成20一份,并且设置为正确的标签
batched_attacked_sample = []
img_ls = []
label_ls = []
for id, img, incorrect_label in attacked_sample:
    img_ls.append(img)
    label_ls.append(correct_label)
    if len(img_ls) == 20:
        batched_attacked_sample.append((torch.stack(img_ls, dim=0), torch.tensor(label_ls, dtype=torch.long).view(-1, 1)))
        img_ls.clear()
        label_ls.clear()

if len(img_ls) != 0:
    batched_attacked_sample.append((torch.stack(img_ls, dim=0), torch.tensor(label_ls, dtype=torch.long).view(-1, 1)))""" 

# 训练
for epoch in range(1, args.epochs + 1):
    total_loss = 0.0
    total_acc = 0.0
    for x, y in train_iter_adver :
        network.train()
        out = network(x)
        loss = loss_function(out, y)
        total_loss = total_loss + loss
        total_acc = total_acc + (((out.argmax(dim=1) == y).float().sum())/args.batch_size)
        optimizer.zero_grad() #清空上一次参数
        loss.backward()
        optimizer.step() #更新参数

    # 输出训练中的loss
    print(f"Epoch:{epoch}, loss:{total_loss/len(train_iter_adver)}, acc:{total_acc/len(train_iter_adver)}")
"""     for x, y in batched_attacked_sample:
        network.train()
        out = network(x)
        loss = loss_function(out, y)
        total_loss = total_loss + loss
        total_acc = total_acc + (((out.argmax(dim=1) == y).float().sum())/1)
        optimizer.zero_grad() #清空上一次参数
        loss.backward()
        optimizer.step() #更新参数 """



# 保存模型
torch.save(network.state_dict(),'Advers_train/model_save/my_cnn_adv.pt')
