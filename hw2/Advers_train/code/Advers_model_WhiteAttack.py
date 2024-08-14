import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from argparse import ArgumentParser
from my_model import my_cnn

parser = ArgumentParser()
parser.add_argument("--data_path", type = str, default = "dataset")

args = parser.parse_args()

fashionmnist_test = torchvision.datasets.FashionMNIST(args.data_path, train = True, download = True, transform = torchvision.transforms.ToTensor())

test_iter = DataLoader(fashionmnist_test, batch_size=64, shuffle=False)

loss_function = torch.nn.CrossEntropyLoss()

model = my_cnn()
model.load_state_dict(torch.load("Advers_train/model_save/my_cnn_adv.pt"))
model.eval()

correct_id_ls = []
sec_prob_labels = []
for i, (x, y) in enumerate(test_iter):
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
alpha = 1e-4

attacked_sample = []
for i in range(select_num):
    img = Variable(fashionmnist_test[selected_id[i]][0].view(1, 1, 28, 28), requires_grad=True)
    incorrect_label = sec_prob_labels[selected_id[i]]

    for s in range(max_step):
        out = model(img)
        if out.argmax(dim=1) == incorrect_label:
            attacked_sample.append((selected_id[i], img[0], incorrect_label))
            break

        loss = loss_function(out, torch.tensor([incorrect_label]))
        # img.grad = 0 if img.grad is not None else None
        loss.backward()

        with torch.no_grad():
            img -= alpha * img.grad
            img.grad = None

print(f"successfully attacked image ratio: {len(attacked_sample) / select_num}")

# 选择10张图片
for i in range(10):
    id, attacked_img, incorrect_label = attacked_sample[i]
    origin_img, correct_label = fashionmnist_test[id]
    print(f"Difference between sample {i}: {torch.mean(origin_img - attacked_img).item()}")
    save_image(origin_img, f"Advers_train/WhiteAttack_samples/{i}raw_label_{correct_label}.png")
    save_image(attacked_img, f"Advers_train/WhiteAttack_samples/{i}attacked_label_{incorrect_label}.png")
