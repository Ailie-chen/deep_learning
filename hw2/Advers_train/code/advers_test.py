import torch
import torchvision
from torch.utils.data import DataLoader
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


total_loss, total_acc = 0.0, 0.0
for i, (x, y) in enumerate(test_iter):
    model.eval()
    out = model(x)
    loss = loss_function(out, y)
    total_loss = total_loss + loss
    total_acc = total_acc + (out.argmax(dim=1) == y).float().sum()

total_loss /= len(test_iter)
total_acc /= len(fashionmnist_test)

print(f"test loss: {total_loss}, test acc: {total_acc}")
