import torch
import numpy as np
from model import my_cnn
from torch.autograd import Variable
from torchvision.utils import save_image
from model_zj import CNN

# 读取提供的1000攻击样本
sample_images, sample_labels = np.load('BlackBoxAttack/attack_data/correct_1k.pkl', allow_pickle = True)


# 使用白盒训练好的模型得到对抗样本
loss_function = torch.nn.CrossEntropyLoss()

model = my_cnn()
model.load_state_dict(torch.load("WhiteBoxAttack/model_save/my_cnn.pt"))
model.eval()

max_step = 100
alpha = 1e-2

attacked_sample = []
for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
    img = torch.tensor(img).view(1, 1, 28, 28)
    img = Variable(img, requires_grad=True)
    incorrect_label = (label + 1) % 10

    for s in range(max_step):
        out = model(img)
        if out.argmax(dim=1) == incorrect_label:
            attacked_sample.append((i, img[0], label, incorrect_label))
            break

        loss = loss_function(out, torch.tensor([incorrect_label]))
        # img.grad = 0 if img.grad is not None else None
        loss.backward()

        with torch.no_grad():
            img -= alpha * img.grad
            img.grad = None

print(f"successfully attacked(white model) image ratio: {len(attacked_sample) / 1000}")

# 加载黑盒模型
black_model = CNN()
black_model.load_state_dict(torch.load("BlackBoxAttack/model-zj/cnn.ckpt"))

# 使用白盒攻击产生的样本攻击黑盒
attacked_black_sample = []
for id, img, label, incorrect_label in attacked_sample:
    out = black_model(img.view(1, 1, 28, 28))
    out_label = out.argmax(dim=1)
    if out_label != label:
        attacked_black_sample.append((id, img, label, out_label))

print(f"successfully attacked(black model) image ratio: {len(attacked_black_sample) / len(attacked_sample)}")

# 保存攻击成功的10张样本
for i in range(10):
    id, attacked_img, label, incorrect_label = attacked_sample[i]
    origin_img = torch.tensor(sample_images[id]).view(1, 28, 28)
    save_image(origin_img, f"BlackBoxAttack/attacked_samples/{i}raw_label_{label}.png")
    save_image(attacked_img, f"BlackBoxAttack/attacked_samples/{i}attacked_label_{incorrect_label}.png")
