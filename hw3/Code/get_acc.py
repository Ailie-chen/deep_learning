import os

class Digit:
    def __init__(self, num, x, y, w, h):
        self.num = num
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def IoU(self, another):
        area_a = self.w * self.h
        area_b = another.w * another.h
        w = min(another.x + another.w, self.x + self.w) - max(self.x, another.x)
        h = min(another.y + another.h, self.y + self.h) - max(self.y, another.y)

        if w <= 0 or h <= 0:
            return 0
        
        area_c = w * h
        return area_c / (area_a + area_b - area_c)


def load_label_file(file_path, conf_thres=0.2):
    if not os.path.exists(file_path):
        return []

    digits = []
    with open(file_path, "r") as f:
        lines = [line.split(' ') for line in f.readlines()]
        for line in lines:
            if line == [] or float(line[-1]) < conf_thres:
                continue
            digits.append(Digit(int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])))
    return digits
        

if __name__ == "__main__":
    total_num, acc_num = 13068, 0

    for i in range(1, total_num + 1):
        gt = load_label_file(f"dataset/labels/test/{i}.txt")
        pred = load_label_file(f"yolov5/svhn_results/exp6/test/exp7/labels/{i}.txt")
        
        if len(gt) == len(pred) and len(pred) != 0:
            all_paired = True
            for gt_j in gt:
                paired = False
                for pred_k in pred:
                    if pred_k.IoU(gt_j) > 0.5 and pred_k.num == gt_j.num:
                        paired = True
                        break
                if not paired:
                   all_paired = False
                   break
            if all_paired:
                acc_num += 1

    print(f"Acc: {acc_num / total_num: .4f}")
