代码运行根目录为：2201213070-CW-HW2/

## 实验结果：
一、白盒攻击
目录：2201213070-CW-HW2/WhiteBoxAttack
（1）采用pytorch中的CNN模型，训练后在训练集和测试集上的准确率为92.9% 和 93.62%，具体信息位于 2201213070-CW-HW2/WhiteBoxAttack/model_save/record.txt
（2）在对样本进行白盒攻击时，选取的y~,为输出概率第二大的标签，也即不正确标签中概率最大的那个，方法是助教论文中给的
（3）在随机选取的1000个成功识别的样本上白盒攻击的成功率为：0.273 具体信息位于 2201213070-CW-HW2/WhiteBoxAttack/attacked_samples/report.txt
（4）攻击成功的随机选取的10个样本样本，对应标签，位于 2201213070-CW-HW2/WhiteBoxAttack/attacked_samples 

二、黑盒攻击
目录：2201213070-CW-HW2/BlackBoxAttack
方法：采用白盒攻击成功的对抗样本对黑盒进行攻击
（1）在助教提供的样本上白盒攻击的成功率: 0.053,（orz，可以看出助教给的样本还是比较刁钻的）
（2）白盒攻击生成的对抗样本在黑盒攻击上成功的概率为：1.0 体信息位于 2201213070-CW-HW2/WBlackBoxAttack/attacked_samples/record.txt
（3）黑盒攻击成功的样本，对应标签，位于 2201213070-CW-HW2/BlackBoxAttack/attacked_samples 

三、对抗训练
目录：目录：2201213070-CW-HW2/Advers_train
(1)通过对抗训练后的新模型my_cnn_adv在训练集和测试集上的准确率为93.5%和93.8%，相对于原始的92.9%和93.62%有所提升
(2)白盒攻击：原始的分类器攻击成功率：0.273 对抗训练后的分类器攻击成功率为 0.17。说明对抗训练还是增强了模型的抗攻击能力。
(3)黑盒攻击：原始的分类器攻击成功率：0.053 对抗训练后的分类器攻击成功率为 0.053。可能是因为得到的新分类器学习的对抗样本来自白盒攻击使用原始数据集得到的，没有获得助教给的1000个数据的信息，刚好这1000数据又比较刁钻，所以黑盒攻击的成功率没有下降


## 目录说明：
├── Readme.md
└── dataset                                 //fashionmnist数据集
├── WhiteBoxAttack                          //白盒攻击
│   ├── attacked_samples                    //对抗样本
│   ├── code                                //源码
│   │   ├── attack.py                       //白盒攻击
│   │   ├── model.py                        //my_cnn类
│   │   ├── test.py                         //白盒模型在测试集准确率
│   │   └── train.py                        //训练白盒模型
│   └── model_save                          //白盒模型
│       ├── my_cnn.pt   
│       └── record.txt
├── BlackBoxAttack                          //黑盒攻击
│   ├── attack_data                         //助教提供的样本
│   │   └── correct_1k.pkl
│   ├── attacked_samples                    //对抗样本
│   ├── code                                //源码
│   │   ├── attack.py                       //黑盒攻击的代码
│   │   ├── model.py                        //my_cnn 类，用于生成白盒模型
│   │   └── model_zj.py                     //助教提供的黑盒模型
│   └── model-zj                            //黑盒模型的pt文件
│       └── cnn.ckpt
├── Advers_train                            //对抗训练
│   ├── BlackAttack_samples                 //黑盒攻击在新分类器my_cnn_adv的上攻击成功的10个样本
│   ├── WhiteAttack_samples                 //黑盒攻击在新分类器my_cnn_adv的上攻击成功的10个样本
│   ├── code                                //源码
│   │   ├── Aders_model_BlackAttack.py      //使用黑盒攻击对新分类器my_cnn_adv进行攻击
│   │   ├── Advers_model_WhiteAttack.py     //使用白盒攻击对新分类器my_cnn_adv进行攻击
│   │   ├── advers_test.py                  //新分类器my_cnn_adv在测试集上的准确率
│   │   ├── advers_train.py                 //使用对抗训练得到新分类器my_cnn_adv
│   │   └── my_model.py                     //定义的my_cnn类
│   └── model_save                          //新分类器my_cnn_adv的pt文件，及其在训练、测试集上的准确率
│       ├── my_cnn_adv.pt
│       └── report.txt



## 环境信息
Python 3.10.10 
Package            Version
------------------ ---------
python-mnist       0.7
numpy              1.24.2
torch              2.0.0
torchvision        0.15.1
typing_extensions  4.5.0
urllib3            1.26.15