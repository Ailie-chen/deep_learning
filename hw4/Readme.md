# HW4: Code Generation

本次作业参考了[CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code)的仓库。以下是文件说明：

- Code: 
  - code：训练代码
  - evaluator：评测代码
  - dataset：数据集
    - test.json：预处理后的测试集
    - test_codexglue.json：CodeXGLUE提供的不含code字段的测试集
    - test_shuffled.json：原始论文提供的测试集，含有code字段
  - save/concode：
    - checkpoint-90000：训练检查点
    - dev.gold：验证集ground truth
    - dev.output：验证集prediction
    - test.output：测试集输出
    - tensorboard：训练曲线
  - preprocess.py：测试集预处理代码
- Papers：参考论文
- Report.pptx：模型训练介绍PPT



由于CodeXGLUE提供的CONCODE测试集不含有code字段，因此我从原始论文提供的链接中下载了测试集（test_shuffled.json），并编写了预处理脚本（preprocess.py）将code字段填充到CodeXGLUE提供的测试集中。