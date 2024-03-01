# 图像分类

## 安装

```shell
pip install -r requirements.txt
pip install -r requirements-onnx.txt
```

## 开始训练分类器

将以下内容保存到 `train.py` 文件中。

```python
import math
from ditk import logging
from torchvision import transforms
from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop, prob_greyscale
from classification.train import train_simple

logging.try_init_root(logging.INFO)

# 任务元信息
LABELS = ['monochrome', 'normal']  # 每个类别的标签
WEIGHTS = [math.e ** 2, 1.0]  # 每个类别的权重
assert len(LABELS) == len(WEIGHTS), \
    f'标签和权重长度应该相同，但发现{len(LABELS)}(标签)和{len(WEIGHTS)}(权重)。'

# 数据集目录（使用你自己的，如下所示）
# <dataset_dir>
# ├── class1
# │   ├── image1.jpg
# │   └── image2.png  # 所有PIL可读取的格式都可以
# ├── class2
# │   ├── image3.jpg
# │   └── image4.jpg
# └── class3
#     ├── image5.jpg
#     └── image6.jpg
DATASET_DIR = '/my/dataset/directory'

# 训练数据集的数据增强和预处理
TRANSFORM_TRAIN = transforms.Compose([
    # 数据增强
    # 如果颜色不重要，使用此行
    # prob_greyscale(0.5),  
    transforms.Resize((500, 500)),
    RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),

    # 预处理（建议与测试相同）
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 测试数据集的预处理
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
# 数据集可视化（当类太多时可能会很慢）
# 如果你不需要这个，只需注释掉
TRANSFORM_VISUAL = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# 准备数据集
# 在大型数据集上训练时禁用缓存
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST, TRANSFORM_VISUAL)
# 如果你不需要可视化，只需使用这个
# test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)

# 开始训练！
if __name__ == '__main__':
    train_simple(
        # 训练任务的工作目录
        # 中断后将会自动恢复
        workdir='runs/demo_exp',

        # timm中所有模型都是可用的，
        # 使用timm.list_models()查看支持的模型，或者在
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        # 查看性能表格
        # 推荐：
        # 1. 训练时使用 caformer_s36.sail_in22k_ft_in1k_384
        # 2. 精馏时使用 mobilenetv3_large_100
        model_name='caformer_s36.sail_in22k_ft_in1k_384',

        # 标签和权重，未给出权重时全部为1
        labels=LABELS,
        loss_weight=WEIGHTS,

        # 数据集
        train_dataset=train_dataset,
        test_dataset=test_dataset,

        # 训练设置，将使用预训练模型
        max_epochs=100,
        num_workers=8,
        eval_epoch=1,
        key_metric='accuracy',
        loss='focal',  # 当数据集不保证干净时使用 `sce`
        seed=0,
        drop_path_rate=0.4,  # 训练caformer时使用这个

        # 超参数
        batch_size=16,
        learning_rate=1e-5,  # caformer微调推荐使用1e-5
        weight_decay=1e-3,
    )
```

接下来运行

```
accelerate launch train.py
```

## 蒸馏模型

将以下内容保存到 `dist.py` 文件中

```python
from ditk import logging
from torchvision import transforms
from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop, prob_greyscale
from classification.train import train_distillation
logging.try_init_root(logging.INFO)
# 任务元信息
LABELS = ['monochrome', 'normal']  # 每个类别的标签
WEIGHTS = [1.0, 1.0]  # 每个类别的权重
assert len(LABELS) == len(WEIGHTS), \
    f'标签和权重长度应该相同，但发现{len(LABELS)}(标签)和{len(WEIGHTS)}(权重)。'
# 数据集目录（使用你自己的，如下所示）
# <dataset_dir>
# ├── class1
# │   ├── image1.jpg
# │   └── image2.png  # 所有PIL可读取的格式都可以
# ├── class2
# │   ├── image3.jpg
# │   └── image4.jpg
# └── class3
#     ├── image5.jpg
#     └── image6.jpg
DATASET_DIR = '/my/dataset/directory'
# 训练数据集的数据增强和预处理
TRANSFORM_TRAIN = transforms.Compose([
    # 数据增强
    # 如果颜色不重要，使用此行
    # prob_greyscale(0.5),  
    transforms.Resize((500, 500)),
    RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),
    # 预处理（建议与测试相同）
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# 测试数据集的预处理
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
# 数据集可视化（当类太多时可能会很慢）
# 如果你不需要这个，只需注释掉
TRANSFORM_VISUAL = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])
# 准备数据集
# 在大型数据集上训练时禁用缓存
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST, TRANSFORM_VISUAL)
# 如果你不需要可视化，只需使用这个
# test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)
# 开始训练！
if __name__ == '__main__':
    train_distillation(
        # 蒸馏任务的学生模型的工作目录
        # 中断后将会自动恢复
        workdir='runs/demo_exp_dist',
        # timm中所有模型都是可用的，
        # 使用timm.list_models()查看支持的模型，或者在
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        # 查看性能表格
        # 推荐：
        # 1. 训练时使用 caformer_s36.sail_in22k_ft_in1k_384
        # 2. 精馏时使用 mobilenetv3_large_100
        model_name='mobilenetv3_large_100',
        # 从runs/demo_exp中的教师模型进行蒸馏
        teacher_workdir='runs/demo_exp',
        # 标签和权重，未给出权重时全部为1
        labels=LABELS,
        loss_weight=WEIGHTS,
        # 数据集
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # 训练设置，将使用预训练模型
        max_epochs=500,
        num_workers=8,
        eval_epoch=5,
        key_metric='accuracy',
        loss='focal',  # 当数据集不保证干净时使用 `sce`
        seed=0,
        # drop_path_rate=0.4,  # 训练caformer时使用这个
        # 蒸馏设置
        temperature=7.0,
        alpha=0.3,
        # 超参数
        batch_size=16,
        learning_rate=1e-4,  # caformer微调推荐使用
        weight_decay=1e-3,
    )

```

接下来运行

```shell
accelerate launch dist.py
```

## 导出 ONNX 模型

### 从检查点导出

请注意，此导出仅支持使用 `classficiation` 模块转储的检查点，因为它包含除网络状态字典之外的额外信息。

```shell
python -m classification.onnx export --help
```

```text
Usage: python -m classification.onnx export [OPTIONS]

  将检查点导出为onnx

Options:
  -i, --input FILE       要导出的输入检查点。 [必填]
  -s, --imgsize INTEGER  输入图像大小。 [默认值: 384]
  -D, --non-dynamic      不要导出具有动态输入高度和宽度的模型。
  -V, --verbose          显示详细信息。
  -o, --output FILE      ONNX 模型的输出文件
  -h, --help             显示此帮助信息并退出。
```

## 从训练工作目录导出

```shell
python -m classification.onnx dump -w runs/demo_exp
```

然后，`runs/demo_exp/ckpts/best.ckpt` 将转储为 `runs/demo_exp/onnxs/best.onnx`

以下是帮助信息：

```text
Usage: python -m classification.onnx dump [OPTIONS]

  从现有的工作目录中转储 ONNX 模型。

Options:
  -w, --workdir DIRECTORY  训练的工作目录。 [必填]
  -s, --imgsize INTEGER    输入图像大小。 [默认值: 384]
  -D, --non-dynamic        不要导出具有动态输入高度和宽度的模型。
  -V, --verbose            显示详细信息。
  -h, --help               显示此帮助信息并退出。
```

## 发布训练好的模型

在开始之前，请将 HF_TOKEN 变量设置为您的 `HF_TOKEN` 令牌

```shell
# 在 Linux 上执行
export HF_TOKEN=xxxxxxxx
```

将训练好的模型（包括 ckpt、onnx、度量数据和图表）发布到 Hugging Face 仓库 `your/huggingface_repo`

```shell
python -m classification.publish huggingface -w runs/your_model_dir -n name_of_the_model -r your/huggingface_repo
```

列出给定仓库 `your/huggingface_repo` 中的所有模型，可用于 README

```shell
python -m classification.list huggingface -r your/huggingface_repo
```

一个示例模型仓库：https://huggingface.co/deepghs/anime_style_ages
