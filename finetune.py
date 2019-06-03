from torchvision import models
from torch import nn
from torch import optim
import torch

resnet_model = models.resnet50(pretrained=True)
# pretrained 设置为 True，会自动下载模型 所对应权重，并加载到模型中
# 也可以自己下载 权重，然后 load 到 模型中，源码中有 权重的地址。

# 假设 我们的 分类任务只需要 分 100 类，那么我们应该做的是
# 1. 查看 resnet 的源码
# 2. 看最后一层的 名字是啥 （在 resnet 里是 self.fc = nn.Linear(512 * block.expansion, num_classes)）
# 3. 在外面替换掉这个层
resnet_model.fc= nn.Linear(in_features=512, out_features=100)

# 这样就 哦了，修改后的模型除了输出层的参数是 随机初始化的，其他层都是用预训练的参数初始化的。

# 如果只想训练 最后一层的话，应该做的是：
# 1. 将其它层的参数 requires_grad 设置为 False
# 2. 构建一个 optimizer， optimizer 管理的参数只有最后一层的参数
# 3. 然后 backward， step 就可以了

ignored_params = list(map(id, resnet_model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     resnet_model.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': resnet_model.fc.parameters(), 'lr': 1e-2}
            ], lr=1e-3, momentum=0.9)
