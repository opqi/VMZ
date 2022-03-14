import warnings

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import R2Plus1dStem, BasicBlock, Bottleneck


from .utils import _generic_resnet, R2Plus1dStem_Pool, Conv2Plus1D
from .bert import BERT

__all__ = ["r2plus1d_34", "r2plus1d_152",
           "rgb_r2plus1d_32f_34_bert", "rgb_r2plus1d_32f_152_bert"]


def r2plus1d_34(pretraining="", use_pool1=False, progress=False, **kwargs):
    avail_pretrainings = [
        "8_kinetics",
        "32_kinetics",
        "8_ig65m",
        "32_ig65m",
    ]
    if pretraining in avail_pretrainings:
        arch = "r2plus1d_34_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            "Unrecognized pretraining dataset, continuing with randomly initialized network."
            f" Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "r2plus1d_34"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )

    return model


def r2plus1d_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]
    if pretraining in avail_pretrainings:
        arch = "r2plus1d_152_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )

        arch = "r2plus1d_152"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )

    return model


class rgb_r2plus1d_32f_34_bert(nn.Module):
    def __init__(self, pretraining='32_ig65m', num_classes=359):
        super(rgb_r2plus1d_32f_34_bert, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.dp = nn.Dropout(p=0.8)

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features = nn.Sequential()
        layers = list(r2plus1d_34(pretraining=pretraining, num_classes=num_classes,
                      use_pool1=False, progress=True).named_children())[:-2]
        [self.features.add_module(name, child) for name, child in layers]
        self.bert = BERT(self.hidden_size, 4, hidden=self.hidden_size,
                         n_layers=self.n_layers, attn_heads=self.attn_heads)

        self.fc = nn.Linear(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), self.hidden_size, 4)
        x = x.transpose(1, 2)
        input_vectors = x
        norm = input_vectors.norm(p=2, dim=-1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output, maskSample = self.bert(x)
        classificationOut = output[:, 0, :]
        sequenceOut = output[:, 1:, :]
        norm = sequenceOut.norm(p=2, dim=-1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output = self.dp(classificationOut)
        x = self.fc(output)
        return x


class rgb_r2plus1d_32f_152_bert(nn.Module):
    def __init__(self, pretraining='ig65m_32frms', num_classes=359):
        super(rgb_r2plus1d_32f_152_bert, self).__init__()
        self.hidden_size = 2048
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.dp = nn.Dropout(p=0.8)

        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.features = nn.Sequential()
        layers = list(r2plus1d_152(pretraining=pretraining,
                      num_classes=num_classes).named_children())[:-2]
        [self.features.add_module(name, child) for name, child in layers]
        self.bert = BERT(self.hidden_size, 4, hidden=self.hidden_size,
                         n_layers=self.n_layers, attn_heads=self.attn_heads)

        self.fc = nn.Linear(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), self.hidden_size, 4)
        x = x.transpose(1, 2)
        input_vectors = x
        norm = input_vectors.norm(p=2, dim=-1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output, maskSample = self.bert(x)
        classificationOut = output[:, 0, :]
        sequenceOut = output[:, 1:, :]
        norm = sequenceOut.norm(p=2, dim=-1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output = self.dp(classificationOut)
        x = self.fc(output)
        return x
