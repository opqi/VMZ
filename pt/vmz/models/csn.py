import warnings

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import BasicStem, BasicBlock, Bottleneck

from .utils import _generic_resnet, Conv3DDepthwise, BasicStem_Pool, IPConv3DDepthwise
from .bert import BERT


__all__ = ["ir_csn_152", "ip_csn_152",
           "rgb_ir_csn_32f_152_bert", "rgb_ip_csn_32f_152_bert"]


def ir_csn_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]

    if pretraining in avail_pretrainings:
        arch = "ir_csn_152_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "ir_csn_152"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[Conv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool if use_pool1 else BasicStem,
        **kwargs,
    )

    return model


def ip_csn_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]

    if pretraining in avail_pretrainings:
        arch = "ip_csn_152_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "ip_csn_152"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[IPConv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool if use_pool1 else BasicStem,
        **kwargs,
    )

    return model


class rgb_ir_csn_32f_152_bert(nn.Module):
    def __init__(self, pretraining, num_classes):
        super(rgb_ir_csn_32f_152_bert, self).__init__()
        self.hidden_size = 2048
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.dp = nn.Dropout(p=0.8)

        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.features = nn.Sequential()
        layers = list(ir_csn_152(pretraining="ig65m_32frms",
                      num_classes=359).named_children())[:-2]
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


class rgb_ip_csn_32f_152_bert(nn.Module):
    def __init__(self, pretraining, num_classes):
        super(rgb_ip_csn_32f_152_bert, self).__init__()
        self.hidden_size = 2048
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.dp = nn.Dropout(p=0.8)

        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.features = nn.Sequential()
        layers = list(ip_csn_152(pretraining="ig65m_32frms",
                      num_classes=359).named_children())[:-2]
        [self.features.add_module(name, child) for name, child in layers]
        self.bert = BERT(self.hidden_size, 4, hidden=self.hidden_size,
                         n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
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
