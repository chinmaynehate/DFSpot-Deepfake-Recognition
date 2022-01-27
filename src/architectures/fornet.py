
from collections import OrderedDict
import torch
from efficientnet_pytorch import EfficientNet
from densenet_pytorch import DenseNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms
import timm
from . import externals

"""
Feature Extractor
"""


class FeatureExtractor(nn.Module):

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
EfficientNet-V2
"""


class TimmV2Gen(FeatureExtractor):
    def __init__(self, model: str):
        super(TimmV2Gen, self).__init__()

        self.efficientnet = timm.create_model(model, pretrained=True)
        self.classifier1 = nn.Linear(1280, 1)
        del self.efficientnet.classifier
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(p=0.5)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.forward_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self._dropout(x)
        x = self.classifier1(x)
        return x


class TimmV2(TimmV2Gen):
    def __init__(self):
        super(TimmV2, self).__init__(model='tf_efficientnetv2_l')


"""
Vision Transformer ViT
"""


class ViTGen(FeatureExtractor):
    def __init__(self, model: str):
        super(ViTGen, self).__init__()

        self.vit = timm.create_model(model, pretrained=True)
        self.classifier1 = nn.Linear(1024, 1)
        del self.vit.head
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(p=0.5)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.forward_features(x)
        #x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self._dropout(x)
        x = self.classifier1(x)
        return x


class ViT(ViTGen):
    def __init__(self):
        super(ViT, self).__init__(model='vit_large_patch16_224')


"""
NfNet
"""


class NfNet3Gen(FeatureExtractor):
    def __init__(self, model: str):
        super(NfNet3Gen, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.nfnet = timm.create_model(model, pretrained=True)
        print("Loaded pretrained weights of NfNetF3")
        self.classifier = nn.Linear(3072, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nfnet.forward_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x


class NfNet3(NfNet3Gen):
    def __init__(self):
        super(NfNet3, self).__init__(model='dm_nfnet_f3')


'''
DenseNet
'''


class DenseNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(DenseNetGen, self).__init__()

        self.densenet = DenseNet.from_pretrained(model)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        self.classifier = nn.Linear(1920, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x


class DenseNet201(DenseNetGen):
    def __init__(self):
        super(DenseNet201, self).__init__(model='densenet201')


"""
Xception
"""


class Xception(FeatureExtractor):
    def __init__(self):
        super(Xception, self).__init__()
        self.xception = externals.xception()
        self.xception.last_linear = nn.Linear(2048, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xception.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xception.forward(x)


"""
Xception
"""


class InceptionResNetV2(FeatureExtractor):
    def __init__(self):
        super(InceptionResNetV2, self).__init__()
        self.inceptionresnetv2 = externals.inceptionresnetv2()
        self.inceptionresnetv2.last_linear = nn.Linear(1536, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inceptionresnetv2.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inceptionresnetv2.forward(x)


"""
Siamese tuning
"""


class SiameseTuning(FeatureExtractor):
    def __init__(self, feat_ext: FeatureExtractor, num_feat: int, lastonly: bool = True):
        super(SiameseTuning, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError(
                'The provided feature extractor needs to provide a features() method')
        self.lastonly = lastonly
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=num_feat),
            nn.Linear(in_features=num_feat, out_features=1),
        )

    def features(self, x):
        x = self.feat_ext.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lastonly:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        if self.lastonly:
            return self.classifier.parameters()
        else:
            return self.parameters()


class XceptionST(SiameseTuning):
    def __init__(self):
        super(XceptionST, self).__init__(
            feat_ext=Xception, num_feat=2048, lastonly=True)


class InceptionResNetV2ST(SiameseTuning):
    def __init__(self):
        super(InceptionResNetV2ST, self).__init__(
            feat_ext=InceptionResNetV2, num_feat=1536, lastonly=True)


class DenseNet201ST(SiameseTuning):
    def __init__(self):
        super(DenseNet201ST, self).__init__(
            feat_ext=DenseNet201, num_feat=1920, lastonly=True)


class NfNet3ST(SiameseTuning):
    def __init__(self):
        super(NfNet3ST, self).__init__(
            feat_ext=NfNet3, num_feat=3072, lastonly=True)


class TimmV2ST(SiameseTuning):
    def __init__(self):
        super(TimmV2ST, self).__init__(
            feat_ext=TimmV2, num_feat=1280, lastonly=True)


class ViTST(SiameseTuning):
    def __init__(self):
        super(ViTST, self).__init__(feat_ext=ViT, num_feat=1024, lastonly=True)
