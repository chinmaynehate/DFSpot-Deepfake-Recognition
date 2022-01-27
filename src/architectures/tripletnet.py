
from . import fornet
from .fornet import FeatureExtractor


class TripletNet(FeatureExtractor):

    def __init__(self, feat_ext: FeatureExtractor):
        super(TripletNet, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError(
                'The provided feature extractor needs to provide a features() method')

    def features(self, x):
        return self.feat_ext.features(x)

    def forward(self, x1, x2, x3):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = self.features(x3)
        return x1, x2, x3


class DenseNet201(TripletNet):
    def __init__(self):
        super(DenseNet201, self).__init__(feat_ext=fornet.DenseNet201)


class InceptionResNetV2(TripletNet):
    def __init__(self):
        super(InceptionResNetV2, self).__init__(
            feat_ext=fornet.InceptionResNetV2)


class Xception(TripletNet):
    def __init__(self):
        super(Xception, self).__init__(feat_ext=fornet.Xception)


class NfNet3(TripletNet):
    def __init__(self):
        super(NfNet3, self).__init__(feat_ext=fornet.NfNet3)


class TimmV2(TripletNet):
    def __init__(self):
        super(TimmV2, self).__init__(feat_ext=fornet.TimmV2)


class ViT(TripletNet):
    def __init__(self):
        super(ViT, self).__init__(feat_ext=fornet.ViT)
