# Modify original PyTorch ResNet architecutres for semantic segmentation application
# load_resnet_model(layers, arch_type='fcn', backbone=custom_resnet, pretrained=False, progress=True, num_classes=1, aux_loss=None, **kwargs) is to be used to retrieve a model

# Import necessary dependencies from PyTorch's Torchvision library
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock, Bottleneck, ResNet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead


# Links to download pre-trained models for standard ResNet architectures
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

# Build ResNet backbone of model using imported ResNet class
def custom_resnet(layers, pretrained=False, progress=True, arch='resnet', **kwargs):
    """ Builds custom ResNet backbone
    Arguments:
        layers (list): configuration of layer-blocks (Bottlenecks)
        pretrained (bool): If True, returns a model pre-trained on ImageNet dataset
        progress (bool): If True, shows progress bar while downloading model
        arch (str): give architecture name if pretrained=True to fetch model params
    """
    model = ResNet(Bottleneck, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# Create model with modified backbone for semantic segmentation
def build_fcn_resnet(name, backbone_fct, num_classes, aux, layers, pretrained_backbone=False):
    """Constructs a custom ResNet backbone and modifies for fully convolutional, semantic segmentation
    Args:
        name (str): either deeplabv3 or fcn
        backbone_fct (function): the model function for the non-fcn-only backbone
        num_classes (int): number of classes
        aux (bool): use of auxiliary loss
        layers (list): configuration of layer-blocks (Bottlenecks)
        pretrained_backbone (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
    """
    backbone = backbone_fct(
        layers=layers,
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


# Load model and initialize weights and biases if pretrained
def load_resnet_model(layers, arch_type='fcn', backbone=custom_resnet, pretrained=False, progress=True, num_classes=1, aux_loss=None, **kwargs):
    """Constructs a fully-convolutional network model with a custom ResNet backbone.
    Arguments:
        layers (list): configuration of layer-blocks (Bottlenecks)
        arch_type (str): choose 'fcn' for fully-convolutional network
        backbone (function): choose the backbone; either standard of custom
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of classes, here standard is binary
        aux_loss (bool): Use of auxiliary loss function
    """
    if pretrained:
        aux_loss = True
    model = build_fcn_resnet(arch_type, backbone, num_classes, aux_loss, layers, **kwargs)
    if pretrained:
        arch = arch_type + '_' + str(backbone.__name__) + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model
