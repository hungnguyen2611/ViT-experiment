import timm
import torch
import torch.nn as nn

from .tiny_vit import Conv2d_BN, tiny_vit_11m_224, tiny_vit_21m_224, tiny_vit_5m_224


class Tiny_Vit_21M(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = tiny_vit_21m_224(pretrained=True)
        self.base.head = nn.Linear(576, num_classes, bias=True)

    def forward(self, x):
        return self.base(x)

    def fuse_model(self):
        for m in self.base.modules():
            if isinstance(m, Conv2d_BN):
                torch.ao.quantization.fuse_modules(m, ['c', 'bn'], inplace=True)


class Tiny_Vit_11M(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = tiny_vit_11m_224(pretrained=True)
        self.base.head = nn.Linear(448, num_classes, bias=True)

    def forward(self, x):
        return self.base(x)

    def fuse_model(self):
        for m in self.base.modules():
            if isinstance(m, Conv2d_BN):
                torch.ao.quantization.fuse_modules(m, ['c', 'bn'], inplace=True)


class Tiny_Vit_5M(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = tiny_vit_5m_224(pretrained=True)
        self.base.head = nn.Linear(320, num_classes, bias=True)

    def forward(self, x):
        return self.base(x)

    def fuse_model(self):
        for m in self.base.modules():
            if isinstance(m, Conv2d_BN):
                torch.ao.quantization.fuse_modules(m, ['c', 'bn'], inplace=True)


class EfficientFormerV2_finetuned(nn.Module):
    def __init__(self, pretrained: str = None) -> None:
        super().__init__()
        self.base = timm.create_model(
            "efficientformerv2_s1",
            num_classes=3,
            distillation=False,
            pretrained=True,
            fuse=True,
        )
        if pretrained:
            if pretrained.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    pretrained, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
            checkpoint_model = checkpoint['model']
            state_dict = self.base.state_dict()
            for k in ['head.weight', 'head.bias',
                      'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.base.load_state_dict(checkpoint_model, strict=False)

    def forward(self, x):
        return self.base(x)
