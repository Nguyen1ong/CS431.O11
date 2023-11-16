import torchvision
import torch
from .TDRG import TDRG

model_dict = {'TDRG': TDRG}

def get_model(num_classes, args):
    try:
        # res101 = torchvision.models.resnet101(weights='DEFAULT')
        res101 = torchvision.models.resnet101()
        res101.load_state_dict(torch.load('/kaggle/input/resnes101/resnet101-63fe2227.pth'))
    except Exception as e:
        print('Exception: {}'.format(e))
    model = model_dict[args.model_name](res101, num_classes)
    return model
