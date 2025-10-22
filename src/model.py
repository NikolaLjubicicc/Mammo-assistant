import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def create_model(pretrained=True, fine_tune_layers=3):
    if pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = mobilenet_v3_small(weights=None)
    
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    if fine_tune_layers > 0:
        for param in model.features[-fine_tune_layers:].parameters():
            param.requires_grad = True
    
    return model