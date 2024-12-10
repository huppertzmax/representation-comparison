import torch
import json 
from torchvision import models, transforms
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def tensor_to_json_compatible(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: tensor_to_json_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_json_compatible(item) for item in obj]
    else:
        return obj
    
def filter_layers(layer):
    return 'relu' in layer or 'avgpool' in layer or 'fc' in layer

#ckpt_path = "/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/eternal-durian-89/checkpoints/epoch=387-step=64796.ckpt"
ckpt_path = "/dss/dsshome1/lxc03/apdl006/thesis/code/spectral_contrastive_learning/log/spectral/completed-2024-12-02spectral-resnet18-mlp1000-norelu-cifar10-lr003-mu1-log_freq:50/checkpoints/400.pth"
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
print(checkpoint)
print('################################################')
state_dict = checkpoint["state_dict"]
print(state_dict.keys())

new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("encoder."):
        new_key = key.replace("encoder.", "")  
        new_state_dict[new_key] = value
    elif key.startswith("backbone."):
        new_key = key.replace("backbone.", "")  
        new_state_dict[new_key] = value
    elif key.startswith("0.") and "proj_resnet_" not in key:
        new_key = key.replace("0.", "")  
        new_state_dict[new_key] = value

print('################################################')
print(new_state_dict.keys())


model = models.resnet18()  
model.load_state_dict(new_state_dict)  
model.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet models expect 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

image, label = next(iter(data_loader))  

#layer_names_resnet18 = [name for name, module in model1.named_modules()]
layer_names_resnet50 = [name for name, module in model.named_modules()]
print(layer_names_resnet50)
print(list(filter(filter_layers, layer_names_resnet50)))
#layer_names_resnet34 = [name for name, module in model2.named_modules()]

with torch.no_grad():
    output = model(image)  

print(f"True Label: {label.item()}")
#print(f"Model Output: {output}")


with open('output.json', 'w') as f:
    json.dump(tensor_to_json_compatible(output), f, indent=4)