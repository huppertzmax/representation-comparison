import os
import torch
import random
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_cka import CKA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils import load_model, tensor_to_json_compatible

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

    
def filter_layers(layer):
    return 'relu' in layer or 'avgpool' in layer or 'fc' in layer
    
transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #TODO check if needed
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

model1_name = 'ResNet18 InfoNCE loss'
model2_name = 'ResNet18 Kernel-InfoNCE loss'
model1_ckpt= '/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/sparkling-plasma-91/checkpoints/epoch=367-step=61456.ckpt' 
model2_ckpt = '/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/likely-surf-92/checkpoints/epoch=360-step=60287.ckpt'
model_nt_xent = load_model(model1_ckpt) 
model_origin = load_model(model2_ckpt)

layers_model_nt_xent = [name for name, module in model_nt_xent.named_modules()]
layers_model_origin = [name for name, module in model_origin.named_modules()]

filtered_layers_model_nt_xent = list(filter(filter_layers, layers_model_nt_xent))
filtered_layers_model_origin = list(filter(filter_layers, layers_model_origin))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device} as device')

cka = CKA(model_nt_xent, model_origin,
          model1_name=model1_name,   
          model2_name=model2_name,  
          model1_layers=filtered_layers_model_nt_xent,
          model2_layers=filtered_layers_model_origin, 
          device=device)

cka.compare(dataloader) 

results = cka.export()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/cka", timestamp)
os.makedirs(folder_name, exist_ok=True)
save_path = os.path.join(folder_name, "cka_heatmap.png")
cka.plot_results(save_path=save_path)

similarity_matrix = results['CKA'].numpy()

results = tensor_to_json_compatible(results)
results[model1_name] = model1_ckpt
results[model2_name] = model2_ckpt

with open(folder_name + "/cka_results.json", 'w') as f:
    json.dump(results, f, indent=4)

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, xticklabels=filtered_layers_model_origin, yticklabels=filtered_layers_model_nt_xent, cmap='magma')
plt.title('CKA Similarity Heatmap')
plt.xlabel(model2_name)
plt.gca().invert_yaxis()
plt.ylabel(model1_name)
plt.tight_layout()

plt.savefig(folder_name + "/cka_sns_heatmap.png")
plt.show()