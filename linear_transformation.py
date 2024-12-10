import os
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from utils import load_model, tensor_to_json_compatible
import torch.nn as nn
import wandb
from tqdm import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

batchsize = 256
epochs = 10

model1_name = 'ResNet18 InfoNCE loss'
model2_name = 'ResNet18 Kernel-InfoNCE loss'
model1_ckpt= '/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/sparkling-plasma-91/checkpoints/epoch=367-step=61456.ckpt' 
model2_ckpt = '/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/likely-surf-92/checkpoints/epoch=360-step=60287.ckpt'

wandb.init(project='representation-comparison', 
           config={'dataset': 'cifar10', 'arch': 'resnet18', 
                   'model1': model1_name, 'model1_ckpt': model1_ckpt,
                   'model2': model2_name, 'model2_ckpt': model2_ckpt,
                   'batchsize': batchsize, 'epochs': epochs})


transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
])

dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader_train = DataLoader(
    dataset_train, 
    batch_size=batchsize, 
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader_test = DataLoader(
    dataset_test, 
    batch_size=batchsize, 
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

class LinearMap(nn.Module):
    def __init__(self, input_dim):
        super(LinearMap, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=False) 

    def forward(self, x):
        return self.linear(x)

model_nt_xent = load_model(model1_ckpt) 
model_origin = load_model(model2_ckpt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device} as device')

representation_dim = 1000
linear_map = LinearMap(representation_dim).to(device)
model_nt_xent = model_nt_xent.to(device)
model_origin = model_origin.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(linear_map.parameters(), lr=1e-3)

for epoch in range(epochs):  
    wandb.log({"epoch": epoch})

    linear_map.train()
    train_loss_values = []
    train_progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
    for idx, (images, labels) in enumerate(train_progress_bar):
        images = images.to(device)
        
        with torch.no_grad():
            rep1 = model_nt_xent(images) 
            rep2 = model_origin(images) 
        
        rep1_mapped = linear_map(rep1)
        loss = criterion(rep1_mapped, rep2)
        train_loss_values.append(loss.item())
        wandb.log({"train_loss": loss.item()})
        
        train_progress_bar.set_postfix(loss=loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    test_loss_values = []
    linear_map.eval()
    with torch.no_grad():
        test_progress_bar = tqdm(dataloader_test, desc="Testing", leave=False)
        for idx, (images, labels) in enumerate(test_progress_bar):
            images = images.to(device)

            rep1 = model_nt_xent(images)
            rep2 = model_origin(images)
            rep1_mapped = linear_map(rep1)
            test_loss = criterion(rep1_mapped, rep2)
            test_loss_values.append(test_loss.item())
            wandb.log({"test_loss": test_loss.item()})
            test_progress_bar.set_postfix(test_loss=test_loss.item())

    wandb.log({"train_loss_avg_epoch": sum(train_loss_values)/len(train_loss_values)})
    wandb.log({"test_loss_avg_epoch": sum(test_loss_values)/len(test_loss_values)})



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/linear-tranformation", timestamp)
os.makedirs(folder_name, exist_ok=True)
save_path = os.path.join(folder_name, "linear_map.pth")
torch.save(linear_map.state_dict(), save_path)

wandb.finish()