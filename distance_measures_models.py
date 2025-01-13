import os
import torch
import random
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from utils import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/distances", timestamp)
os.makedirs(folder_name, exist_ok=True)


parser = ArgumentParser()

parser.add_argument("--normed_euc", type=bool, default=True)
parser.add_argument("--model1_name", type=str, default="ResNet18 InfoNCE loss")
parser.add_argument("--model2_name", type=str, default="ResNet18 Kernel-InfoNCE loss")
parser.add_argument("--model1_path", type=str, default="/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/sparkling-plasma-91/checkpoints/epoch=367-step=61456.ckpt")
parser.add_argument("--model2_path", type=str, default="/dss/dsshome1/lxc03/apdl006/thesis/code/Kernel-InfoNCE/Kernel-InfoNCE/likely-surf-92/checkpoints/epoch=360-step=60287.ckpt")

args = parser.parse_args()

model1_name = args.model1_name
model2_name = args.model2_name
model1_ckpt=  args.model1_path
model2_ckpt = args.model2_path
use_normed_euc = args.normed_euc

batchsize = 32

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


label_map = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device} as device')

model1 = load_model(model1_ckpt).to(device)
model2 = load_model(model2_ckpt).to(device)
model1.eval()
model2.eval()

def visualize_distances(mean_distances, highest_distances, lowest_distances,
                         title, output_path="batch_distances.png"):
    assert len(mean_distances) == len(highest_distances) == len(lowest_distances), \
        "All input lists must have the same length."

    batches = list(range(1, len(mean_distances) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(batches, mean_distances, label="Mean Distance", marker="o")
    plt.plot(batches, highest_distances, label="Highest Distance", marker="^")
    plt.plot(batches, lowest_distances, label="Lowest Distance", marker="v")

    plt.xlabel("Batch Number")
    plt.ylabel("Distance")
    plt.title(title)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(output_path)
    plt.close()

def append_infos(distances, labels, mean_list, min_list, max_list, min_indices_list, max_indices_list):
    mean = torch.mean(distances)
    min = torch.min(distances, dim=0)
    max = torch.max(distances, dim=0)

    mean_list.append(mean.item())
    min_list.append(min.values.item())
    min_indices_list.append(labels[int(min.indices.item())].item())
    max_list.append(max.values.item())
    max_indices_list.append(labels[int(max.indices.item())].item())
    return mean.item()
    
def count_index_occurrences_in_list(indices_list, label_map):
    counts = {i: 0 for i in range(10)}
    for index in indices_list:
        if 0 <= index <= 9:
            counts[index] += 1

    return {f"{label_map[i]}": counts[i] for i in counts}

cos = torch.nn.CosineSimilarity(dim=1)
euc = torch.nn.PairwiseDistance(p=2)
def normed_euc(rep1, rep2):
    normed_rep1 = torch.nn.functional.normalize(rep1, dim=1)
    normed_rep2 = torch.nn.functional.normalize(rep2, dim=1)
    return euc(normed_rep1, normed_rep2)



def compute_distances(model1, model2, dataloader, desc, data):
    cos_dist = []
    cos_min_dist = []
    cos_max_dist = []
    cos_min_ind = []
    cos_max_ind = []
    euc_dist = []
    euc_min_dist = []
    euc_max_dist = []
    euc_min_ind = []
    euc_max_ind = []

    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    for images, labels in progress_bar:
        images = images.to(device)

        with torch.no_grad():
            rep1 = model1(images)
            rep2 = model2(images)
            c_dist = cos(rep1, rep2)
            c_dist = torch.sub(torch.ones_like(c_dist), c_dist)
            e_dist = normed_euc(rep1, rep2) if use_normed_euc else euc(rep1, rep2) 

        c_dist_mean = append_infos(
            c_dist, labels, cos_dist, cos_min_dist, cos_max_dist, cos_min_ind, cos_max_ind
        )
        e_dist_mean = append_infos(
            e_dist, labels, euc_dist, euc_min_dist, euc_max_dist, euc_min_ind, euc_max_ind
        )
        
        visualize_distances(cos_dist, cos_max_dist, cos_min_dist,
                     f"Cosine distances - {desc}", folder_name + f"/cos_{data}.png")

        visualize_distances(euc_dist, euc_max_dist, euc_min_dist,
                     f"Euclidean distances - {desc}", folder_name + f"/euc_{data}.png")

        progress_bar.set_postfix(euc=e_dist_mean, cos=c_dist_mean)

    cos_avg = sum(cos_dist) / len(cos_dist)
    euc_avg = sum(euc_dist) / len(euc_dist)

    return cos_avg, euc_avg, cos_min_ind, cos_max_ind, euc_min_ind, euc_max_ind


cos_dist_train, euc_dist_train, train_cos_min_ind, train_cos_max_ind, train_euc_min_ind, train_euc_max_ind = compute_distances(
    model1, model2, dataloader_train, "Training Data", "train")

cos_dist_test, euc_dist_test, test_cos_min_ind, test_cos_max_ind, test_euc_min_ind, test_euc_max_ind = compute_distances(
    model1, model2, dataloader_test, "Test Data", "test")

cos_dist = (cos_dist_train + cos_dist_test)/2.
euc_dist = (euc_dist_train + euc_dist_test)/2.

euclidean_name = "normed_euclidean_distance" if use_normed_euc else "euclidean_distance"
results = {
    "timestamp": timestamp,
    "model1_name": model1_name,
    "model1_ckpt": model1_ckpt,
    "model2_name": model2_name,
    "model2_ckpt": model2_ckpt,
    "cosine_distance": {
        "train_data": cos_dist_train,
        "test_data": cos_dist_test,
        "total": cos_dist,
        "min_train": count_index_occurrences_in_list(train_cos_min_ind, label_map),
        "min_test": count_index_occurrences_in_list(test_cos_min_ind, label_map),
        "max_train": count_index_occurrences_in_list(train_cos_max_ind, label_map),
        "max_test": count_index_occurrences_in_list(test_cos_max_ind, label_map),
    },
    euclidean_name: {
        "train_data": euc_dist_train,
        "test_data": euc_dist_test,
        "total": euc_dist,
        "min_train": count_index_occurrences_in_list(train_euc_min_ind, label_map),
        "min_test": count_index_occurrences_in_list(test_euc_min_ind, label_map),
        "max_train": count_index_occurrences_in_list(train_euc_max_ind, label_map),
        "max_test": count_index_occurrences_in_list(test_euc_max_ind, label_map),
    },
}

with open(folder_name + "/distances.json", 'w') as f:
    json.dump(results, f, indent=4)