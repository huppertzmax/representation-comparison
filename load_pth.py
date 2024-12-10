import torch
from torchvsision.models import resnet18

ckpt_path = ""
model = resnet18()
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['state_dict']
load_checkpoint(model, state_dict, ckpt_path, args=args, nomlp=args.nomlp)