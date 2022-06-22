import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms

# SageMaker data parallel: Import the library PyTorch API
import smdistributed.dataparallel.torch.torch_smddp

# SageMaker data parallel: Import PyTorch's distributed API
import torch.distributed as dist

# SageMaker data parallel: Initialize the process group
dist.init_process_group(backend='smddp')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Define models
def _get_model(model_type):
    if model_type == 'resnet18':
        return torchvision.models.resnet18(pretrained=False)
    else:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        return Net()

# Define data augmentation
def _get_transforms():
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        return transform

# Define data loader for training dataset
def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")
    

   
    train_set = torchvision.datasets.CIFAR10(root=training_dir, 
                                             train=True, 
                                             download=False, 
                                             transform=_get_transforms()) 
    
    # SageMaker data parallel: Set num_replicas and rank in DistributedSampler
    train_sampler = \
    torch.utils.data.distributed.DistributedSampler(train_set,
                                                    num_replicas=dist.get_world_size(),
                                                    rank=dist.get_rank())
    
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler)

# Define data loader for test dataset
def _get_test_data_loader(test_batch_size, training_dir):
    logger.info("Get test data loader")
    
    test_set = torchvision.datasets.CIFAR10(root=training_dir, 
                                            train=False, 
                                            download=False, 
                                            transform=_get_transforms())
    
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True)

# Average gradients (only for multi-node CPU)
def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

# Define training loop
def train(args):
    
    # SM DDP only supports GPU-only training
    device = torch.device("cuda")
    logger.info(
        "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
            args.backend, dist.get_world_size()
        ))

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    
    # SageMaker data parallel: Scale batch size by world size
    batch_size = args.batch_size // dist.get_world_size()
    batch_size = max(batch_size, 1)
    
    train_loader = _get_train_data_loader(batch_size, args.data_dir)
    test_loader  = _get_test_data_loader(args.test_batch_size, args.data_dir)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = _get_model(args.model_type).to(device)
    
    # SageMaker data parallel: Wrap the PyTorch model with the library's DDP
    model = torch.nn.parallel.DistributedDataParallel(model)
    
    # SageMaker data parallel: Pin each GPU to a single library process.
    local_rank = os.environ["LOCAL_RANK"] 
    torch.cuda.set_device(int(local_rank))
    model.cuda(int(local_rank))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    
    logger.info('Saving trained model only on rank 0')
    # SageMaker data parallel: Save model on master node.
    if dist.get_rank() == 0:
        save_model(model, args.model_dir)

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}\n".format(
            test_loss, correct / len(test_loader.dataset)
        )
    )

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # PyTorch environments
    parser.add_argument("--model-type",type=str,default='resnet18',
                        help="custom model or resnet18")
    parser.add_argument("--batch-size",type=int,default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size",type=int,default=1000,
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs",type=int,default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval",type=int,default=100,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--backend",type=str,default='smddp',
                        help="backend for dist. training, this script only supports gloo")

    # SageMaker environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    train(parser.parse_args())