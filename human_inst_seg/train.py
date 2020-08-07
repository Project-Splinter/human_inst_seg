import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision

from dataset import Dataset
from unet import Segmentation

loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)

os.makedirs("./data/snapshots/", exist_ok=True)
os.makedirs("./data/visualize/train/", exist_ok=True)
os.makedirs("./data/visualize/test/", exist_ok=True)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_Softmax(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
    input_norm = input[0:8] * 0.5 + 0.5 #[-1, 1] -> [0, 1]
    output_norm = F.softmax(output, dim=1)[0:8, 1:2].repeat(1, 3, 1, 1).float()
    target_norm = target[0:8].unsqueeze(1).repeat(1, 3, 1, 1).float()

    torchvision.utils.save_image(
        torch.cat([input_norm, output_norm, target_norm], dim=0),
        f"./data/visualize/train/latest.jpg", 
        normalize=True, range=(0, 1), nrow=len(input_norm), padding=10, pad_value=0.5
    )
    
    torch.save(
        model.state_dict(), 
        f"./data/snapshots/latest.pt",
    )

best = 0.0
def test(args, model, device, test_loader):
    model.eval()
    correct = 0
    iou = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            
            input_norm = data * 0.5 + 0.5 #[-1, 1] -> [0, 1]
            output_norm = output[:, 3:4, :, :].repeat(1, 3, 1, 1).float()
            target_norm = target.unsqueeze(1).repeat(1, 3, 1, 1).float()
            torchvision.utils.save_image(
                torch.cat([input_norm, output_norm, target_norm], dim=0),
                f"./data/visualize/test/latest_{batch_idx}.jpg", 
                normalize=True, range=(0, 1), nrow=len(input_norm), padding=10, pad_value=0.5
            )
            
            pred = output_norm>0.5
            gt = target_norm>0.5
            correct += pred.eq(gt).sum().item() / gt.numel()
            iou += ((pred & gt).sum()+1e-6) / ((pred | gt).sum()+1e-6)

    print('\nTest set: , Accuracy: {:.2f}%, IOU: {:.4f}\n'.format(
        100. * correct / len(test_loader.dataset) * pred.size(0),
        iou / len(test_loader.dataset) * pred.size(0)
    ))

    global best
    if iou > best:
        best = iou
        torch.save(
            model.state_dict(), 
            f"./data/snapshots/best-{(iou*100): .2f}.pt",
        )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=20.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='dS',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--ckpt', type=str, default="",
                        help='load from which file.')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        Dataset(input_size=256, train=True,
                image_dir="./data/ATR_RemoveBG/JPEGImages/", 
                label_dir="./data/ATR_RemoveBG/RemoveBG/"),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Dataset(input_size=256, train=False, 
                image_dir="./data/alignment", 
                label_dir="./data/alignment"),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Segmentation().to(device)
    model.train()

    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt))
        print (f"load from snapshots: {args.ckpt}")
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print (f"lr: {scheduler.get_lr()[0]}")
        test(args, model, device, test_loader)
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()


if __name__ == '__main__':
    main()
