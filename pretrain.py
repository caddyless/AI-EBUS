import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from model.interface import obtain_net
from torch.utils.tensorboard import SummaryWriter
from args import args, device
from metainfo.schedule import get_schedule
from datamanager.data_loader import data_loader


def train(net, save_dir, batch_size=32, epoch=100, writer=None):

    model_dir = save_dir.replace('tensorboard', 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    train_loader, val_loader = data_loader(batch_size)

    optimizer, scheduler = get_schedule('imagenet', net.model.parameters())

    if args.resume and os.path.isfile('../models/pretrain/checkpoint.pth'):
        state = torch.load('../models/pretrain/checkpoint.pth')
        start = state['epoch']
        # start = 1
        net.model.module.load_state_dict(state['param'])
        # optimizer.load_state_dict(state['optimizer'])
    else:
        start = 0

    for e in range(start, epoch):
        avg_loss = update_loop(net, train_loader, optimizer, scheduler, e,
                               os.path.join(model_dir, 'checkpoint.pth'))
        acc = evaluation(net, val_loader)
        writer.add_scalar('Accuracy', acc, e)
        scheduler.step()
    return


def save_model(params, optimal_param, epoch, save_path):
    state = {'param': params, 'epoch': epoch,
             'optimizer': optimal_param}
    torch.save(state, save_path)


def evaluation(model, val_loader):
    model.model.eval()
    correct = 0
    total = 0
    bar = tqdm(val_loader)
    with torch.no_grad():
        for i, (x, label, number) in enumerate(bar):
            label.squeeze_()
            scores = model.set_forward(x)
            scores = F.softmax(scores, dim=1)
            total += scores.size(0)
            _, predicted = torch.max(scores.data, 1)
            label = label.to(device)
            correct += (label == predicted).sum().item()
            bar.set_description('Iteration: {:5d} | Accuracy: {:4.2f}%'.format(i, 100 * correct / total))
        acc = 100 * correct / total
        print('Accuracy is {:4.2f}%'.format(acc))
        return acc


def update_loop(model, train_loader, optimizer, scheduler, epoch, save_path):
    model.model.train()

    avg_loss = 0
    length = len(train_loader)
    bar = tqdm(train_loader)
    for i, (x, label, number) in enumerate(bar):
        optimizer.zero_grad()
        loss = net.set_loss(x, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_loss = (avg_loss * i + loss.data.item()) / float(i + 1)
        writer.add_scalar('loss', loss.data.item(), epoch * length + i)
        if i % 100 == 0:
            save_model(model.model.module.state_dict(), optimizer.state_dict(),
                       epoch, save_path)
        if i % 2000 == 0:
            for name, param in model.model.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)
        bar.set_description('Epoch {:d} | Batch {:2d}/{:d} | Loss {:f} | lr {:f}'.format(
                epoch, i, len(train_loader), loss.data.item(), optimizer.param_groups[0]['lr']))
    return avg_loss


writer_dir = os.path.join('../tensorboard/pretrain')
writer = SummaryWriter(writer_dir)
params = {'model': args.backbone, 'net': args.net, 'writer': writer}
net = obtain_net(**params)
train(net, writer_dir, args.batch, args.epoch, writer)

