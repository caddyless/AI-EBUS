import torch.optim as optim
from model.utils import GradualWarmupScheduler, RAdam


def get_schedule(learning_type, net, total_epoch=150, init_lr=0.01,
                 warm_epoch=10, last_epoch=10, final_lr=1e-4, finetune=False, lr_times=100):
    print(finetune, lr_times)
    step = 5
    decay_epochs = total_epoch - warm_epoch - last_epoch
    if init_lr == 0:
        gamma = 0
    else:
        gamma = (final_lr / (init_lr * 10)) ** (step / decay_epochs)
    print('init with %f, warm up to epoch %d, then decay with gamma %f %d epochs every %d epoch' % (init_lr, warm_epoch, gamma, decay_epochs, step))
    if learning_type == 'mix':
        optim_params = {'lr': init_lr, 'weight_decay': 5e-4}
        schedule_params = {'milestones': list(range(0, 130, 5)), 'gamma': 0.8}
        warm_params = [10, warm_epoch]

    elif learning_type == 'folder':
        optim_params = {'lr': init_lr, 'weight_decay': 5e-4}
        schedule_params = {'milestones': list(range(0, decay_epochs, step)), 'gamma': gamma}
        warm_params = [10, warm_epoch]

    elif learning_type == 'cen':
        optim_params = {'lr': init_lr, 'weight_decay': 5e-4}
        schedule_params = {'milestones': list(range(0, decay_epochs, step)), 'gamma': gamma}
        warm_params = [10, warm_epoch]

    elif learning_type == 'union':
        optim_params = {'lr': init_lr, 'weight_decay': 5e-4}
        schedule_params = {'milestones': list(range(0, decay_epochs, step)), 'gamma': gamma}
        warm_params = [15, warm_epoch]

    elif learning_type == 'compare':
        optim_params = {'lr': init_lr, 'weight_decay': 5e-4}
        schedule_params = {'milestones': list(range(0, 130, 5)), 'gamma': 0.8}
        warm_params = [10, warm_epoch]
        optimizer1 = optim.Adam(filter(lambda x: x.requires_grad is not False, net.parameters[0]), **optim_params)
        optimizer2 = optim.Adam(filter(lambda x: x.requires_grad is not False, net.parameters[1]), **optim_params)
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, **schedule_params)
        scheduler1 = GradualWarmupScheduler(optimizer1, *warm_params, scheduler1)
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, **schedule_params)
        scheduler2 = GradualWarmupScheduler(optimizer2, *warm_params, scheduler2)
        return optimizer1, optimizer2, scheduler1, scheduler2

    elif learning_type == 'imagenet':
        optim_params = {'lr': init_lr, 'weight_decay': 5e-4}
        schedule_params = {'milestones': list(range(0, 100 * 10010, 5000)), 'gamma': 0.96979}
        warm_params = [1, 1]

    else:
        raise ValueError('Unknow learning type %s' % learning_type)
    
    if finetune:
        params_network = []
        params_fusion = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                if 'fusion' in k:
                    params_fusion.append(v)
                else:
                    params_network.append(v)
        optimizer = optim.Adam([{'params': params_network}, {'params': params_fusion, 'lr': optim_params['lr'] * lr_times}], **optim_params)
    
    else:
        optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False, net.parameters()), **optim_params)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **schedule_params)
    scheduler = GradualWarmupScheduler(optimizer, *warm_params, scheduler)

    return optimizer, scheduler

