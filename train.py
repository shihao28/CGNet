import os
import torch
from torch import optim
from torchvision import transforms, models
import logging
import numpy as np
import pandas as pd
import copy
from thop import profile
from pthflops import count_ops
import yaml

from data import MyDataset, BalancedBatchSampler
from model import *
from losses import *
from eval import eval_model
from utils import AverageMeter, accuracy, Mixup


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load config
with open('config/train_config.yml', 'r') as f:
    config = yaml.safe_load(f)
device = config.get('device') if torch.cuda.is_available() else 'cpu'
batch_size = config.get('batch_size')
epochs = config.get('epochs')
learning_rate = config.get('learning_rate')
warmup_epoch = config.get('warmup_epoch')
warmup_lr = config.get('warmup_lr')
num_classes = config.get('num_classes')
data_dir = config.get('data_dir')
use_bbox_train = config.get('use_bbox_train')
use_bbox_val = config.get('use_bbox_val')
model_no = config.get('model_no')
use_my_model = config.get('use_my_model')
aux_logits = config.get('aux_logits')
use_mixup = config.get('use_mixup')
lr_scheduler_epoch_update = config.get('lr_scheduler_epoch_update')
use_cls_weight_for_loss = config.get('use_cls_weight_for_loss')
use_batch_sampler = config.get('use_batch_sampler')
n_img_per_label = config.get('n_img_per_label')
use_dp = config.get('use_dp')
dp_gpu_id = config.get('dp_gpu_id')


def train_one_epoch(
    dataloaders_train, model_no, model, optimizer_, lr_scheduler_, 
    lr_scheduler_epoch_update
):
    
    model.train()
    criterion.train()
    train_epoch_loss = AverageMeter()
    train_epoch_accuracy = AverageMeter()

    for _, inputs, labels in dataloaders_train:
        inputs = inputs.to(device)
        labels = labels[0].to(device)

        labels_mixup = None
        if use_mixup:
            inputs, labels_mixup = mixup_fn(inputs, labels)

        optimizer_.zero_grad()

        with torch.set_grad_enabled(True):
            if model_no != 2:
                logits = model(inputs)
                train_batch_loss = criterion(logits, labels_mixup if use_mixup else labels)
                acc1 = accuracy(logits, labels.data)[0]

            elif model_no == 2:
                if model.training() and aux_logits:
                    logits, aux2, aux1 = model(inputs)
                    train_batch_loss = criterion(logits, labels_mixup if use_mixup else labels)
                    train_batch_loss += 0.3 * (criterion(aux1, labels) + criterion(aux2, labels))
                else:
                    logits = model(inputs)
                train_batch_loss = criterion(logits, labels_mixup if use_mixup else labels)

            train_batch_loss.backward()
            optimizer_.step()

        if not lr_scheduler_epoch_update:
            lr_scheduler_.step()

        train_epoch_loss.update(train_batch_loss, inputs.size(0))
        train_epoch_accuracy.update(acc1.item(), inputs.size(0))

    return train_epoch_loss.avg, train_epoch_accuracy.avg

    
def train_model(
    dataloaders, model, optimizer_, lr_scheduler_, 
    epochs=25, model_no=0, lr_scheduler_epoch_update=True, warmup=False
):

    best_state_dict = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    train_loss, train_accuracy, val_loss, val_accuracy, lr = [], [], [], [], []

    for epoch in range(epochs):

        lr.append(lr_scheduler_.get_last_lr()[0])       
        
        # Train
        train_epoch_loss, train_epoch_accuracy = train_one_epoch(
            dataloaders['train'], model_no, model, optimizer_,
            lr_scheduler_, lr_scheduler_epoch_update
        )
        if lr_scheduler_epoch_update:
            lr_scheduler_.step()
        train_loss.append(train_epoch_loss.item())
        train_accuracy.append(train_epoch_accuracy)
        logging.info(
            f"Epoch {epoch:3d}/{epochs-1:3d} {'Train':5s}, "
            f"Loss: {train_epoch_loss:.4f}, "
            f"Acc: {train_epoch_accuracy:.4f}"
        )
        if warmup:
            # If warmp, do not proceed with the rest
            continue

        # Validate
        val_epoch_loss, val_epoch_accuracy = torch.tensor(0), 0
        if epoch > 39:
            val_epoch_loss, val_epoch_accuracy = eval_model(
                dataloaders['val'], model_no, model,
                nn.CrossEntropyLoss() if use_mixup else criterion, device,
                alt_criterion=alt_criterion,
            )
        val_loss.append(val_epoch_loss.item())
        val_accuracy.append(val_epoch_accuracy)
        logging.info(
            f"Epoch {epoch:3d}/{epochs-1:3d} {'Val':5s}, "
            f"Loss: {val_epoch_loss:.4f}, "
            f"Acc: {val_epoch_accuracy:.4f}"
        )
        
        # Check if val_epoch acc > best_acc
        if val_epoch_accuracy > best_accuracy:
            best_accuracy = val_epoch_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())

    if not warmup:
        # Load best model
        model.load_state_dict(best_state_dict)
        
        # Evalaute best model
        logging.info('Best Val Acc: {:4f}'.format(best_accuracy))
        val_epoch_loss, val_epoch_accuracy = eval_model(
            dataloaders['val'], model_no, model,
            nn.CrossEntropyLoss() if use_mixup else criterion, device,
            alt_criterion=alt_criterion, print_cls_report=True
        )
        
        # Save best model
        model_name = '{}_model{}_{:.4f}'.format(os.path.split(data_dir)[1], model_no, best_accuracy)
        if use_dp:
            best_state_dict = copy.deepcopy(model.module.state_dict())    
        else:
            best_state_dict = copy.deepcopy(model.state_dict())
        torch.save(best_state_dict, f'{model_name}.pth')

        # Save training details
        pd.DataFrame({
            'Epochs': range(epochs), 'Learning Rate': lr, 'Training Loss': train_loss, 
            'Training Accuracy': train_accuracy, 'Validation Loss': val_loss, 
            'Validation Accuracy':val_accuracy
        }).to_csv(f'{model_name}.csv', index=False)

    return model


if __name__ == '__main__':
    
    logging.info(f"Running model {model_no}")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(
                size=224, scale=(0.2, 1)
            ),
            transforms.RandomHorizontalFlip(p=0.5), # 0.5   
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {x: MyDataset(
            root=data_dir, is_train=True if x=='train' else False, 
            use_bbox=use_bbox_train if x=='train' else use_bbox_val,
            transform=data_transforms[x]
        ) for x in ['train', 'val']
    }
    if use_batch_sampler:
        sampler = {'train': BalancedBatchSampler(
            image_datasets['train'], label=image_datasets['train'].label_model, 
            batch_size=batch_size, n_samples_per_cls=n_img_per_label
        )}
        dataloaders = {
            'train': torch.utils.data.DataLoader(
                image_datasets['train'], batch_sampler=sampler['train'], 
                num_workers=4, pin_memory=True
            ),
            'val': torch.utils.data.DataLoader(
                image_datasets['val'], batch_size=batch_size, shuffle=True, 
                num_workers=4, drop_last=False
            )
        }
    else:
        dataloaders = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, 
            drop_last=False
            ) # False, True if x == 'train' else False
            for x in ['train', 'val']
        }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    logging.info(f"Training on {dataset_sizes['train']} images")
    logging.info(f"Validating on {dataset_sizes['val']} images")
            
    if model_no == 0:
        # Inceptionv3
        if use_my_model:         
            mymodel = inception_v3(
                pretrained=True, num_classes=num_classes,
                aux_logits=aux_logits
            )
        else:            
            mymodel = models.inception_v3(
                pretrained=True, num_classes=1000, 
                aux_logits=aux_logits
            )
            if aux_logits:
                num_ftrs = mymodel.AuxLogits.fc.in_features
                mymodel.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)

    elif model_no == 1:
        # resnet50
        if use_my_model:
            mymodel = resnet50(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.resnet50(pretrained=True, num_classes=1000)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_no == 2:
        # densenet169
        if use_my_model:
            mymodel = densenet169(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.densenet169(pretrained=True, num_classes=1000)
            num_ftrs = mymodel.classifier.in_features
            mymodel.classifier = nn.Linear(num_ftrs, num_classes)

    mymodel = mymodel.to(device)

    # Calculate params and flops
    # input_dummy = torch.rand((1, 3, 224, 224)).to(device)
    # flops = count_ops(mymodel, input)
    # flops, params = profile(mymodel, inputs=(input_dummy, ))
    # logging.info(f'FLOPs (B): {flops/1e9}, Model Parameters (M): {params/1e6}')

    if use_dp:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=dp_gpu_id)

    # Calculate model parameters
    model_parameters = filter(lambda p: p.requires_grad, mymodel.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'Model Parameters (M): {params/1e6}')

    # class weight
    _, cls_sample_count = np.unique(image_datasets['train'].label_model, return_counts=True)
    cls_weight = cls_sample_count.sum() / (len(cls_sample_count) * cls_sample_count)
    cls_weight = torch.tensor(cls_weight, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=cls_weight if use_cls_weight_for_loss else None).to(device)
    # criterion = LabelSmoothingLoss(num_classes, 0.1).cuda()
    alt_criterion = None

    if use_mixup:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(
            mixup_alpha=0.1, cutmix_alpha=0.1, cutmix_minmax=None,
            prob=0.5, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=num_classes
        )

    params_to_update = mymodel.parameters()
    if warmup_epoch > 0:
        warmup_optimizer_ = optim.SGD(
            params_to_update, lr=warmup_lr, momentum=0.9, 
            weight_decay=5e-4
        )
        warmup_lr_scheduler_ = optim.lr_scheduler.StepLR(
            warmup_optimizer_, warmup_epoch*len(dataloaders['train'])//3+1,
            gamma=10, verbose=False
        )
        logging.info('Warming up...')
        mymodel = train_model(
            dataloaders, mymodel, warmup_optimizer_, warmup_lr_scheduler_,
            warmup_epoch, model_no, False, True
        )
        logging.info('Warming up completed')

    optimizer_ = optim.SGD(
        params_to_update, lr=learning_rate, momentum=0.9,
        weight_decay=5e-4
    )

    lr_scheduler_ = optim.lr_scheduler.StepLR(
        optimizer_, step_size=40, gamma=0.1
    )

    mymodel = train_model(
        dataloaders, mymodel, optimizer_, lr_scheduler_, 
        epochs, model_no, lr_scheduler_epoch_update, False
    )
