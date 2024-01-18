import logging
import torch
from torchvision import transforms
import yaml

from data import MyDataset
from model import *
from losses import *
from utils import AverageMeter, accuracy
from sklearn.metrics import classification_report


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load config
with open('config/eval_config.yml', 'r') as f:
    config = yaml.safe_load(f)
device = config.get('device') if torch.cuda.is_available() else 'cpu'
batch_size = config.get('batch_size')
num_classes = config.get('num_classes')
data_dir = config.get('data_dir')
use_bbox_train = config.get('use_bbox_train')
use_bbox_val = config.get('use_bbox_val')
model_no = config.get('model_no')
use_my_model = config.get('use_my_model')
aux_logits = config.get('aux_logits')
use_dp = config.get('use_dp')
dp_gpu_id = config.get('dp_gpu_id')
model_path = config.get('model_path')


def eval_model(dataloaders_val, model_no, model, criterion, device, alt_criterion=None, print_cls_report=False):
    model.eval()
    criterion.eval()
    val_epoch_loss = AverageMeter()
    val_epoch_accuracy = AverageMeter()

    with torch.set_grad_enabled(False):
        labels_all = []
        preds_all = []
        for _, inputs, labels in dataloaders_val:
            inputs = inputs.to(device)
            labels = labels[0].to(device)

            logits = model(inputs)
            val_batch_loss = criterion(logits, labels)
            acc1 = accuracy(logits, labels.data)[0]

            val_epoch_loss.update(val_batch_loss, inputs.size(0))
            val_epoch_accuracy.update(acc1.item(), inputs.size(0))
            
            if print_cls_report:
                labels_all.append(labels)
                preds_all.append(logits.argmax(1))

        if print_cls_report:
            labels_all = torch.cat(labels_all, 0).cpu().numpy()
            preds_all = torch.cat(preds_all, 0).cpu().numpy()
            cls_report = classification_report(labels_all, preds_all, digits=4)
            logging.info(cls_report)

    return val_epoch_loss.avg, val_epoch_accuracy.avg


if __name__ == '__main__':
    
    data_transforms = {
        'val': transforms.Compose([
            # CompCars
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),          
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {x: MyDataset(
            root=data_dir, is_train=True if x == 'train' else False, 
            use_bbox=use_bbox_train if x == 'train' else use_bbox_val,
            transform=data_transforms[x]
        ) for x in ['val']
    }
    dataloaders = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4, 
            drop_last=False
        ) for x in ['val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
            
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

    elif model_no == 2:
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

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    mymodel.load_state_dict(state_dict)

    mymodel = mymodel.to(device)
    if use_dp:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=dp_gpu_id)

    # Calculate model parameters
    # model_parameters = filter(
    #     lambda p: p.requires_grad, mymodel.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # logging.info(f'Model Parameters (M): {params/1e6}')
    # params_to_update = mymodel.parameters()
    
    criterion = nn.CrossEntropyLoss().to(device)  # LabelSmoothingLoss(num_classes, 0.1)

    val_epoch_loss, val_epoch_accuracy = eval_model(
        dataloaders['val'], model_no, mymodel, criterion, device,
        print_cls_report=True
    )
