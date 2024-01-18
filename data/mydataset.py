import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, BatchSampler
import scipy
from scipy import io
from tqdm import tqdm
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root, is_train, use_bbox, transform=None, **kwargs):
        assert isinstance(is_train, bool)
        assert isinstance(use_bbox, bool)
        
        self.use_bbox = use_bbox

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform
        
        img_path = []
        label_model = []
        bbox = []

        if root.find('CompCarsWeb') != -1:
            if is_train:
                file = open(os.path.join(root, 'train_test_split', 'classification_train.txt'))
                img_path_ori = file.readlines()
                file.close()
                if kwargs.get("small_dataset"):
                    file = open(os.path.join(root, 'train_test_split', 'classification', 'train.txt'))
                    img_path_ori = file.readlines()
                    file.close()
                if os.path.exists(os.path.join(root, 'class_mapping_model_webnature.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_webnature.npy'),
                                                  allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                
                for i in tqdm(img_path_ori):                    
                    img_path.append(os.path.join(root, 'image', i.strip()))

                    model = '/'.join(i.split('/')[:2])
                    label_model_tmp = class_mapping_model.get(model)
                    if label_model_tmp is None:
                        class_mapping_model[model] = len(class_mapping_model) + 1
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    
                    if use_bbox:
                        file = open(os.path.join(root, 'label', i.replace('jpg', 'txt').strip()))
                        xmin, xmax, ymin, ymax = file.readlines()[-1].strip().split()
                        file.close()
                        bbox.append((int(xmin), int(xmax), int(ymin), int(ymax)))
                        
                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model_webnature.npy'), class_mapping_model)

            elif not is_train:
                file = open(os.path.join(root, 'train_test_split', 'classification_test.txt'))
                img_path_ori = file.readlines()
                file.close()
                assert os.path.exists(os.path.join(root, 'class_mapping_model_webnature.npy')), 'Class mapping (model) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_webnature.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):                    
                    img_path.append(os.path.join(root, 'image', i.strip()))

                    model = '/'.join(i.split('/')[:2])
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    
                    if use_bbox:
                        file = open(os.path.join(root, 'label', i.replace('jpg', 'txt').strip()))
                        xmin, xmax, ymin, ymax = file.readlines()[-1].strip().split()
                        file.close()
                        bbox.append((int(xmin), int(xmax), int(ymin), int(ymax)))

        elif root.find('CompCarsSV') != -1:
            annotation = scipy.io.loadmat(os.path.join(root, 'sv_make_model_name.mat'))['sv_make_model_name']
            if is_train:
                file = open(os.path.join(root, 'train_surveillance.txt'))
                img_path_ori = file.readlines()
                file.close()
                if os.path.exists(os.path.join(root, 'class_mapping_model_sv.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_sv.npy'),
                                                                allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                    
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'image', i.strip()))
                    model = i.split('/')[0]
                    label_model_tmp = class_mapping_model.get(model)
                    if label_model_tmp is None:
                        class_mapping_model[model] = len(class_mapping_model) + 1
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)

                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model_sv.npy'), class_mapping_model)
            
            elif not is_train:
                file = open(os.path.join(root, 'test_surveillance.txt'))
                img_path_ori = file.readlines()
                file.close()
                assert os.path.exists(os.path.join(root, 'class_mapping_model_sv.npy')), 'Class mapping (model) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_sv.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'image', i.strip()))
                    model = i.split('/')[0]
                    make = annotation[int(model) - 1][0].item()
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
        
        elif root.find('StanfordCars') != -1:
            cars_meta = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_meta.mat'))['class_names'][0]
            if is_train:
                img_path_ori = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_train_annos.mat'))['annotations'][0]
                if os.path.exists(os.path.join(root, 'class_mapping_model_cars.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_cars.npy'),
                                                                allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                    
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'cars_train', i[-1].item()))
                    model = i[-2].item()
                    label_model_tmp = class_mapping_model.get(model)
                    if label_model_tmp is None:
                        class_mapping_model[model] = model
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)

                    if use_bbox:
                        xmin, ymin, xmax, ymax = i[0].item(), i[2].item(), i[1].item(), i[3].item()
                        bbox.append((xmin, xmax, ymin, ymax))

                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model_cars.npy'), class_mapping_model)

            elif not is_train:
                img_path_ori = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_test_annos_withlabels.mat'))['annotations'][0]
                assert os.path.exists(os.path.join(root, 'class_mapping_model_cars.npy')), 'Class mapping (model) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_cars.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'cars_test', i[-1].item()))
                    model = i[-2].item()
                    make = cars_meta[model - 1].item().split()[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)

                    if use_bbox:
                        xmin, ymin, xmax, ymax = i[0].item(), i[2].item(), i[1].item(), i[3].item()
                        bbox.append((xmin, xmax, ymin, ymax))

        elif root.find("Car-FG3K") != -1:
            if is_train:
                with open(os.path.join(root, "191train.txt"), "r") as file:
                    img_path_ori = file.readlines()
                if os.path.exists(os.path.join(root, 'class_mapping_model.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model.npy'),
                                                  allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, i.split('\t')[0]))
                    model = '/'.join(i.split('/')[:2])
                    label_model_tmp = class_mapping_model.get(model)
                    if label_model_tmp is None:
                        class_mapping_model[model] = len(class_mapping_model) + 1
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                        
                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model.npy'), class_mapping_model)
            
            elif not is_train:
                with open(os.path.join(root, "191test.txt"), "r") as file:
                    img_path_ori = file.readlines()
                assert os.path.exists(os.path.join(root, 'class_mapping_model.npy')), 'Class mapping (model) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, i.split('\t')[0]))
                    model = '/'.join(i.split('/')[:2])
                    make = i.split('/')[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
        
        self.img_path = np.array(img_path)
        self.bbox = np.array(bbox)
        self.label_model = np.array(label_model) - 1

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx]).convert(mode='RGB')
        if self.use_bbox and len(self.bbox) > 0:
            image = image.crop((self.bbox[idx][0], self.bbox[idx][1], 
                                self.bbox[idx][2], self.bbox[idx][3]))
        image = self.transform(image)
        label = ()
        label_model = self.label_model[idx]
        label += (label_model,)
        return self.img_path[idx], image, label
        

class BalancedBatchSampler(BatchSampler):
    """
    https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, label, batch_size, n_samples_per_cls):
        self.labels_list = label
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        assert batch_size >= n_samples_per_cls
        self.batch_size = batch_size
        self.n_samples_per_cls = n_samples_per_cls
        self.n_classes_to_sample = batch_size // n_samples_per_cls
        self.dataset = dataset

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes_to_sample, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples_per_cls])
                self.used_label_indices_count[class_] += self.n_samples_per_cls
                if self.used_label_indices_count[class_] + self.n_samples_per_cls > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes_to_sample * self.n_samples_per_cls

    def __len__(self):
        return len(self.dataset) // self.batch_size
