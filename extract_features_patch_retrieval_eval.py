import torch.nn as nn
import vision_transformer as vits
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import timm
from transformers import ResNetModel, BeitModel, BitModel, ConvNextModel, CvtModel, DeiTModel, \
    DinatModel, DPTModel, EfficientFormerModel, GLPNModel, MobileNetV1Model, ImageGPTModel, \
    LevitModel, MobileNetV1Model, MobileNetV2Model, MobileViTModel, NatModel, PoolFormerModel, \
    SwinModel, Swinv2Model, ViTModel, ViTHybridModel, ViTMAEModel, ViTMSNModel, CLIPModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor, CLIPProcessor
from common import HF_MODELS_DICT
import ResNet as ResNet
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
import sys, os, glob
import pandas as pd
import time
import math
from collections import Counter, defaultdict
from typing import List, Union, Tuple
import argparse
import h5py
import torch
import openslide
import copy
from collections import OrderedDict
from torchvision.models import densenet121
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pdb
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from tensorflow.keras.backend import bias_add, constant
import clip
from types import SimpleNamespace
import json
import faiss


class PatchDataset_bak(Dataset):
    # patch_name = patch_label_file.loc[idx, 'Patch Names']
    # label = patch_label_file.loc[idx, 'label']
    def __init__(self, csv_filename, patch_data_path, patch_name_col='Patch Names', label_col='label', transform=None):
        super().__init__()
        self.df = pd.read_csv(csv_filename)
        self.patch_names = [os.path.join(patch_data_path, fname) for fname in self.df[patch_name_col].values.tolist()]
        self.labels = self.df[label_col].values.tolist()
        self.transform = transform
    def __len__(self):
        return len(self.patch_names)
    def __getitem__(self, idx):
        im = Image.open(self.patch_names[idx]).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.labels[idx]

class PatchDataset(Dataset):
    # patch_name = patch_label_file.loc[idx, 'Patch Names']
    # label = patch_label_file.loc[idx, 'label']
    def __init__(self, csv_filename, patch_data_path, patch_name_col='Patch Names', label_col='label', image_resize=None):
        super().__init__()
        self.df = pd.read_csv(csv_filename)
        self.patch_names = [os.path.join(patch_data_path, fname) for fname in self.df[patch_name_col].values.tolist()]
        self.labels = self.df[label_col].values.tolist()
        self.image_resize = image_resize
    def __len__(self):
        return len(self.patch_names)
    def __getitem__(self, idx):
        im = Image.open(self.patch_names[idx]).convert('RGB')
        if self.image_resize is not None:
            im = im.resize((self.image_resize, self.image_resize))
        return im, self.labels[idx]




def get_kimianet_model(network_input_patch_width, weights_address='./backbones/KimiaNetKerasWeights.h5'):
    '''
    Function to get the KimiaNet model
    Args:
        network_input_patch_width: width of the input patch
        weights_address: address of the weights
    Returns:
        kn_feature_extractor_seq: Sequential model with the KimiaNet model
    '''
    dnx = DenseNet121(include_top=False, weights=weights_address,
                      input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')

    kn_feature_extractor = Model(
        inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))

    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn_kimianet,
                                                  arguments={
                                                      'network_input_patch_width': network_input_patch_width},
                                                  input_shape=(None, None, 3), dtype=tf.float32)])

    kn_feature_extractor_seq.add(kn_feature_extractor)

    return kn_feature_extractor_seq


def preprocessing_fn_densenet121(inp, sz=(1000, 1000)):
    '''
    Function to preprocess the input image for densenet121
    Args:
        inp: Input image
        sz: Size of the image
    Returns:
        out: Preprocessed image
    '''
    # cast to float
    out = tf.cast(inp, 'float') / 255.
    # resize
    out = tf.cond(tf.equal(tf.shape(inp)[1], sz[0]),
                  lambda: out, lambda: tf.image.resize(out, sz))
    # normalize
    mean = tf.reshape((0.485, 0.456, 0.406), [1, 1, 1, 3])
    std = tf.reshape((0.229, 0.224, 0.225), [1, 1, 1, 3])
    out = out - mean
    out = out / std
    # Return the output
    return out


def preprocessing_fn_densenet121_224(inp, sz=(224, 224)):
    '''
    Function to preprocess the input image for densenet121
    Args:
        inp: Input image
        sz: Size of the image
    Returns:
        out: Preprocessed image
    '''
    # cast to float
    out = tf.cast(inp, 'float') / 255.
    # resize
    out = tf.cond(tf.equal(tf.shape(inp)[1], sz[0]),
                  lambda: out, lambda: tf.image.resize(out, sz))
    # normalize
    mean = tf.reshape((0.485, 0.456, 0.406), [1, 1, 1, 3])
    std = tf.reshape((0.229, 0.224, 0.225), [1, 1, 1, 3])
    out = out - mean
    out = out / std
    # Return the output
    return out


def preprocessing_fn_kimianet(input_batch, network_input_patch_width):
    '''
    Function to preprocess the input batch for KimiaNet
    Args:
        input_batch: batch of images
        network_input_patch_width: width of the input patch
    Returns:
        standardized_input_batch: standardized input batch
    '''
    # get the original input size
    org_input_size = tf.shape(input_batch)[1]
    # standardization
    scaled_input_batch = tf.cast(input_batch, 'float') # / 255.
    # resizing the patches if necessary
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width),
                                  lambda: scaled_input_batch,
                                  lambda: tf.image.resize(scaled_input_batch,
                                                          (network_input_patch_width, network_input_patch_width)))
    # normalization, this is equal to tf.keras.applications.densenet.preprocess_input()---------------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_format = "channels_last"
    mean_tensor = constant(-np.array(mean))
    standardized_input_batch = bias_add(
        resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= std
    return standardized_input_batch


def get_dn121_model():
    '''
    Function to get the DenseNet121 model
    Returns:
        seq_model: Sequential model with the DenseNet121 model
    '''
    # get the model
    model = tf.keras.applications.densenet.DenseNet121(input_shape=(1000, 1000, 3),
                                                       include_top=False,
                                                       pooling='avg')
    # add the preprocessing layer
    seq_model = tf.keras.models.Sequential([tf.keras.layers.Lambda(preprocessing_fn_densenet121,
                                                                   input_shape=(
                                                                       None, None, 3),
                                                                   dtype=tf.uint8)])
    # add the model
    seq_model.add(model)
    # return the model
    return seq_model


def get_dn121_model_224():
    '''
    Function to get the DenseNet121 model
    Returns:
        seq_model: Sequential model with the DenseNet121 model
    '''
    # get the model
    model = tf.keras.applications.densenet.DenseNet121(input_shape=(224, 224, 3),
                                                       include_top=False,
                                                       pooling='avg')
    # add the preprocessing layer
    seq_model = tf.keras.models.Sequential([tf.keras.layers.Lambda(preprocessing_fn_densenet121_224,
                                                                   input_shape=(
                                                                       None, None, 3),
                                                                   dtype=tf.uint8)])
    # add the model
    seq_model.add(model)
    # return the model
    return seq_model


def get_args():
    parser = argparse.ArgumentParser(
        description='Build daatbase for patch data')
    parser.add_argument("--exp_name", type=str, default='kather100k',
                        help="Patch data name for the experiment")
    parser.add_argument("--patch_label_file", type=str,
                        help="The csv file that contain patch name and its label")
    parser.add_argument("--patch_data_path", type=str,
                        help="Path to the folder that contains all patches")
    parser.add_argument("--codebook_semantic", type=str, default="./checkpoints/codebook_semantic.pt",
                        help="Path to semantic codebook")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints//model_9.pt",
                        help="Path to VQ-VAE checkpoint")
    parser.add_argument("--HERE_checkpoint", type=str, default="",
                        help="Path to HERE step2 checkpoint")
    parser.add_argument("--network", type=str, default='kimianet',
                        help="kimianet or densenet121")
    parser.add_argument("--save_filename", type=str, default="",
                        help="Path to save")
    parser.add_argument("--method_name", type=str, default="",
                        help="method name")
    parser.add_argument("--image_size", type=int, default=1000,
                        help="image_size")
    parser.add_argument("--do_hash_evaluation", type=int, default=0,
                        help="do hash evaluation on the given features")
    parser.add_argument("--action", type=str, default="",
                        help="action")
    parser.add_argument("--backbone", type=str, default="",
                        help="action")
    return parser.parse_args()



def extract_feats_Yottixel1(args):
    # get the model
    if args.network == 'kimianet':
        model = get_kimianet_model(network_input_patch_width=1000)
    elif args.network == 'kimianet_imagenet':
        model = get_kimianet_model(
            network_input_patch_width=1000, weights_address='imagenet')
    elif args.network == 'densenet121':
        if args.image_size == 224:
            model = get_dn121_model_224()
        else:
            model = get_dn121_model()
    else:
        raise ValueError('Network not supported')

    patch_label_file = pd.read_csv(args.patch_label_file)

    X = []
    Y = []
    patch_names = []
    t_enc_start = time.time()
    for idx in tqdm(range(len(patch_label_file))):
        patch_name = patch_label_file.loc[idx, 'Patch Names']
        label = patch_label_file.loc[idx, 'label']
        if args.exp_name == 'kather100k':
            patch = openslide.open_slide(os.path.join(args.patch_data_path, patch_name))
            patch_rescaled = patch.read_region((0, 0), 0, (224, 224)).convert('RGB')
        else:
            patch = Image.open(os.path.join(args.patch_data_path, patch_name))
            patch_rescaled = patch.convert('RGB')
        final_feat = model.predict(np.array(patch_rescaled)[None, :])

        X.append(final_feat)
        Y.append(label)
        patch_names.append(patch_name.split(".")[0])

    feat_extract_time = time.time() - t_enc_start

    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_Yottixel(args):
    # get the model
    if args.network == 'kimianet':
        model = get_kimianet_model(network_input_patch_width=1000)
    elif args.network == 'kimianet_imagenet':
        model = get_kimianet_model(
            network_input_patch_width=1000, weights_address='imagenet')
    elif args.network == 'densenet121':
        if args.image_size == 224:
            model = get_dn121_model_224()
        else:
            model = get_dn121_model()
    else:
        raise ValueError('Network not supported')


    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor()
        ]
    )

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values.permute([0,2,3,1]), labels

    kwargs = {'num_workers': 8, 'pin_memory': False, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)

    # pdb.set_trace()
    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.cpu().numpy() 
            features = model.predict(batch_images)
            X.append(features)
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)

    if False:
        patch_label_file = pd.read_csv(args.patch_label_file)

        X = []
        Y = []
        patch_names = []
        t_enc_start = time.time()
        for idx in tqdm(range(len(patch_label_file))):
            patch_name = patch_label_file.loc[idx, 'Patch Names']
            label = patch_label_file.loc[idx, 'label']
            if args.exp_name == 'kather100k':
                patch = openslide.open_slide(os.path.join(args.patch_data_path, patch_name))
                patch_rescaled = patch.read_region((0, 0), 0, (224, 224)).convert('RGB')
            else:
                patch = Image.open(os.path.join(args.patch_data_path, patch_name))
                patch_rescaled = patch.convert('RGB')
            final_feat = model.predict(np.array(patch_rescaled)[None, :])

            X.append(final_feat)
            Y.append(label)
            patch_names.append(patch_name.split(".")[0])

        feat_extract_time = time.time() - t_enc_start

        with open(args.save_filename, 'wb') as fp:
            pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                        'feat_extract_time': feat_extract_time}, fp)



def extract_feats_RetCCL(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load('./backbones/RetCCL_best_ckpt.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path, image_resize=256)

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)
    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            features = model(batch_images)
            X.append(features.detach().cpu().numpy())
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_HERE_new(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    feature_tensors = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook

    model_name = args.method_name.replace('HERE_', '')
    model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None
    print('HF_MODELS_DICT:', HF_MODELS_DICT)

    # pdb.set_trace()

    config = None
    transform = None
    image_processor = None 
    if 'mobilenetv3' in model_name:
        feature_extractor = timm.create_model('mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.flatten.register_forward_hook(get_activation('after_flatten'))
    elif model_name == 'ProvGigaPath':
        feature_extractor = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif model_name == 'CONCH':
        from conch.open_clip_custom import create_model_from_pretrained
        feature_extractor, image_processor = create_model_from_pretrained('conch_ViT-B-16','./backbones/CONCH_weights_pytorch_model.bin')
    elif model_name == 'UNI':
        feature_extractor = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        feature_extractor.load_state_dict(torch.load("./backbones/UNI_pytorch_model.bin", map_location="cpu"), strict=True)
        transform = create_transform(**resolve_data_config(feature_extractor.pretrained_cfg, model=feature_extractor))
    else:
        print('model_params: ', model_params)
        feature_extractor = globals()[model_params[0]].from_pretrained(model_params[1])
        if 'PLIP' in model_name or 'CLIP' in model_name:
            image_processor = CLIPProcessor.from_pretrained(model_params[1])
        else:
            image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    feature_extractor.to(device)
    feature_extractor.eval()

    # pdb.set_trace()

    # encoder + attention except final
    if args.HERE_checkpoint != "":
        args_filepath = os.path.join(os.path.dirname(args.HERE_checkpoint), 'args.txt')
        state_dict = torch.load(args.HERE_checkpoint, map_location='cpu') # CLIP-alike backbone
    else:
        state_dict = torch.load('snapshot_22.pt', map_location='cpu')  # PanCancer, mobilenetv3 backbone
        with open('args.pkl', 'rb') as fp:
            argsdata = pickle.load(fp)

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn(examples):
        pixel_values = image_processor(images=[example[0] for example in examples], return_tensors='pt')
        labels = np.vstack([example[1] for example in examples])
        return pixel_values['pixel_values'], labels

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    def collate_fn_CONCH(examples):
        pixel_values = torch.stack([image_processor(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    batch_size = 16 if 'mobilenetv3' in model_name else 64
    kwargs = {'num_workers': 4 if 'mobilenetv3' in model_name else 4, 'pin_memory': True, 'shuffle': False, 'batch_size': batch_size}
    if transform is not None:
        dataloader = DataLoader(dataset=dataset, **kwargs, collate_fn=collate_fn2)
    else:
        if 'CONCH' in model_name:
            dataloader = DataLoader(dataset=dataset, **kwargs, collate_fn=collate_fn_CONCH)
        else:
            dataloader = DataLoader(dataset=dataset, **kwargs, collate_fn=collate_fn)

    attention_net_W = state_dict['MODEL_STATE']['attention_net.0.weight'].T.to(device)
    attention_net_b = state_dict['MODEL_STATE']['attention_net.0.bias'].to(device)
    print(attention_net_W.shape, attention_net_b.shape)
    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device, non_blocking=True)
            if model_name == 'mobilenetv3':
                _ = feature_extractor(batch_images)
                batch_feats = feature_tensors.get('after_flatten_feat')
            elif model_name == 'ProvGigaPath':
                batch_feats = feature_extractor(batch_images).detach()
            elif model_name == 'CONCH':
                batch_feats = feature_extractor.encode_image(batch_images, proj_contrast=False, normalize=False).detach()
            elif model_name == 'UNI':
                batch_feats = feature_extractor(batch_images).detach()
            else: # CLIP, PLIP
                if transform is not None:
                    batch_feats = feature_extractor.encode_image(batch_images).detach()
                else:
                    batch_feats = feature_extractor.get_image_features(batch_images).detach()

            features1 = batch_feats.float() @ attention_net_W + attention_net_b
            X.append(features1.detach().cpu().numpy())
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_MobileNetV3(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    feature_tensors = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook


    model_name = 'mobilenetv3'
    model_params = ['timm_mobilenetv3', '', 'flatten']
    if model_name == 'mobilenetv3':
        feature_extractor = timm.create_model(
            'mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.flatten.register_forward_hook(
            get_activation('after_flatten'))
        feature_extractor.global_pool.register_forward_hook(
            get_activation('after_global_pool'))
        image_processor = None
    else:
        feature_extractor = globals(
        )[model_params[0]].from_pretrained(model_params[1])
        config = None
        transform = None
        image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels


    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)

    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            _ = feature_extractor(batch_images)
            batch_feats = feature_tensors.get('after_flatten_feat')
            X.append(batch_feats.detach().cpu().numpy())
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_DenseNet121(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    feature_tensors = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()

        return hook

    # with open('args.pkl', 'rb') as fp:
    #     argsdata = pickle.load(fp)

    model_name = 'densenet121'
    model_params = ['timm_mobilenetv3', '', 'flatten']
    if model_name == 'mobilenetv3':
        feature_extractor = timm.create_model(
            'mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.flatten.register_forward_hook(
            get_activation('after_flatten'))
        image_processor = None
    elif model_name == 'densenet121':
        feature_extractor = timm.create_model('densenet121', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.global_pool.register_forward_hook(
            get_activation('after_flatten'))
        image_processor = None
    else:
        feature_extractor = globals(
        )[model_params[0]].from_pretrained(model_params[1])
        config = None
        transform = None
        image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()


    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels


    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)

    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            _ = feature_extractor(batch_images)
            batch_feats = feature_tensors.get('after_flatten_feat')
            X.append(batch_feats.detach().cpu().numpy())
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_PLIP_RetrainedV5(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    config = None
    transform = None
    image_processor = None
    model_name = args.method_name
    model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None
    if model_name == 'mobilenetv3':
        feature_extractor = timm.create_model(
            'mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.flatten.register_forward_hook(
            get_activation('after_flatten')) 
    elif model_name == 'densenet121':
        feature_extractor = timm.create_model('densenet121', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.global_pool.register_forward_hook(
            get_activation('after_flatten')) 
    
    else:
        feature_extractor = globals()[model_params[0]].from_pretrained(model_params[1]) 
        image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn(examples):
        pixel_values = image_processor(images=[example[0] for example in examples], return_tensors='pt')
        labels = np.vstack([example[1] for example in examples])
        return pixel_values['pixel_values'], labels

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    if transform is not None:
        dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn)

    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            if transform is not None:
                batch_feats = feature_extractor.encode_image(batch_images).detach().cpu().numpy()
            else:
                batch_feats = feature_extractor.get_image_features(batch_images).detach().cpu().numpy()
            X.append(batch_feats)
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_ProvGigaPath(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    config = None
    transform = None
    image_processor = None
    # model_name = args.method_name
    # model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None

    feature_extractor = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn(examples):
        pixel_values = image_processor(images=[example[0] for example in examples], return_tensors='pt')
        labels = np.vstack([example[1] for example in examples])
        return pixel_values['pixel_values'], labels

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)

    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            batch_feats = feature_extractor(batch_images).detach().cpu().numpy()
            X.append(batch_feats)
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)



def extract_feats_UNI(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    config = None
    transform = None
    image_processor = None
    # model_name = args.method_name
    # model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None

    feature_extractor = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
    feature_extractor.load_state_dict(torch.load("./UNI_pytorch_model.bin", map_location="cpu"), strict=True)
    transform = create_transform(**resolve_data_config(feature_extractor.pretrained_cfg, model=feature_extractor))
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn(examples):
        pixel_values = image_processor(images=[example[0] for example in examples], return_tensors='pt')
        labels = np.vstack([example[1] for example in examples])
        return pixel_values['pixel_values'], labels

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)

    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            batch_feats = feature_extractor(batch_images).detach().cpu().numpy()
            X.append(batch_feats)
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)




def extract_feats_CONCH(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    config = None
    transform = None
    image_processor = None
    from conch.open_clip_custom import create_model_from_pretrained
    feature_extractor, image_processor = create_model_from_pretrained('conch_ViT-B-16','./CONCH_weights_pytorch_model.bin')
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn_CONCH(examples):
        pixel_values = torch.stack([image_processor(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels

    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn_CONCH)
    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            # batch_feats = feature_extractor(batch_images).detach().cpu().numpy()
            batch_feats = feature_extractor.encode_image(batch_images, proj_contrast=False, normalize=False).detach().cpu().numpy()
            X.append(batch_feats)
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)


def extract_feats_HIPT(args):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    feature_tensors = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook

    model_name = 'HIPT'
    model_params = ['CLIPModel', 'vinid/plip', '']
    if model_name == 'mobilenetv3':
        feature_extractor = timm.create_model(
            'mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.flatten.register_forward_hook(
            get_activation('after_flatten'))
        image_processor = None
    elif model_name == 'densenet121':
        feature_extractor = timm.create_model('densenet121', pretrained=True)
        config = resolve_data_config({}, model=feature_extractor)
        transform = create_transform(**config)
        feature_extractor.global_pool.register_forward_hook(
            get_activation('after_flatten'))
        image_processor = None
    elif model_name == 'PLIP':
        feature_extractor = CLIPModel.from_pretrained('vinid/plip')
        config = None
        transform = None
        image_processor = CLIPProcessor.from_pretrained("vinid/plip")
    elif model_name == 'HIPT':
        feature_extractor = vits.__dict__[
            'vit_small'](patch_size=16)  # 256x256
        state_dict = torch.load('./backbones/HIPT_vit256_small_dino.pth', map_location="cpu")
        state_dict = state_dict['teacher']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k,
                      v in state_dict.items()}
        msg = feature_extractor.load_state_dict(state_dict, strict=False)
        config = None
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image_processor = None
    else:
        feature_extractor = globals()[model_params[0]].from_pretrained(model_params[1])
        config = None
        transform = None
        image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()


    time.sleep(np.random.randint(1, 5))
    dataset = PatchDataset(args.patch_label_file, args.patch_data_path)

    def collate_fn2(examples):
        pixel_values = torch.stack([transform(example[0]) for example in examples])
        labels = np.vstack([example[1] for example in examples])
        return pixel_values, labels


    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': False}
    dataloader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)

    X = []
    t_enc_start = time.time()
    with torch.no_grad():
        for batch_index, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
            batch_images = batch_images.to(device)
            batch_feats = feature_extractor(batch_images).detach().cpu().numpy()
            X.append(batch_feats)
    
    feat_extract_time = time.time() - t_enc_start
    Y = dataset.labels
    patch_names = [os.path.basename(patch_name).split(".")[0] for patch_name in dataset.patch_names]
    with open(args.save_filename, 'wb') as fp:
        pickle.dump({'X': X, 'Y': Y, 'patch_names': patch_names,
                    'feat_extract_time': feat_extract_time}, fp)





def get_results_v4(args):
    save_filename = args.save_filename.replace('.pkl', '_results1.pkl')

    if 'bcss' in args.exp_name:
        bcss_root = '/data/zhongz2/0_Public-data-Amgad2019_0.25MPP/'
        df = pd.read_csv(os.path.join(
            bcss_root, 'meta/gtruth_codes.tsv'), sep='\t')
        labels_dict = {int(label): label_name for label_name, label in zip(
            df['label'].values, df['GT_code'].values)}
        palette_filename = os.path.join(bcss_root, 'palette.npy')
        palette = np.load(palette_filename)
    elif 'PanNuke' in args.exp_name:
        labels_dict = {0: 'Neoplastic cells', 1: 'Inflammatory',
                       2: 'Connective/Soft tissue cells', 3: 'Dead Cells', 4: 'Epithelial', 5: 'Background'}
    elif 'kather' in args.exp_name:
        labels_dict = {'ADI': 0,
                       'BACK': 1,
                       'DEB': 2,
                       'LYM': 3,
                       'MUC': 4,
                       'MUS': 5,
                       'NORM': 6,
                       'STR': 7,
                       'TUM': 8}
        labels_dict = {int(v):k for k,v in labels_dict.items()}
    elif 'NuCLS' in args.exp_name:
        labels_dict = {0: 'AMBIGUOUS',
                1: 'lymphocyte',
                2: 'macrophage',
                3: 'nonTILnonMQ_stromal',
                4: 'other_nucleus',
                5: 'plasma_cell',
                6: 'tumor_nonMitotic'}
    else:
        raise ValueError("wrong labels_dict")

    with open(args.save_filename, 'rb') as fp:
        data = pickle.load(fp)

    if isinstance(data['X'][0], np.ndarray):
        X = np.concatenate(data['X'], axis=0)
    else:
        X = torch.concat(data['X'], axis=0).cpu().numpy()
    
    Y = np.stack(data['Y'])

    patch_names = data['patch_names']
    topk_MV = 5
    all_results = {}
    if 'kather' in args.exp_name:
        print('kather, using random100 searching')
        newdf = pd.read_csv('./kather100k_patch_label_file_random100.csv', index_col=0)
        indices = newdf['Unnamed: 0'].values

        distances = pairwise_distances(X[indices, :], X)
        print(distances.shape)
        # for ind in range(len(Y)):
        for iii, ind in enumerate(indices):
            # D, I = index.search(feats1[ind][None, :], k=10)
            tempdist = distances[iii, :]
            # tempdist = pairwise_distances(X[ind].reshape(1, -1), X).reshape(-1)
            inds = np.argsort(tempdist)
            I = inds[:10]
            query_label = Y[ind]
            label_name = labels_dict[query_label]
            query_prefix = patch_names[ind]

            results = []
            for ind1 in I:
                if ind1 == ind:
                    continue
                label = Y[ind1]
                label_name = labels_dict[label]
                prefix = patch_names[ind1]
                results.append((tempdist[ind1], label, prefix))
                if len(results) == topk_MV:
                    break
            all_results[query_prefix] = {
                'results': results, 'label_query': query_label}

    else: 
        num_parts = int(np.ceil(len(Y) / 2000))
        print(len(Y))
        print(num_parts)
        for parti in range(num_parts):
            start = parti * 2000
            end = min(len(Y), (parti+1)*2000)
            distances = pairwise_distances(X[start:end, :], X)
            for ind in range(start, end): 
                # tempdist = pairwise_distances(X[ind].reshape(1, -1), X).reshape(-1)
                tempdist = distances[ind-start, :]
                inds = np.argsort(tempdist)
                I = inds[:10]
                query_label = Y[ind]
                label_name = labels_dict[query_label]
                query_prefix = patch_names[ind]

                results = []
                for ind1 in I:
                    if ind1 == ind:
                        continue
                    label = Y[ind1]
                    label_name = labels_dict[label]
                    prefix = patch_names[ind1]
                    results.append((tempdist[ind1], label, prefix))
                    if len(results) == topk_MV:
                        break
                all_results[query_prefix] = {
                    'results': results, 'label_query': query_label}

    with open(save_filename, 'wb') as handle:
        pickle.dump(all_results, handle)

    results = all_results
    total_slide = defaultdict(int)
    for v in results.values():
        total_slide[v['label_query']] += 1

    metric_dict = {k: {'Acc': 0, 'Percision': 0, 'total_patch': 0}
                   for k in total_slide.keys()}
    topk_MV = 5
    ret_dict = defaultdict(list)
    t_start = time.time()
    for evlb in total_slide.keys():

        # Evaluating the result diagnosis by diagnoiss
        corr = 0
        percision = 0
        avg_percision = 0
        count = 0
        for patch in results.keys():
            test_patch_result = results[patch]['results']
            label_query = results[patch]['label_query']
            if label_query != evlb:
                continue
            else:
                # Process to calculate the final ret slide
                ret_final = [r[1] for r in test_patch_result[0:topk_MV]]
                ap_at_k = 0
                corr_index = []
                for lb in range(len(ret_final)):
                    if ret_final[lb] == evlb:
                        corr_index.append(lb)
                if len(corr_index) == 0:
                    avg_percision += ap_at_k
                else:
                    for i_corr in corr_index:
                        ap_at_idx_tmp = 0
                        for j in range(i_corr + 1):
                            if ret_final[j] == evlb:
                                ap_at_idx_tmp += 1
                        ap_at_idx_tmp /= (i_corr + 1)
                        ap_at_k += ap_at_idx_tmp
                    ap_at_k /= 5
                    avg_percision += ap_at_k
                if len(ret_final) != 0:
                    hit_label = Counter(ret_final).most_common(1)[0][0]
                else:
                    hit_label = 'NA'

                if hit_label == label_query:
                    if len(ret_final) == topk_MV:
                        corr += 1
                    elif len(ret_final) < topk_MV and\
                            Counter(ret_final).most_common(1)[0][1] >= math.ceil(topk_MV):
                        corr += 1
                else:
                    pass
                count += 1
        metric_dict[evlb]['Acc'] = corr / count
        metric_dict[evlb]['Percision'] = avg_percision / count
        metric_dict[evlb]['total_patch'] = count
    print(time.time() - t_start)
    print(metric_dict)

    metric_dict = dict(sorted(metric_dict.items()))
    df = pd.DataFrame({labels_dict[k]: v for k, v in metric_dict.items()})
    df.to_csv(save_filename.replace('.pkl', '.csv'))



def build_faiss_indexes(feats, Y):

    ITQ_Dims = [32, 64, 128]
    Ms = [8, 16, 32]
    nlists = [128, 256]
    faiss_types = [('IndexFlatIP', None), ('IndexFlatL2', None)]
    faiss_types.extend(
        [(f'IndexBinaryFlat_ITQ{dd}_LSH', dd) for dd in ITQ_Dims])
    for m in Ms:
        for nlist in nlists:
            faiss_types.append(
                (f'IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8', m, nlist))

    d = feats.shape[1]
    indexes = {}
    index_times = {}
    for params in faiss_types:
        t_index_start = time.time()
        faiss_type = params[0]

        quantizer = None
        binarizer = None

        if 'ITQ' in faiss_type:
            binarizer = faiss.index_factory(d, "ITQ{},LSH".format(params[1]))

        if 'HNSW' in faiss_type:
            quantizer = faiss.IndexHNSWFlat(d, params[1])

        if binarizer is not None:
            binarizer.train(feats)

        if faiss_type == 'IndexFlatL2':
            index = faiss.IndexFlatL2(d)
        elif faiss_type == 'IndexFlatIP':
            index = faiss.IndexFlatIP(d)
        elif 'IndexBinaryFlat_ITQ' in faiss_type:
            index = faiss.IndexBinaryFlat(params[1])
        elif 'HNSW' in faiss_type:
            index = faiss.IndexIVFPQ(quantizer, d, params[2], params[1], 8)
            index.train(feats)
        else:
            print('wrong faiss type')
            sys.exit(0)

        if faiss_type == 'IndexBinaryFlat':
            # feats is [-1, 1]
            # [-1, 1] --> [0, 256]
            feats1 = (feats + 1.) * 128
            feats1 = np.clip(np.round(feats1), 0, 256).astype(np.uint8)

        if 'Binary' in faiss_type:
            feats1 = binarizer.sa_encode(feats)
        else:
            feats1 = np.copy(feats)

        index.add(feats1)
        
        index_times[faiss_type] = time.time() - t_index_start
        indexes[faiss_type] = {'binarizer': binarizer, 'quantizer': quantizer, 'index': index}

    return indexes, index_times

def get_results_v5_hash_evaluation(args):

    if 'bcss' in args.exp_name:
        bcss_root = '/data/zhongz2/0_Public-data-Amgad2019_0.25MPP/'
        df = pd.read_csv(os.path.join(
            bcss_root, 'meta/gtruth_codes.tsv'), sep='\t')
        labels_dict = {int(label): label_name for label_name, label in zip(
            df['label'].values, df['GT_code'].values)}
        palette_filename = os.path.join(bcss_root, 'palette.npy')
        palette = np.load(palette_filename)
    elif 'PanNuke' in args.exp_name:
        labels_dict = {0: 'Neoplastic cells', 1: 'Inflammatory',
                       2: 'Connective/Soft tissue cells', 3: 'Dead Cells', 4: 'Epithelial', 5: 'Background'}
    elif 'kather' in args.exp_name:
        labels_dict = {'ADI': 0,
                       'BACK': 1,
                       'DEB': 2,
                       'LYM': 3,
                       'MUC': 4,
                       'MUS': 5,
                       'NORM': 6,
                       'STR': 7,
                       'TUM': 8}
        labels_dict = {int(v):k for k,v in labels_dict.items()}
    elif 'NuCLS' in args.exp_name:
        labels_dict = {0: 'AMBIGUOUS',
                1: 'lymphocyte',
                2: 'macrophage',
                3: 'nonTILnonMQ_stromal',
                4: 'other_nucleus',
                5: 'plasma_cell',
                6: 'tumor_nonMitotic'}
    else:
        raise ValueError("wrong labels_dict")

    with open(args.save_filename, 'rb') as fp:
        data = pickle.load(fp)

    if isinstance(data['X'][0], np.ndarray):
        X = np.concatenate(data['X'], axis=0)
    else:
        X = torch.concat(data['X'], axis=0).cpu().numpy()
    
    feats = X / np.linalg.norm(X, axis=1)[:, None]
    Y = np.stack(data['Y'])

    faiss_indexes, faiss_index_times = build_faiss_indexes(feats, Y)

    if True:
        print('save faiss indexes ...')
        prefix = 'HERE_'
        project_name = args.exp_name
        faiss_bins_dir = os.path.join(os.path.dirname(args.save_filename), 'faiss_bins')
        os.makedirs(faiss_bins_dir, exist_ok=True)
        backbone = args.backbone
        for faiss_type, index in faiss_indexes.items():
            save_filename = f'{faiss_bins_dir}/all_data_feat_before_attention_feat_faiss_{faiss_type}_{project_name}_{prefix}{backbone}.bin'
            if os.path.exists(save_filename):
                continue
            if 'Binary' in faiss_type:
                with open(save_filename, 'wb') as fp:
                    pickle.dump({'binarizer': index['binarizer'],
                                 'index': faiss.serialize_index_binary(index['index'])}, fp)
            else:
                faiss.write_index(index['index'], save_filename)

    patch_names = data['patch_names']
    topk_MV = 5

    search_times = {}
    for faiss_type, faiss_index in faiss_indexes.items():
        save_filename = args.save_filename.replace('.pkl', f'_binary_{faiss_type}_results1.pkl')
        # if os.path.exists(save_filename.replace('.pkl', '.csv')):
        #     return
        all_results = {}

        if 'kather' in args.exp_name:
            print('kather, using random100 searching')
            newdf = pd.read_csv('./kather100k_patch_label_file_random100.csv', index_col=0)
            indices = newdf['Unnamed: 0'].values
        else:
            indices = np.arange(len(Y))

        search_time = 0
        for iii, ind in enumerate(indices):
            t_search_start = time.time()
            query_embedding = feats[ind].reshape(1, -1)
            query_embedding_binary = None
            if 'Binary' in faiss_type and 'ITQ' in faiss_type:
                query_embedding_binary = faiss_index['binarizer'].sa_encode(query_embedding)
            if 'Binary' in faiss_type:
                tempdist, I = faiss_index['index'].search(query_embedding_binary, topk_MV*2)
            elif 'HNSW' in faiss_type:
                tempdist, I = faiss_index['index'].search(query_embedding, topk_MV*2)
            elif faiss_type == 'IndexFlatIP' or faiss_type == 'IndexFlatL2':
                tempdist, I = faiss_index['index'].search(query_embedding, topk_MV*2)
            else:
                raise ValueError("error")
            search_time += time.time() - t_search_start

            tempdist, I = tempdist.flatten(), I.flatten()
            query_label = Y[ind]
            label_name = labels_dict[query_label]
            query_prefix = patch_names[ind]

            results = []
            for jjj, ind1 in enumerate(I):
                if ind1 == ind:
                    continue
                label = Y[ind1]
                label_name = labels_dict[label]
                prefix = patch_names[ind1]
                results.append((tempdist[jjj], label, prefix))
                if len(results) == topk_MV:
                    break
            all_results[query_prefix] = {
                'results': results, 'label_query': query_label}

        search_times[faiss_type] = search_time / len(indices) # average search time

        with open(save_filename, 'wb') as handle:
            pickle.dump(all_results, handle)

        results = all_results
        total_slide = defaultdict(int)
        for v in results.values():
            total_slide[v['label_query']] += 1

        metric_dict = {k: {'Acc': 0, 'Percision': 0, 'total_patch': 0}
                    for k in total_slide.keys()}
        topk_MV = 5
        ret_dict = defaultdict(list)
        t_start = time.time()
        for evlb in total_slide.keys():

            # Evaluating the result diagnosis by diagnoiss
            corr = 0
            percision = 0
            avg_percision = 0
            count = 0
            for patch in results.keys():
                test_patch_result = results[patch]['results']
                label_query = results[patch]['label_query']
                if label_query != evlb:
                    continue
                else:
                    # Process to calculate the final ret slide
                    ret_final = [r[1] for r in test_patch_result[0:topk_MV]]
                    ap_at_k = 0
                    corr_index = []
                    for lb in range(len(ret_final)):
                        if ret_final[lb] == evlb:
                            corr_index.append(lb)
                    if len(corr_index) == 0:
                        avg_percision += ap_at_k
                    else:
                        for i_corr in corr_index:
                            ap_at_idx_tmp = 0
                            for j in range(i_corr + 1):
                                if ret_final[j] == evlb:
                                    ap_at_idx_tmp += 1
                            ap_at_idx_tmp /= (i_corr + 1)
                            ap_at_k += ap_at_idx_tmp
                        ap_at_k /= 5
                        avg_percision += ap_at_k
                    if len(ret_final) != 0:
                        hit_label = Counter(ret_final).most_common(1)[0][0]
                    else:
                        hit_label = 'NA'

                    if hit_label == label_query:
                        if len(ret_final) == topk_MV:
                            corr += 1
                        elif len(ret_final) < topk_MV and\
                                Counter(ret_final).most_common(1)[0][1] >= math.ceil(topk_MV):
                            corr += 1
                    else:
                        pass
                    count += 1
            metric_dict[evlb]['Acc'] = corr / count
            metric_dict[evlb]['Percision'] = avg_percision / count
            metric_dict[evlb]['total_patch'] = count
        print(time.time() - t_start)
        print(metric_dict)

        metric_dict = dict(sorted(metric_dict.items()))
        df = pd.DataFrame({labels_dict[k]: v for k, v in metric_dict.items()})
        df.to_csv(save_filename.replace('.pkl', '.csv'))

    with open(args.save_filename.replace('.pkl', '_binary_search_times.pkl'), 'wb') as fp:
        pickle.dump({'search_times': search_times, 'index_times': faiss_index_times}, fp)



# 20240620 TCGA-COMBINED
def get_results_v7_hash_evaluation():
    from types import SimpleNamespace
    import os,pickle,faiss
    import pandas as pd
    import numpy as np
    import time

    args = SimpleNamespace(**{'exp_name': 'bcss_512_0.8', 'save_filename': '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/bcss_512_0.8_HERE_CONCH_0_feats.pkl'})

    if 'bcss' in args.exp_name:
        bcss_root = '/data/zhongz2/0_Public-data-Amgad2019_0.25MPP/'
        df = pd.read_csv(os.path.join(
            bcss_root, 'meta/gtruth_codes.tsv'), sep='\t')
        labels_dict = {int(label): label_name for label_name, label in zip(
            df['label'].values, df['GT_code'].values)}
        palette_filename = os.path.join(bcss_root, 'palette.npy')
        palette = np.load(palette_filename)
    elif 'PanNuke' in args.exp_name:
        labels_dict = {0: 'Neoplastic cells', 1: 'Inflammatory',
                       2: 'Connective/Soft tissue cells', 3: 'Dead Cells', 4: 'Epithelial', 5: 'Background'}
    elif 'kather' in args.exp_name:
        labels_dict = {'ADI': 0,
                       'BACK': 1,
                       'DEB': 2,
                       'LYM': 3,
                       'MUC': 4,
                       'MUS': 5,
                       'NORM': 6,
                       'STR': 7,
                       'TUM': 8}
        labels_dict = {int(v):k for k,v in labels_dict.items()}
    elif 'NuCLS' in args.exp_name:
        labels_dict = {0: 'AMBIGUOUS',
                1: 'lymphocyte',
                2: 'macrophage',
                3: 'nonTILnonMQ_stromal',
                4: 'other_nucleus',
                5: 'plasma_cell',
                6: 'tumor_nonMitotic'}
    else:
        raise ValueError("wrong labels_dict")

    with open(args.save_filename, 'rb') as fp:
        data = pickle.load(fp)

    if isinstance(data['X'][0], np.ndarray):
        X = np.concatenate(data['X'], axis=0)
    else:
        X = torch.concat(data['X'], axis=0).cpu().numpy()
    
    feats = X / np.linalg.norm(X, axis=1)[:, None]
    Y = np.stack(data['Y'])
    print('feats', feats.shape)

    # faiss_indexes, faiss_index_times = build_faiss_indexes(feats, Y)
    faiss_index_times = None
    faiss_bin_dir = '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/faiss_relatedV6/faiss_bins'

    faiss_ITQ_ds = [32, 64, 128]
    faiss_Ms = [8, 16, 32]
    faiss_nlists = [128, 256]
    runs = {'NCIData': ['KenData_20240814'], 'TCGA': ['TCGA-COMBINED']}
    prefix = 'HERE_'
    backbone = 'CONCH'
    for Key, project_names in runs.items():   
        print('begin ', Key, '='*80)     
        print('loading faiss indexes')    
        faiss_indexes = {'faiss_IndexFlatIP': {}, 'faiss_IndexFlatL2': {}}
        for project_name in project_names:  # only ST support IndexFlatIP search
            faiss_indexes['faiss_IndexFlatIP'][project_name] = \
                faiss.read_index(
                    f"{faiss_bin_dir}/all_data_feat_before_attention_feat_faiss_IndexFlatIP_{project_name}_{prefix}{backbone}.bin")
            faiss_indexes['faiss_IndexFlatL2'][project_name] = \
                faiss.read_index(
                    f"{faiss_bin_dir}/all_data_feat_before_attention_feat_faiss_IndexFlatL2_{project_name}_{prefix}{backbone}.bin")
        for dd in faiss_ITQ_ds:
            faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'] = {}
            for project_name in project_names:
                with open(f"{faiss_bin_dir}/all_data_feat_before_attention_feat_faiss_IndexBinaryFlat_ITQ{dd}_LSH_{project_name}_{prefix}{backbone}.bin", 'rb') as fp:
                    faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'][project_name] = pickle.load(
                        fp)
                    faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'][project_name]['index'] = \
                        faiss.deserialize_index_binary(
                            faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'][project_name]['index'])
        for m in faiss_Ms:
            for nlist in faiss_nlists:
                faiss_indexes[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'] = {}
                for project_name in project_names:
                    faiss_indexes[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'][project_name] = \
                        faiss.read_index(
                            f"{faiss_bin_dir}/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8_{project_name}_{prefix}{backbone}.bin")

        patch_names = data['patch_names']
        topk_MV = 5

        print('begin ', Key)
        search_times = {}
        for faiss_type, faiss_index in faiss_indexes.items():
            save_filename = args.save_filename.replace('.pkl', f'_binary_{faiss_type}_results1.pkl')
            # if os.path.exists(save_filename.replace('.pkl', '.csv')):
            #     return
            all_results = {}

            if 'kather' in args.exp_name:
                print('kather, using random100 searching')
                newdf = pd.read_csv('./kather100k_patch_label_file_random100.csv', index_col=0)
                indices = newdf['Unnamed: 0'].values
            else:
                indices = np.arange(len(Y))

            # distances = pairwise_distances(X[indices, :], X)
            # print(distances.shape)
            # for ind in range(len(Y)):
            search_time = 0
            iii = 0
            # for iiiii, project_name in enumerate(project_names):
            #     print(faiss_type, faiss_index[project_name].d)
            for iii, ind in enumerate(indices):
                # # D, I = index.search(feats1[ind][None, :], k=10)
                # tempdist = distances[iii, :]
                # # tempdist = pairwise_distances(X[ind].reshape(1, -1), X).reshape(-1)
                # inds = np.argsort(tempdist)
                # I = inds[:10]
                t_search_start = time.time()
                query_embedding = feats[ind].reshape(1, -1)
                query_embedding_binary = None

                # print('query_embedding', query_embedding)
                # print('query_embedding', query_embedding.shape)

                for iiiii, project_name in enumerate(project_names):
                    if 'Binary' in faiss_type and 'ITQ' in faiss_type:
                        query_embedding_binary = faiss_index[project_name]['binarizer'].sa_encode(query_embedding)
                    if 'Binary' in faiss_type:
                        tempdist, I = faiss_index[project_name]['index'].search(query_embedding_binary, topk_MV*2)
                    elif 'HNSW' in faiss_type:
                        tempdist, I = faiss_index[project_name].search(query_embedding, topk_MV*2)
                    elif 'IndexFlatIP' in faiss_type or 'IndexFlatL2' in faiss_type: 
                        tempdist, I = faiss_index[project_name].search(query_embedding, topk_MV*2)
                    else:
                        raise ValueError("error")
                
                search_time += time.time() - t_search_start

                if iii == 100:
                    break

            search_times[faiss_type] = search_time / iii # len(indices) # average search time

        # get number of rows
        total_rows = {}
        for faiss_type, faiss_index in faiss_indexes.items():
            ntotal = 0
            for iiiii, project_name in enumerate(project_names):
                if 'Binary' in faiss_type:
                    ntotal += faiss_index[project_name]['index'].ntotal
                elif 'HNSW' in faiss_type:
                    ntotal += faiss_index[project_name].ntotal
                elif 'IndexFlatIP' in faiss_type or 'IndexFlatL2' in faiss_type:
                    ntotal += faiss_index[project_name].ntotal
                else:
                    raise ValueError("error")
            total_rows[faiss_type] = ntotal

        del faiss_indexes
        with open(args.save_filename.replace('.pkl', f'_{Key}_binary_search_times.pkl'), 'wb') as fp:
            pickle.dump({'search_times': search_times, 'index_times': faiss_index_times, 'total_rows': total_rows}, fp)


    all_total_rows = {}
    all_sizes = {}
    runs = {'NCIData': ['KenData_20240814'], 'TCGA': ['TCGA-COMBINED'], 'ST': ['ST'], 'Kather100K': ['kather100k']}
    for Key, project_names in runs.items():
        if Key in all_total_rows:
            print(f'{Key} existed, skip')
            continue
        if Key == 'Kather100K':
            faiss_bin_dir1 = '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/faiss_bins'
        else:
            faiss_bin_dir1 = faiss_bin_dir
        total_size = {}
        if 'faiss_indexes' in locals() or 'faiss_indexes' in globals():
            del faiss_indexes
        faiss_indexes = {'faiss_IndexFlatIP': {}, 'faiss_IndexFlatL2': {}}
        total_size['faiss_IndexFlatIP'] = 0
        total_size['faiss_IndexFlatL2'] = 0
        for project_name in project_names:  # only ST support IndexFlatIP search
            faiss_indexes['faiss_IndexFlatIP'][project_name] = \
                faiss.read_index(
                    f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexFlatIP_{project_name}_{prefix}{backbone}.bin")
            total_size['faiss_IndexFlatIP'] += os.path.getsize(f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexFlatIP_{project_name}_{prefix}{backbone}.bin")

            faiss_indexes['faiss_IndexFlatL2'][project_name] = \
                faiss.read_index(
                    f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexFlatL2_{project_name}_{prefix}{backbone}.bin")
            total_size['faiss_IndexFlatL2'] += os.path.getsize(f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexFlatL2_{project_name}_{prefix}{backbone}.bin")
        for dd in faiss_ITQ_ds:
            faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'] = {}
            total_size[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'] = 0
            for project_name in project_names:
                total_size[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'] += os.path.getsize(f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexBinaryFlat_ITQ{dd}_LSH_{project_name}_{prefix}{backbone}.bin")
                with open(f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexBinaryFlat_ITQ{dd}_LSH_{project_name}_{prefix}{backbone}.bin", 'rb') as fp:
                    faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'][project_name] = pickle.load(
                        fp)
                    faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'][project_name]['index'] = \
                        faiss.deserialize_index_binary(
                            faiss_indexes[f'faiss_IndexBinaryFlat_ITQ{dd}_LSH'][project_name]['index'])
        for m in faiss_Ms:
            for nlist in faiss_nlists:
                faiss_indexes[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'] = {}
                total_size[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'] = 0
                for project_name in project_names:
                    total_size[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'] += os.path.getsize(f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8_{project_name}_{prefix}{backbone}.bin")
                    faiss_indexes[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'][project_name] = \
                        faiss.read_index(
                            f"{faiss_bin_dir1}/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8_{project_name}_{prefix}{backbone}.bin")
        all_sizes[Key] = total_size
        # get number of rows
        total_rows = {}
        for faiss_type, faiss_index in faiss_indexes.items():
            ntotal = 0
            size = 0
            for iiiii, project_name in enumerate(project_names):
                if 'Binary' in faiss_type:
                    ntotal += faiss_index[project_name]['index'].ntotal
                elif 'HNSW' in faiss_type:
                    ntotal += faiss_index[project_name].ntotal
                elif 'IndexFlatIP' in faiss_type or 'IndexFlatL2' in faiss_type:
                    ntotal += faiss_index[project_name].ntotal
                else:
                    raise ValueError("error")
            total_rows[faiss_type] = ntotal
        
        all_total_rows[Key] = total_rows

    with open('/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/faiss_bins_count_and_size.pkl', 'wb') as fp:
        pickle.dump({'all_sizes': all_sizes, 'all_total_rows': all_total_rows}, fp)



if __name__ == '__main__':

    args = get_args()
    if args.action == 'faiss_bins_count_and_size':
        get_results_v7_hash_evaluation()
        sys.exit(0)

    save_dir = os.path.dirname(args.save_filename)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(args.save_filename):
        if args.method_name == 'Yottixel':
            args.network = 'kimianet'
            extract_feats_Yottixel(args)   # 1024
        elif args.method_name == 'RetCCL':
            extract_feats_RetCCL(args)   # 1024
        elif args.method_name in ['HERE_mobilenetv3', 'HERE_CLIP', 'HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_UNI']:  # with different backbone
            extract_feats_HERE_new(args)   # 1024
        elif args.method_name == 'MobileNetV3':
            extract_feats_MobileNetV3(args)   # 1280
        elif args.method_name == 'DenseNet121':
            extract_feats_DenseNet121(args)   # 1024
        elif 'PLIP_Retrained' in args.method_name or args.method_name in ['CLIP', 'PLIP']:
            extract_feats_PLIP_RetrainedV5(args)   # 512
        elif 'ProvGigaPath' in args.method_name:
            extract_feats_ProvGigaPath(args)
        elif 'CONCH' in args.method_name:
            extract_feats_CONCH(args)
        elif 'UNI' in args.method_name:
            extract_feats_UNI(args)
        elif args.method_name == 'HIPT':
            extract_feats_HIPT(args)   # 384
    else:
        print('features are existed')

    print('begin processing results ...') 
    save_filename = args.save_filename.replace('.pkl', '_results1.pkl')
    if not os.path.exists(save_filename.replace('.pkl', '.csv')):
        get_results_v4(args) 

    if 'HERE_' == args.method_name[:5] and not os.path.exists(args.save_filename.replace('.pkl', '_binary_search_times.pkl')):
        get_results_v5_hash_evaluation(args)

