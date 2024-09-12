import sys, os, glob, shutil, json
import pdb
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage import exposure
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import time
import h5py
import openslide
from sklearn.preprocessing import MinMaxScaler
import torch
from multiprocessing import Pool
import itertools
from natsort import natsorted
import idr_torch
import argparse
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoImageProcessor, CLIPProcessor
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import ResNetModel, BeitModel, BitModel, ConvNextModel, CvtModel, DeiTModel, \
    DinatModel, DPTModel, EfficientFormerModel, GLPNModel, MobileNetV1Model, ImageGPTModel, \
    LevitModel, MobileNetV1Model, MobileNetV2Model, MobileViTModel, NatModel, PoolFormerModel, \
    SwinModel, Swinv2Model, ViTModel, ViTHybridModel, ViTMAEModel, ViTMSNModel, CLIPModel
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import set_layer_config
from timm.models import is_model, model_entrypoint, load_checkpoint
from urllib.parse import urlsplit
import gc
import clip
import socket
from common import HF_MODELS_DICT
from dataset import PatchDatasetV2
from utils import save_hdf5
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_model_config_from_hf(model_id: str):
    cached_file = './backbones/ProvGigaPath/config.json'

    hf_config = load_cfg_from_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args


def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def parse_model_name(model_name: str):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def create_model():
    model_name = 'hf_hub:prov-gigapath/prov-gigapath'
    model_source, model_name = parse_model_name(model_name)
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    kwargs = {}
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            **kwargs
        )
    load_checkpoint(model, './backbones/ProvGigaPath/pytorch_model.bin')

    return model


def get_args():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default='')  
    parser.add_argument('--data_slide_dir', type=str, default='')  
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default='') 
    parser.add_argument('--feat_dir', type=str, default='') 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--start_idx', type=int, default=-1)
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model_name = args.model_name
    model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None
    print('HF_MODELS_DICT:', HF_MODELS_DICT)

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None or csv_path == "None":
        DX_filenames = sorted(glob.glob(os.path.join(args.data_slide_dir, '*{}*'.format(args.slide_ext))))
        df = pd.DataFrame({'DX_filename': DX_filenames})
    else:
        if 'xlsx' in csv_path: 
            df = pd.read_excel(csv_path, low_memory=False)
        else:
            df = pd.read_csv(csv_path, low_memory=False)

    if 'slide_id' not in df:
        slide_ids = [os.path.basename(f).replace(args.slide_ext, '') for f in df['DX_filename'].values]
        df['slide_id'] = slide_ids
    else:
        df['slide_id'] = df['slide_id'].astype(str)

    print('before', len(df))
    if args.end_idx > args.start_idx > -1:
        if  args.end_idx >= len(df):
            args.end_idx = len(df)
        df = df.iloc[np.arange(args.start_idx, args.end_idx)].reset_index(drop=True)
    print('after', len(df))

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    print('local_rank: {},{} '.format(socket.gethostname(), idr_torch.local_rank))
    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device('cuda')
    local_temp_dir = os.path.join(os.environ['HOME'], model_name, str(idr_torch.rank), str(idr_torch.local_rank))
    if os.path.isdir(local_temp_dir):
        shutil.rmtree(local_temp_dir, ignore_errors=True)
        time.sleep(1)
    os.makedirs(local_temp_dir, exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    existed_prefixes = set([os.path.basename(f).replace('.pt', '') for f in dest_files])
    h5_prefixes = set([os.path.basename(f).replace('.h5', '') for f in glob.glob(os.path.join(args.data_h5_dir, 'patches',  '*.h5'))])

    drop_ids = []
    for ind, f in enumerate(df['DX_filename'].values):
        svs_prefix = os.path.basename(f).replace(args.slide_ext, '')
        if svs_prefix in existed_prefixes or svs_prefix not in h5_prefixes:
            drop_ids.append(ind)
    print('before0', len(df))
    if len(drop_ids) > 0:
        df = df.drop(drop_ids)
    df = df.reset_index(drop=True)
    if len(df) == 0:        
        print('all done')
        sys.exit(0)
    print('before', len(df))

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    print('local rank:', idr_torch.local_rank)
    print('world_size: ', idr_torch.world_size)
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)
    if len(sub_df) == 0:
        print('all done')
        sys.exit(0)
    print(idr_torch.rank, sub_df['DX_filename'].values[0])

    feature_tensors = {}
    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook


    config = None
    transform = None
    image_processor = None 
    if model_name == 'mobilenetv3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        model.flatten.register_forward_hook(get_activation('after_flatten'))
    elif model_name == 'ProvGigaPath':
        # model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model = create_model()  # timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
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
        model, image_processor = create_model_from_pretrained('conch_ViT-B-16','./backbones/CONCH_weights_pytorch_model.bin')
    elif model_name == 'UNI':
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load("./backbones/UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        print('model_params: ', model_params)
        model = globals()[model_params[0]].from_pretrained(model_params[1])
        if 'PLIP' in model_name or 'CLIP' in model_name:
            image_processor = CLIPProcessor.from_pretrained(model_params[1])
        else:
            image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    model = model.to(device)
    model.eval() 

    def collate_fn2(examples): 
        pixel_values = torch.stack([transform(example["pixel_values"]) for example in examples])
        coords = np.vstack([example["coords"] for example in examples])
        return pixel_values, coords

    def collate_fn(examples):
        pixel_values = image_processor(images=[example["pixel_values"] for example in examples], return_tensors='pt')
        coords = np.vstack([example["coords"] for example in examples])
        return pixel_values['pixel_values'], coords

    def collate_fn_CONCH(examples):
        pixel_values = torch.stack([image_processor(example["pixel_values"]) for example in examples])
        coords = np.vstack([example["coords"] for example in examples])
        return pixel_values, coords
        
    time.sleep(3)
    for index, row in sub_df.iterrows():
        slide_file_path = row['DX_filename']
        svs_prefix = os.path.basename(slide_file_path).replace(args.slide_ext, '')
        final_feat_filename = os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt')
        if os.path.exists(final_feat_filename):
            continue

        slide_file_path = os.path.join(args.data_slide_dir, svs_prefix+ args.slide_ext)

        h5_file_path = os.path.join(args.data_h5_dir, 'patches', svs_prefix + '.h5')
        if not os.path.exists(h5_file_path):
            continue

        local_temp_dir1 = os.path.join(local_temp_dir, str(np.random.randint(1e5) + np.random.randint(1e10)))
        os.makedirs(local_temp_dir1, exist_ok=True)

        local_svs_filename = os.path.join(local_temp_dir1, svs_prefix + args.slide_ext)
        os.system(f'cp -RL "{slide_file_path}" "{local_svs_filename}"')
        time.sleep(1)

        h5file = h5py.File(h5_file_path, 'r')
        dset = h5file['coords']
        coords = dset[:]
        patch_level = h5file['coords'].attrs['patch_level']
        patch_size = h5file['coords'].attrs['patch_size']

        # filter the coords
        slide = openslide.open_slide(local_svs_filename)
        time.sleep(1)
        print('extract features')

        dataset = PatchDatasetV2(slide, coords, patch_level, patch_size)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': False}
        if transform is not None:
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_fn2)
        else:
            if model_name == 'CONCH':
                loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_fn_CONCH)
            else:
                loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_fn)

        output_path = os.path.join(local_temp_dir1, svs_prefix + '.h5')
        time_start = time.time()
        mode = 'w'
        for count, (images, coords) in enumerate(loader):
            with torch.no_grad():
                images = images.to(device)

                if model_name == 'mobilenetv3':
                    output = model(images)
                    features = feature_tensors.get('after_flatten_feat').cpu().numpy().reshape((len(coords), -1))
                elif model_name == 'UNI':
                    features = model(images).detach().cpu().numpy().reshape((len(coords), -1))
                elif model_name == 'ProvGigaPath':
                    features = model(images).detach().cpu().numpy().reshape((len(coords), -1))
                elif model_name == 'CONCH':
                    features = model.encode_image(images, proj_contrast=False, normalize=False).detach().cpu().numpy().reshape((len(coords), -1))
                elif model_params is not None and model_params[2] is not None:
                    features = model(images, return_dict=True)
                    features = features.get(model_params[2]).cpu().numpy().reshape((len(coords), -1))
                elif 'PLIP_Retrained' in model_name:
                    features = model.encode_image(images).detach().cpu().numpy().reshape((len(coords), -1))
                else:
                    features = model.get_image_features(images).detach().cpu().numpy().reshape((len(coords), -1)) # CLIP, PLIP

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))
        # del model
        gc.collect() 

        with h5py.File(output_path, "r") as file:
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt'))

        time.sleep(0.5)
        os.system(f'rm -rf "{output_path}"')

        slide.close()
        os.system(f'rm -rf "{local_svs_filename}"')
        time.sleep(1)

        torch.cuda.empty_cache()
        
        if os.path.isdir(local_temp_dir1):
            os.system(f'rm -rf "{local_temp_dir1}"')


    time.sleep(2)
    if os.path.isdir(local_temp_dir):
        os.system(f'rm -rf "{local_temp_dir}"')



if __name__ == '__main__':
    main()









