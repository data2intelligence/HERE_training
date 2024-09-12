import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
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
from torch.utils.data import DataLoader
import multiprocessing
import idr_torch
from statsmodels.stats.multitest import multipletests
from scipy.stats import ranksums, percentileofscore
import scanpy
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tarfile
import io
import argparse
from torchvision.utils import save_image
from torchvision import transforms
from sklearn import cluster, neighbors
from scipy.spatial import distance_matrix
import time
import openslide
import matplotlib.pyplot as plt
import gc
import pdb
import socket
import sys
import os
import glob
import h5py
import json
import pandas as pd
import numpy as np
import torch
from model import AttentionModel
import umap
from utils import get_svs_prefix, _assertLevelDownsamplesV2, new_web_annotation
from common import HF_MODELS_DICT, CLASSIFICATION_DICT
from dataset import PatchDatasetV2
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True


def cohend(d1, d2) -> pd.Series:
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1, axis=0), np.mean(d2, axis=0)
    # return the effect size
    return (u1 - u2) / s


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', default='/data/zhongz2/temp29/debug/debug_results/ngpus2_accum4_backboneProvGigaPath_dropout0.25/analysis/ST', type=str)
    parser.add_argument('--svs_dir', default='/data/zhongz2/ST_256/svs', type=str)
    parser.add_argument('--patches_dir', default='/data/zhongz2/ST_256/patches', type=str)
    parser.add_argument('--image_ext', default='.svs', type=str)
    parser.add_argument('--backbone', default='ProvGigaPath', type=str)
    parser.add_argument('--ckpt_path', default='/data/zhongz2/temp29/debug/results/ngpus2_accum4_backboneProvGigaPath_dropout0.25/split_1/snapshot_39.pt', type=str)
    parser.add_argument('--cluster_feat_name', default='feat_before_attention_feat', type=str)
    parser.add_argument('--csv_filename', default='/data/zhongz2/ST_256/all_20231117.xlsx', type=str)
    parser.add_argument('--cluster_task_name', default='one_patient', type=str)
    # parser.add_argument('--csv_filename', default='/data/zhongz2/ST_256/response_group_sample.xlsx', type=str)
    # parser.add_argument('--cluster_task_name', default='response_groups', type=str)
    parser.add_argument('--cluster_task_index', default=0, type=int)
    parser.add_argument('--num_patches', default=128, type=int)
    parser.add_argument('--only_step1', default='gen_patches', type=str)  # yes, no, gen_patches
    parser.add_argument('--pre_computed_vst', default=False, type=bool)  # yes, no, gen_patches
    return parser.parse_args()

def main():
    args = get_args()
    save_root1 = '{}/{}/{}'.format(
        args.save_root,
        args.backbone,
        args.cluster_feat_name)

    save_root3 = os.path.join(save_root1, 'big_images')
    save_root3_1 = os.path.join(args.save_root, 'big_images')
    save_root4 = os.path.join(save_root1, 'cached_data')
    save_root44 = os.path.join(save_root1, 'cached_data1')
    save_root_vst = os.path.join(save_root1, 'vst_dir')
    save_root_gene_data = os.path.join(save_root1, 'gene_data')
    save_root_gene_map_data = os.path.join(save_root1, 'gene_map')
    save_root_gene_da_dir = os.path.join(save_root1, 'save_root_gene_da_dir')
    os.makedirs(save_root1, exist_ok=True)
    os.makedirs(save_root3, exist_ok=True)
    os.makedirs(save_root3_1, exist_ok=True)
    os.makedirs(save_root4, exist_ok=True)
    os.makedirs(save_root44, exist_ok=True)
    os.makedirs(save_root_vst, exist_ok=True)
    os.makedirs(save_root_gene_data, exist_ok=True)
    os.makedirs(save_root_gene_map_data, exist_ok=True)
    os.makedirs(save_root_gene_da_dir, exist_ok=True)

    if args.csv_filename == 'None':
        raise ValueError('no csv_filename')
    else:
        csv_filename = args.csv_filename

    if 'xlsx' in csv_filename:
        df = pd.read_excel(csv_filename)
    else:
        df = pd.read_csv(csv_filename)

    label_title = args.cluster_task_name

    # only processing these cases with H&E files
    df = df[~df['HEfiles'].isna()] if 'HEfiles' in df.columns else df[~df['DX_filename'].isna()]
    HEfiles = []
    for row_id, row in df.iterrows():
        svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
        svs_filename_list = svs_filenames.split(',')  # 2 slides
        svs_filename_list_new = []
        for j, filename in enumerate(svs_filename_list):
            svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
            svs_prefix = get_svs_prefix(svs_filename)
            h5_filename = os.path.join(args.patches_dir, svs_prefix + '.h5')
            if os.path.exists(svs_filename) and os.path.exists(h5_filename):
                svs_filename_list_new.append(filename)
        HEfiles.append(','.join(svs_filename_list_new))
    df['HEfiles'] = HEfiles
    df = df[df['HEfiles']!=''].reset_index(drop=True)

    file_inds_dict = {}
    file_ind = 0
    patient_inds = []
    patient_ind = 0
    for row_ind, row in df.iterrows():
        svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
        svs_filename_list = svs_filenames.split(',')  # 2 slides
        patient_inds.append(patient_ind)
        for j, filename in enumerate(svs_filename_list):
            svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
            svs_prefix = get_svs_prefix(svs_filename)
            file_inds_dict[svs_prefix] = file_ind
            file_ind += 1
        patient_ind += 1

    if 'PATIENT_ID' in df.columns:
        df = df.astype({'PATIENT_ID': str})

    print(file_inds_dict)
    if 'PATIENT_ID' in df.columns:
        df['patient_ind'] = df['PATIENT_ID'].values.tolist()
    elif 'patient_ind' not in df.columns:
        df['patient_ind'] = patient_inds
    else:
        print('need patient_ind column')
        sys.exit(-1)

    if 'PATIENT_ID' not in df.columns:
        df['PATIENT_ID'] = ['Patient_{}'.format(
            ii) for ii in df['patient_ind'].values.tolist()]

    if args.cluster_task_name == 'one_patient' and 'one_patient' not in df.columns:
        df['one_patient'] = [0 for _ in range(len(df))]

    df = df.reset_index() 
    df_all = df.copy()

    if idr_torch.rank == 0:
        if not os.path.exists(os.path.join(args.save_root, 'patient_list.txt')):
            with open(os.path.join(args.save_root, 'patient_list.txt'), 'w') as fp:
                fp.writelines([name + '\n' for name in df_all['PATIENT_ID'].values.tolist()])
        if not os.path.exists(os.path.join(args.save_root, 'all_data.csv')):
            df_all.to_csv(os.path.join(args.save_root, 'all_data.csv'))

    if args.only_step1 == 'yes':
        dones = []
        for row_ind, row in df.iterrows():
            svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
            svs_filename_list = svs_filenames.split(',')  # 2 slides
            patient_ind = row['patient_ind']
            PATIENT_ID = row['PATIENT_ID']
            label = 0
            done = []
            for j, filename in enumerate(svs_filename_list):
                svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
                svs_prefix = get_svs_prefix(svs_filename)
                save_filename1 = '{}/{}_big_orig.zip'.format(save_root3_1, svs_prefix)
                save_filename = '{}/{}_big_attention_map.zip'.format(save_root3, svs_prefix)
                if os.path.exists(save_filename) and os.path.exists(save_filename1):
                    done.append(True)
                else:
                    done.append(False)
            if np.all(done):
                dones.append(1)
            else:
                dones.append(0)
        df['isDone'] = dones
        df = df[df['isDone']==0].reset_index(drop=True)
        
    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)
    df = sub_df

    if args.only_step1 == 'yes':
        if idr_torch.rank == 0:
            if not os.path.exists(os.path.join(args.save_root, 'patient_list.txt')):
                with open(os.path.join(args.save_root, 'patient_list.txt'), 'w') as fp:
                    fp.writelines([name + '\n' for name in df_all['PATIENT_ID'].values.tolist()])
            if not os.path.exists(os.path.join(args.save_root, 'all_data.csv')):
                df_all.to_csv(os.path.join(args.save_root, 'all_data.csv'))
        print(df)
        print('local_rank: {},{} '.format(socket.gethostname(), idr_torch.local_rank))
        torch.cuda.set_device(idr_torch.local_rank)
        device = torch.device('cuda')

        local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank))
        os.makedirs(local_temp_dir, exist_ok=True)

        model_name = args.backbone  # mobilenetv3, CLIP, PLIP
        model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None

        feature_tensors = {}
        def get_activation(name):
            def hook(model, input, output):
                feature_tensors[name + '_feat'] = output.detach()
            return hook

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
            feature_extractor, image_processor = create_model_from_pretrained('conch_ViT-B-16','./CONCH_weights_pytorch_model.bin')
        else:
            print('model_params: ', model_params)
            feature_extractor = globals()[model_params[0]].from_pretrained(model_params[1])
            if 'PLIP' in model_name or 'CLIP' in model_name:
                image_processor = CLIPProcessor.from_pretrained(model_params[1])
            else:
                image_processor = AutoImageProcessor.from_pretrained(model_params[1])
        feature_extractor.to(device)
        feature_extractor.eval()

        model = AttentionModel(backbone=args.backbone)
        # encoder + attention except final
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['MODEL_STATE'], strict=False)
        model = model.to(device)
        model.eval()

        model.attention_net[0].register_forward_hook(get_activation('feat_before_attention')),
        model.rho[0].register_forward_hook(get_activation('feat_after_attention_0_linear')),
        model.rho[1].register_forward_hook(get_activation('feat_after_attention_1_relu')),
        model.rho[2].register_forward_hook(get_activation('feat_after_attention_2_dropout')),
        print('feature_tensors keys:', feature_tensors.keys())

        def collate_fn(examples):
            pixel_values = image_processor(images=[example['pixel_values'] for example in examples], return_tensors='pt')
            labels = np.vstack([example['coords'] for example in examples])
            return pixel_values['pixel_values'], labels

        def collate_fn2(examples):
            pixel_values = torch.stack([transform(example['pixel_values']) for example in examples])
            labels = np.vstack([example['coords'] for example in examples])
            return pixel_values, labels

        def collate_fn_CONCH(examples):
            pixel_values = torch.stack([image_processor(example["pixel_values"]) for example in examples])
            labels = np.vstack([example["coords"] for example in examples])
            return pixel_values, labels

        job_params = []
        for row_ind, row in df.iterrows():

            svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
            svs_filename_list = svs_filenames.split(',')  # 2 slides
            patient_ind = row['patient_ind']
            PATIENT_ID = row['PATIENT_ID']
            label = 0 

            # for ST data
            if 'coord_filename' in row and 'counts_filename' in row and 'barcode_col_name' in row:
                coord_filename_list = row['coord_filename'].split(',')
                counts_filename_list = row['counts_filename'].split(',')
                barcode_col_name = row['barcode_col_name']
                X_col_name = row['X_col_name']
                Y_col_name = row['Y_col_name']
            else:
                coord_filename_list = None
                counts_filename_list = None
                barcode_col_name = None
                X_col_name = None
                Y_col_name = None

            done = []
            for j, filename in enumerate(svs_filename_list):
                svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
                svs_prefix = get_svs_prefix(svs_filename)

                save_filename1 = '{}/{}_big_orig.zip'.format(save_root3_1, svs_prefix)
                save_filename = '{}/{}_big_attention_map.zip'.format(save_root3, svs_prefix)
                if os.path.exists(save_filename) and os.path.exists(save_filename1):
                    done.append(True)
                else:
                    done.append(False)
            if np.all(done):
                print(PATIENT_ID, 'done')
                continue

            As = []
            jjs = []
            after_encoder_feats_all = {}
            print(patient_ind, len(svs_filename_list), svs_filename_list)
            for j, filename in enumerate(svs_filename_list):
                svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
                svs_prefix = get_svs_prefix(svs_filename)

                svs_filename1 = os.path.realpath(svs_filename)
                local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename1))
                if not os.path.exists(local_svs_filename):
                    os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')
                with h5py.File(os.path.join(args.patches_dir, svs_prefix + '.h5'), 'r') as h5file:  # the mask_root is the CLAM patches dir
                    all_coords = h5file['coords'][()]
                    patch_level = h5file['coords'].attrs['patch_level']
                    patch_size = h5file['coords'].attrs['patch_size']

                print('local_svs_filename', local_svs_filename, os.path.getsize(local_svs_filename))
                slide = openslide.open_slide(local_svs_filename)
                dataset = PatchDatasetV2(slide, all_coords, patch_level, patch_size)
                kwargs = {'num_workers': 0,'pin_memory': True, 'shuffle': False}

                if transform is not None:
                    loader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn2)
                else:
                    if model_name == 'CONCH':
                        loader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn_CONCH)
                    else:
                        loader = DataLoader(dataset=dataset, batch_size=64, **kwargs, collate_fn=collate_fn)

                after_encoder_feats = []
                for count, (images, coords) in enumerate(loader):
                    with torch.no_grad():
                        images = images.to(device, non_blocking=True)
                        if model_name == 'mobilenetv3':
                            _ = feature_extractor(images)
                            features = feature_tensors.get('after_flatten_feat')
                        elif model_name == 'ProvGigaPath':
                            features = feature_extractor(images).detach()
                        elif model_name == 'CONCH':
                            features = feature_extractor.encode_image(images, proj_contrast=False, normalize=False).detach()
                        else:  # CLIP, PLIP
                            if transform is not None:
                                features = feature_extractor.encode_image(images).detach()
                            else:
                                features = feature_extractor.get_image_features(images).detach()
                        after_encoder_feats.append(features.cpu().numpy().reshape(len(coords), -1))

                feats = torch.from_numpy(np.concatenate(after_encoder_feats, axis=0))
                del after_encoder_feats, features, images, dataset, loader
                feats = feats.to(device, non_blocking=True)
                after_encoder_feats_all[str(j)] = feats
                with torch.no_grad():
                    results_dict = model(feats, attention_only=True)
                A_raw = results_dict['A_raw'].detach().cpu().numpy()

                A = A_raw[args.cluster_task_index]
                As.append(A)
                jjs.append(j * np.ones_like(A, dtype=np.int32))

            ccs = [len(A) for A in As]
            cc = [0] + np.cumsum(ccs).tolist()
            A = np.concatenate(As)
            jj = np.concatenate(jjs)
            assert len(A) == len(jj), 'wrong, check it'
            ref_scores = np.copy(A)
            for ind1 in range(len(A)):
                A[ind1] = percentileofscore(ref_scores, A[ind1])

            patient_ind_cache_filename = os.path.join(save_root4, 'case_{}_cache.pkl'.format(PATIENT_ID))
            with open(patient_ind_cache_filename, 'wb') as fff:
                pickle.dump({'A': A, 'cc': cc, 'jj': jj, 'As': As, 'jjs': jjs}, fff)

            for j, filename in enumerate(svs_filename_list):

                svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
                svs_prefix = get_svs_prefix(svs_filename)
                svs_filename1 = os.path.realpath(svs_filename)
                local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename1))
                if not os.path.exists(local_svs_filename):
                    os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')
                slide = openslide.open_slide(local_svs_filename)
                
                h5filename = os.path.join(args.patches_dir, svs_prefix + '.h5')
                with h5py.File(h5filename, 'r') as h5file:  # the mask_root is the CLAM patches dir
                    all_coords = h5file['coords'][()]
                    patch_level = h5file['coords'].attrs['patch_level']
                    patch_size = h5file['coords'].attrs['patch_size']

                print("do heatmap big")
                if True:
                    job_params.append(
                        (f"{svs_filename}", f"{patient_ind_cache_filename}", f"{save_root3}", f"{args.image_ext}",
                         f"{h5filename}", f"{j}", f"{save_root3_1}")
                    )

                    if len(job_params) == 5:
                        job_df = pd.DataFrame(job_params, columns=['svs_filename', 'cache_filename', 'save_root3', 'image_ext', 'h5filename', 'j', 'save_root3_1'])
                        job_params_filename = os.path.join(save_root4, 'job_params_{}_{}_{}_{}_{}.xlsx'.format(os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank), row_ind, j))
                        job_df.to_excel(job_params_filename)
                        command_txt = f'sbatch gen_big_images_and_atten_map_v3.sh "{job_params_filename}"'
                        print(command_txt)
                        os.system(command_txt)
                        job_params = []

                file_ind = file_inds_dict[svs_prefix]
                print(svs_prefix)

                feats = after_encoder_feats_all[str(j)]
                if 'after_encoder' in args.cluster_feat_name:
                    x = feats.cpu().numpy()
                else:
                    with torch.no_grad():
                        results_dict = model(feats, attention_only=True)
                    x = feature_tensors[args.cluster_feat_name].cpu().numpy()
                y = label * np.ones((x.shape[0],), dtype=np.int32)

                # extract the image patches
                patches = []
                positions = []
                for ind1, coord in enumerate(all_coords):
                    patches.append((int(coord[0]), int(coord[1])))
                    positions.append((svs_prefix, int(coord[0]), int(coord[1]), label))

                with open(os.path.join(save_root4, 'slide_{}_cache.pkl'.format(svs_prefix)), 'wb') as fff:
                    pickle.dump({'x': x, 'y': y, 'patches': patches, 'positions': positions,
                                 'patch_info_dict': {'patch_level': patch_level, 'patch_size': patch_size, 'locations': all_coords},
                                 'level_dimensions': slide.level_dimensions,
                                 'level_downsamples': slide.level_downsamples}, fff)

                if 'openslide.mpp-x' in slide.properties:
                    mpp = float(slide.properties['openslide.mpp-x'])
                elif 'tiff.XResolution' in slide.properties:
                    mpp = 10000 / float(slide.properties['tiff.XResolution'])
                else:
                    mpp = 0.25

                if mpp < 0.1 or mpp > 0.7:
                    print('check H&E files, mpp is {}'.format(mpp))

                slide.close()
                gc.collect()

                if os.path.exists(local_svs_filename):
                    os.system(f'rm -rf "{local_svs_filename}"')

                # processing gene data for ST data
                if not args.pre_computed_vst and coord_filename_list is not None:
                    if isinstance(X_col_name, int):
                        coord_df = pd.read_csv(coord_filename_list[j], header=None, index_col=0)
                    else:
                        coord_df = pd.read_csv(coord_filename_list[j])
                    if 'h5' in counts_filename_list[j]:
                        counts_df = scanpy.read_10x_h5(counts_filename_list[j]).to_df().T
                    else:
                        counts_df = pd.read_csv(counts_filename_list[j], sep='\t')
                    vst_filename = os.path.join(save_root_vst, '{}.tsv'.format(svs_prefix))
                    new_counts_filename = os.path.join(save_root_vst, svs_prefix + '.csv')
                    if not os.path.exists(new_counts_filename):
                        print('compute vst file from gene counts')
                        if counts_df.isna().sum().sum() > 0:
                            print('fatal error for {}'.format(svs_prefix))
                        counts_df = counts_df.astype(np.float32)
                        counts_df = counts_df.fillna(0)
                        counts_df = counts_df.groupby(counts_df.index).sum()
                        invalid_col_index = np.where((counts_df != 0).sum() < 500)[0]

                        if len(invalid_col_index):
                            counts_df = counts_df.drop(columns=counts_df.columns[invalid_col_index])

                        row_index = np.where(counts_df.sum(axis=1) == 0)[0]
                        if len(row_index):
                            counts_df = counts_df.drop(index=counts_df.iloc[row_index].index)
                        counts_df.to_csv(new_counts_filename, sep='\t')

                        joblines = [
                            '#!/bin/bash\nmodule load R\n',
                            'Rscript --vanilla compute_vst.R "{}" "{}"\n\n\n'.format(new_counts_filename, vst_filename)
                        ]

                        temp_job_filename = f'{local_temp_dir}/job_{row_ind}.sh'
                        with open(temp_job_filename, 'w') as fp:
                            fp.writelines(joblines)
                        time.sleep(0.5)
                        os.system(f'bash "{temp_job_filename}"')

                    final_pkl_filename = os.path.join(
                        save_root_gene_data, svs_prefix + '_gene_data.pkl')
                    with open(final_pkl_filename, 'wb') as fp:
                        pickle.dump({
                            'gene_data_vst': vst_filename,
                            'all_coords': all_coords,
                            'patch_level': patch_level,
                            'patch_size': patch_size,
                            'mpp': mpp,
                            'svs_filename': svs_filename,
                            'coord_df': coord_df,
                            'counts_df': counts_df,
                            'barcode_col_name': barcode_col_name,
                            'X_col_name': X_col_name,
                            'Y_col_name': Y_col_name
                        }, fp)

        if len(job_params) > 0:
            job_df = pd.DataFrame(job_params, columns=['svs_filename', 'cache_filename', 'save_root3', 'image_ext', 'h5filename', 'j', 'save_root3_1'])
            job_params_filename = os.path.join(save_root4, 'job_params_{}_{}_{}_final.xlsx'.format(os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank)))
            job_df.to_excel(job_params_filename)
            command_txt = f'sbatch gen_big_images_and_atten_map_v3.sh "{job_params_filename}"'
            print(command_txt)
            os.system(command_txt)
            job_params = []

    if args.only_step1 == 'yes':
        sys.exit(0)


    if args.only_step1 == 'gen_patches':
        local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank),
                                      str(idr_torch.local_rank))
        os.makedirs(local_temp_dir, exist_ok=True)

        json_files = glob.glob(os.path.join(save_root1, 'analysis', '{}_top_{}/'.format(args.cluster_task_name, args.num_patches), '**', '*_cs.json'), recursive=True)
        all_patches = {}
        if len(json_files):
            for f in json_files:
                svs_prefix = os.path.basename(f).replace('_cs.json', '')
                if svs_prefix not in all_patches:
                    all_patches[svs_prefix] = set()
                with open(f, 'r') as fp:
                    dd = json.load(fp)
                for k, v in dd.items():
                    if len(v):
                        lines = v.strip().split('\n')
                        for line in lines:
                            splits = line.split(',')
                            all_patches[svs_prefix].add(
                                (int(splits[-2]), int(splits[-1])))

        save_root5 = os.path.join(save_root1, 'patch_images', '{}_top_{}/'.format(args.cluster_task_name, args.num_patches))
        os.makedirs(save_root5, exist_ok=True)

        existed_prefixes = set([os.path.basename(f).replace(
            '.tar.gz', '') for f in glob.glob(os.path.join(save_root5, '*.tar.gz'))])

        for row_ind, row in df.iterrows():

            svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
            svs_filename_list = svs_filenames.split(',')  # 2 slides

            for j, filename in enumerate(svs_filename_list):
                svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
                svs_prefix = get_svs_prefix(svs_filename)
                if svs_prefix in existed_prefixes:
                    print('existed')
                    continue
                if svs_prefix not in all_patches:
                    continue
                svs_filename1 = os.path.realpath(svs_filename)
                local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename1))
                if not os.path.exists(local_svs_filename):
                    os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')
                slide = openslide.open_slide(local_svs_filename)
                h5filename = os.path.join(args.patches_dir, svs_prefix + '.h5')
                with h5py.File(h5filename, 'r') as h5file:  # the mask_root is the CLAM patches dir
                    all_coords = h5file['coords'][()]
                    patch_level = h5file['coords'].attrs['patch_level']
                    patch_size = h5file['coords'].attrs['patch_size']
                if svs_prefix not in all_patches:
                    all_patches[svs_prefix] = all_coords

                fh = io.BytesIO()
                tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
                for coord in all_patches[svs_prefix]:
                    patch = slide.read_region(location=(int(coord[0]), int(coord[1])), level=patch_level,
                                              size=(patch_size, patch_size)).convert(
                        'RGB')  # .resize((128, 128))  # BGR
                    # patch.save(os.path.join(patches_save_dir, 'x{}_y{}.JPEG'.format(coord[0], coord[1])))
                    im_buffer = io.BytesIO()
                    patch.save(im_buffer, format='JPEG')
                    info = tarfile.TarInfo(
                        name='{}/x{}_y{}.JPEG'.format(svs_prefix, coord[0], coord[1]))
                    info.size = im_buffer.getbuffer().nbytes
                    info.mtime = time.time()
                    im_buffer.seek(0)
                    tar_fp.addfile(info, im_buffer)
                tar_fp.close()
                with open('{}/{}.tar.gz'.format(save_root5, svs_prefix), 'wb') as fp:
                    fp.write(fh.getvalue())

                if os.path.exists(local_svs_filename):
                    os.system(f'rm -rf "{local_svs_filename}"')
        sys.exit(0)
        
    labels_dict = {
        'response_groups': {0: 'BadResponse', 1: 'GoodResponse'},
        'group_label': {0: 'group0', 1: 'group1'},
        'one_class': {0: 'class0'},
        'one_patient': {0: 'class0'},
    }

    step2_results = []
    if args.cluster_task_name == 'one_class' or args.cluster_task_name == 'one_patient':  # for ST data
        if args.cluster_task_name not in df.columns:
            df[args.cluster_task_name] = [0 for _ in range(len(df))]
        for row_ind in range(len(df)):
            row_df = df.iloc[row_ind:(row_ind + 1), :].reset_index(drop=True)
            result = step2(args, row_df, labels_dict, save_root4, file_inds_dict, save_root1, save_root_gene_da_dir, is_single_patient=True)
            step2_results.append(result)
    else:
        included_columns = ['patient_ind', 'PATIENT_ID'] + \
            ['HEfiles' if 'HEfiles' in df.columns else 'DX_filename']
        if args.cluster_task_name in df.columns:
            included_columns.append(args.cluster_task_name)
        df = df[included_columns].set_index('PATIENT_ID')

        df.index.name = 'PATIENT_ID'
        df = df.reset_index()
        print(df)

        for k, v in CLASSIFICATION_DICT.items():
            if k in labels_dict:
                continue
            labels_dict[k] = {kk: vv for kk, vv in enumerate(v)}

        if args.cluster_task_name in df.columns:  # only use a subset
            df = df[~df[args.cluster_task_name].isna()]
            df = df.astype({args.cluster_task_name: int})

        try:
            df = df[df[args.cluster_task_name].isin(list(labels_dict[args.cluster_task_name]))].reset_index(drop=True)
        except:
            return step2_results
        print('begin step2')
        print(df)
        result = step2(args, df, labels_dict, save_root4, file_inds_dict,
              save_root1, save_root_gene_da_dir,
              is_single_patient=False)
        step2_results.append(result)
    return step2_results


def step2(args, df, labels_dict, save_root4, file_inds_dict, save_root1, save_root_gene_da_dir, is_single_patient: bool = False):
    if is_single_patient and len(df) != 1:
        print('error configuration')
        return
    PATIENT_ID = None
    if is_single_patient:
        PATIENT_ID = str(df.loc[0, 'PATIENT_ID'])

    if PATIENT_ID is None:
        temp_data_filename = os.path.join(save_root_gene_da_dir, '{}.pkl'.format(args.cluster_task_name))
    else:
        temp_data_filename = os.path.join(save_root_gene_da_dir, PATIENT_ID, '{}.pkl'.format(args.cluster_task_name))
        os.makedirs(os.path.join(save_root_gene_da_dir, PATIENT_ID), exist_ok=True)

    if os.path.exists(temp_data_filename):
        return None

    X0 = []
    Y0 = []
    file_inds = []
    patch_info_dicts = {}
    level_dimensions_dict = {}
    level_downsamples_dict = {}
    PATIENT_ID = None
    for row_ind, row in df.iterrows():

        svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
        svs_filename_list = svs_filenames.split(',')  # 2 slides
        patient_ind = row['patient_ind']
        PATIENT_ID = row['PATIENT_ID']

        patient_ind_cache_filename = os.path.join(save_root4, 'case_{}_cache.pkl'.format(PATIENT_ID))
        if not os.path.exists(patient_ind_cache_filename):
            continue
        with open(patient_ind_cache_filename, 'rb') as fff:
            tmpdata = pickle.load(fff)
            A = tmpdata['A']
            cc = tmpdata['cc']
            jj = tmpdata['jj']
            As = tmpdata['As']
            jjs = tmpdata['jjs']

        sort_ind = np.argsort(A)
        num_ps = args.num_patches if len(sort_ind) > args.num_patches else len(sort_ind)
        if True:
            selected_indices = sort_ind if is_single_patient else sort_ind[-int(num_ps):]
        else:
            selected_indices = sort_ind
        jj_selected = jj[selected_indices]

        for j, filename in enumerate(svs_filename_list):
            svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
            svs_prefix = get_svs_prefix(svs_filename)
            file_ind = file_inds_dict[svs_prefix]

            selected_indices_j = selected_indices[jj_selected == j] - cc[j]
            if len(selected_indices_j) == 0:
                print('no patches in this image')
                continue

            single_cache_filename = os.path.join(save_root4, 'slide_{}_cache.pkl'.format(svs_prefix))
            if not os.path.exists(single_cache_filename):
                continue

            with open(single_cache_filename, 'rb') as fff:
                tmpdata = pickle.load(fff)
                x = tmpdata['x'][selected_indices_j]
                y = tmpdata['y'][selected_indices_j]
                patches = [tmpdata['patches'][ii] for ii in selected_indices_j]
                positions = [tmpdata['positions'][ii] for ii in selected_indices_j]
                patch_info_dicts[file_ind] = tmpdata['patch_info_dict']
                level_dimensions_dict[file_ind] = tmpdata['level_dimensions']
                level_downsamples_dict[file_ind] = tmpdata['level_downsamples']
                X0 += x.tolist()
                Y0.append(y)
            file_inds.append(file_ind * np.ones((x.shape[0],), dtype=np.int32))

    if len(X0) == 0:
        print('no X0')
        return

    X0 = np.array(X0)
    Y0 = np.concatenate(Y0, axis=0)
    file_inds = np.concatenate(file_inds, axis=0)

    print('collect data for UMAP')
    print(X0.shape)
    print(Y0.shape)


    is_reduced_sample = False
    kmeans0_Y = None
    if False:  # len(X0) > 30000:  # downsample the data
        print('too many samples, subsample it')
        kmeans0 = cluster.KMeans(n_clusters=10000, n_init='auto', random_state=42)
        kmeans0_Y = kmeans0.fit_predict(X0)
        X = kmeans0.cluster_centers_
        is_reduced_sample = True
    else:
        X = X0
        Y = Y0

    all_params = []
    for feature_normalization_type in ['meanstd']: 
        for dimension_reduction_method in ['none', 'umap3d']: # ['none', 'pca3d', 'umap3d']:
            for clustering_method in ['kmeans']: #['kmeans', 'hierarchical']:
                for num_clusters in [8]:  # [8, 16]:
                    for clustering_distance_metric in ['euclidean']:
                        all_params.append((args, X, Y, is_reduced_sample, kmeans0_Y, df, labels_dict, save_root4, file_inds_dict, save_root1,
                                           is_single_patient, file_inds, patch_info_dicts, level_dimensions_dict, level_downsamples_dict,
                                           feature_normalization_type, dimension_reduction_method, clustering_method, num_clusters, clustering_distance_metric))

    ncpus = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else min(8, multiprocessing.cpu_count())        
    print('ncpus', ncpus)
    with multiprocessing.Pool(processes=ncpus) as p:
        results = p.starmap(step2_routine, all_params)

    time.sleep(1)
    print(len(results))

    alldata = []
    for item in results:
        for item1 in item:
            item1['data'] = pickle.dumps(item1['data'])
            alldata.append(item1)

    gc.collect()

    if len(alldata):
        with open(temp_data_filename, 'wb') as fp:
            pickle.dump({'alldata': alldata}, fp)

        generate_result_files_from_pkl(temp_data_filename)
    else:
        print('no alldata')
    return None


def step2_routine(args, X, Y, is_reduced_sample, kmeans0_Y, df, labels_dict, save_root4, file_inds_dict, save_root1,
                  is_single_patient, file_inds, patch_info_dicts, level_dimensions_dict, level_downsamples_dict,
                  feature_normalization_type, dimension_reduction_method, clustering_method, num_clusters, clustering_distance_metric):

    label_title = args.cluster_task_name

    if feature_normalization_type == 'meanstd':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif feature_normalization_type == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = np.copy(X)

    if dimension_reduction_method == 'none':
        feats_embedding = X_scaled
        reducer = None
    elif dimension_reduction_method == 'pca3d':
        reducer = PCA(n_components=3, random_state=42)
        feats_embedding = reducer.fit_transform(X_scaled)
    elif dimension_reduction_method == 'umap3d':
        reducer = umap.UMAP(random_state=42, metric=clustering_distance_metric, n_components=3)
        feats_embedding = reducer.fit_transform(X_scaled)
    else:
        raise ValueError('wrong dimension reduction method.')
    del X_scaled

    if clustering_method == 'kmeans':
        print('do kmeans')
        kmeans = cluster.KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        all_preds0 = kmeans.fit_predict(feats_embedding)
        final_centroids0 = kmeans.cluster_centers_
        nearest_centroid = None

    elif clustering_method == 'hierarchical':
        print('do hierarchical')
        hierarchical = cluster.AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', metric=clustering_distance_metric)
        all_preds0 = hierarchical.fit_predict(feats_embedding)
        nearest_centroid = neighbors.NearestCentroid(metric=clustering_distance_metric)
        nearest_centroid.fit(feats_embedding, all_preds0)
        final_centroids0 = nearest_centroid.centroids_
    else:
        raise ValueError('wrong clustering method.')

    if is_reduced_sample:
        if scaler is not None:
            X_scaled = scaler.transform(X0)
        else:
            X_scaled = X0
        if dimension_reduction_method == 'none':
            feats_embedding = X_scaled
        elif dimension_reduction_method == 'pca3d':
            feats_embedding = reducer.transform(X_scaled)
        elif dimension_reduction_method == 'umap3d':
            feats_embedding = reducer.transform(X_scaled)
        else:
            raise ValueError('wrong dimension reduction method.')
        del X0
        print('is_reduced_sample == True')
        all_preds0 = all_preds0[kmeans0_Y]
        print('all_preds0 shape: ', all_preds0.shape)

    clustering_keep_thresholds = [1]  # [1, 50, 100]
    if is_single_patient:
        clustering_keep_thresholds = [1]  # [1, 5, 10]  # ST

    all_results = []
    for clustering_keep_threshold in clustering_keep_thresholds:
        print(f'begin clustering_keep_threshold={clustering_keep_threshold}')
        all_preds = np.copy(all_preds0)
        final_centroids = np.copy(final_centroids0)

        save_root = '{}/analysis/{}_top_{}/{}_{}_{}_{}_{}_{}_clustering'.format(
            save_root1,
            args.cluster_task_name,
            args.num_patches,
            feature_normalization_type, dimension_reduction_method,
            clustering_method, clustering_distance_metric,
            num_clusters, clustering_keep_threshold)
        if is_single_patient:
            save_root = os.path.join(save_root, str(df.loc[0, 'PATIENT_ID']))
        else:
            save_root = os.path.join(save_root, 'ALL')
        
        os.makedirs(save_root, exist_ok=True)

        models_pickle = {
            'args': args,
            'scaler': scaler,
            'reducer': reducer,
            'clustering_model': kmeans if clustering_method == 'kmeans' else hierarchical,
            'nearest_centroid': nearest_centroid,
            'all_preds0': all_preds0,
            'final_centroids0': final_centroids0
        }

        # remove the clusters whose sample count is less than this threshold
        invalid_clustering_labels = []
        clustering_labels = np.unique(all_preds)
        final_clustering_labels = []
        if clustering_keep_threshold > 1:
            print('need to remove some clusters')
            for clustering_label in clustering_labels:
                clustering_inds = np.where(all_preds == clustering_label)[0]
                if len(clustering_inds) < clustering_keep_threshold:
                    invalid_clustering_labels.append(clustering_label)
                else:
                    final_clustering_labels.append(clustering_label)
        else:
            final_clustering_labels = clustering_labels

        if clustering_keep_threshold > 1 and len(final_clustering_labels) == len(clustering_labels):
            print('all labels kept, skip')
            continue

        if len(invalid_clustering_labels) > 0:
            final_centroids = np.delete(
                final_centroids, invalid_clustering_labels, axis=0)

        if len(final_centroids) == 0:
            print('no clusters.')
            continue

        final_clustering_labels = {label: ind for ind, label in enumerate(final_clustering_labels)}
        distances = distance_matrix(feats_embedding, final_centroids)

        cluster_counts_per_case = []
        limited_count = 200000
        current_label_counts = {label: 0 for label in labels_dict[label_title].keys()}

        all_data_for_umap3dvis = []
        all_distances_for_each_cluster = {k: 0 for k in range(len(final_centroids))}
        all_cluster_data = {}
        cs_dict_data = {}
        patch_annos_data = {}
        for row_ind, row in df.iterrows():
            svs_filenames = row['HEfiles'] if 'HEfiles' in row else row['DX_filename']
            patient_ind = row['patient_ind']
            PATIENT_ID = row['PATIENT_ID']
            label = row[label_title]
            label_text = labels_dict[label_title][label]
            current_label_counts[label] += 1
            svs_filename_list = svs_filenames.split(',')

            with open(os.path.join(save_root4, 'case_{}_cache.pkl'.format(PATIENT_ID)), 'rb') as fff:
                tmp = pickle.load(fff)
                A = tmp['A']
                jj = tmp['jj']
                cc = tmp['cc']
                del tmp

            sort_ind = np.argsort(A)
            num_ps = args.num_patches if len(sort_ind) > args.num_patches else len(sort_ind)
            if True:
                selected_indices = sort_ind if is_single_patient else sort_ind[-int(num_ps):]
            else:
                selected_indices = sort_ind
            jj_selected = jj[selected_indices]

            for j, filename in enumerate(svs_filename_list):
                svs_filename = os.path.join(args.svs_dir, os.path.basename(filename).replace(' ', '').replace('&', ''))
                svs_prefix = get_svs_prefix(svs_filename)
                file_ind = file_inds_dict[svs_prefix]

                file_inds_j = np.where(file_inds == file_ind)[0]
                reduced_feats = feats_embedding[file_inds_j, :]
                distances_j = distances[file_inds_j, :]
                nn_labels_j = all_preds[file_inds_j]
                selected_indices_j = selected_indices[jj_selected == j] - cc[j]

                invalid_inds = []
                if len(invalid_clustering_labels) > 0:
                    for invalid_clustering_label in invalid_clustering_labels:
                        invalid_inds.append(np.where(nn_labels_j == invalid_clustering_label)[0])
                    if len(invalid_inds):
                        invalid_inds = np.concatenate(invalid_inds)
                if len(invalid_inds) > 0:
                    reduced_feats = np.delete(reduced_feats, invalid_inds, axis=0)
                    distances_j = np.delete(distances_j, invalid_inds, axis=0)
                    nn_labels_j = np.delete(nn_labels_j, invalid_inds, axis=0)
                    selected_indices_j = np.delete(selected_indices_j, invalid_inds, axis=0)

                if len(selected_indices_j) == 0:
                    print('no patches in this image')
                    continue

                # update label mapping
                nn_labels_j_new = np.zeros_like(nn_labels_j)
                for nn_labels_j_old in np.unique(nn_labels_j):
                    nn_labels_j_new[np.where(nn_labels_j == nn_labels_j_old)[0]] = final_clustering_labels[nn_labels_j_old]
                nn_labels_j = nn_labels_j_new

                level_dimensions = level_dimensions_dict[file_ind]
                if len(level_dimensions) == 1:
                    vis_level = 0
                else:
                    dimension = level_dimensions[1]
                    if dimension[0] > 100000 or dimension[1] > 100000:
                        vis_level = 2
                    else:
                        vis_level = 1

                patch_info_dict = patch_info_dicts[file_ind]
                patch_level = patch_info_dict['patch_level']
                dimension = level_dimensions[patch_level]
                dimension_vis_level = level_dimensions[vis_level]
                patch_size = patch_info_dict['patch_size']
                coords_original = patch_info_dict['locations']

                level_downsamples = _assertLevelDownsamplesV2(level_dimensions, level_downsamples_dict[file_ind])
                downsample0 = level_downsamples[patch_level]
                downsample = level_downsamples[vis_level]
                scale = [downsample0[0] / downsample[0], downsample0[1] / downsample[1]]
                patch_size_vis_level = np.ceil(patch_size * scale[0]).astype(int)

                unique1, counts1 = np.unique(nn_labels_j, return_counts=True)
                aaa = np.zeros_like(np.arange(len(final_centroids)))
                aaa[unique1] = counts1
                counts2 = counts1.astype(np.float32)
                cluster_counts_per_case.append([svs_prefix, label, label_text, patch_level, vis_level, dimension[0], dimension[1], dimension_vis_level[0], dimension_vis_level[1], patient_ind] + aaa.tolist())

                coords = coords_original[selected_indices_j]
                coords1 = np.copy(coords)
                coords2 = np.floor(coords / patch_size).astype(np.int32)
                coords_in_vis_level = np.ceil(coords_original * np.array(scale)).astype(int)[selected_indices_j]
                coords_in_original = coords_original[selected_indices_j]

                cluster_data_filename = '{}/{}_cluster_data.pkl'.format(save_root, svs_prefix)
                all_cluster_data[svs_prefix] = {'coords_in_original': coords_in_original, 'cluster_labels': nn_labels_j}

                gene_data_filename = f'{save_root}/../../../../gene_data/{svs_prefix}_gene_data.pkl'
                if args.cluster_task_name == 'one_class' or args.cluster_task_name == 'one_patient':
                    if os.path.exists(gene_data_filename):
                        with open(gene_data_filename, 'rb') as fp:
                            gene_data_dict = pickle.load(fp)
                        
                        barcode_col_name = gene_data_dict['barcode_col_name']
                        Y_col_name = gene_data_dict['Y_col_name']
                        X_col_name = gene_data_dict['X_col_name']
                        mpp = gene_data_dict['mpp']
                        coord_df = gene_data_dict['coord_df']
                        counts_df = gene_data_dict['counts_df']

                        vst_filename = gene_data_dict['gene_data_vst']
                    else:
                        mpp = None
                        coord_filename = svs_filename.replace('/svs/', '/coords/').replace('.svs', '.csv')
                        vst_filename = svs_filename.replace('/svs/', '/gene_vst/').replace('.svs', '.tsv')
                        coord_df = pd.read_csv(coord_filename, index_col=0, low_memory=False)
                        barcode_col_name = row['barcode_col_name']
                        X_col_name = row['X_col_name']
                        Y_col_name = row['Y_col_name']
                        try:
                            barcode_col_name = int(float(barcode_col_name)) # read_csv index_col=0
                            X_col_name = int(float(X_col_name))
                            Y_col_name = int(float(Y_col_name))
                        except:
                            pass

                    if os.path.exists(vst_filename):

                        vst = pd.read_csv(vst_filename, sep='\t', index_col=0, low_memory=False)
                        vst = vst.subtract(vst.mean(axis=1), axis=0)

                        barcodes = coord_df.index.values.tolist()
                        stY = coord_df[Y_col_name if Y_col_name in coord_df.columns else str(Y_col_name)].values.tolist()
                        stX = coord_df[X_col_name if X_col_name in coord_df.columns else str(X_col_name)].values.tolist()

                        if mpp is None:
                            st_patch_size = patch_size
                        else:
                            st_patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
                        st_all_coords = np.array([stX, stY]).T
                        st_all_coords[:, 0] -= st_patch_size // 2
                        st_all_coords[:, 1] -= st_patch_size // 2
                        st_all_coords = st_all_coords.astype(np.int32)

                        vst = vst.T
                        vst.index.name = 'barcode'
                        vst.columns = [n.upper() for n in vst.columns]  # genes upper
                        valid_barcodes = set(vst.index.values.tolist())

                        cluster_coords = coords_in_original
                        cluster_labels = nn_labels_j

                        cluster_barcodes = []
                        innnvalid = 0
                        iinds = []
                        for iiii, (x, y) in enumerate(cluster_coords):
                            ind = np.where((st_all_coords[:, 0] == x) & (st_all_coords[:, 1] == y))[0]
                            if len(ind) > 0:
                                barcoode = barcodes[ind[0]]
                                if barcoode in valid_barcodes:
                                    cluster_barcodes.append(barcoode)
                                    iinds.append(iiii)
                            else:
                                innnvalid += 1
                        cluster_labels = cluster_labels[iinds]
                        cluster_coords = cluster_coords[iinds]
                        vst1 = vst.loc[cluster_barcodes]
                        # counts_df1 = counts_df.T
                        # coord_df1 = coord_df.set_index(barcode_col_name)
                        # coord_df1.index.name = 'barcode'

                        texts = []
                        texts2 = []
                        texts3 = []
                        gene_names = vst1.columns.values
                        cluster_labels_unique = np.unique(cluster_labels)
                        dfs = {
                            'gene_names': gene_names,
                            'labels': cluster_labels_unique,
                            'cluster_barcodes': cluster_barcodes,
                            'cluster_labels': cluster_labels,
                            'cluster_coords': cluster_coords,
                            'coords_in_original': coords_in_original,
                            'cluster_labels_in_original': nn_labels_j
                        }
                        for label in cluster_labels_unique:
                            ind1 = np.where(cluster_labels == label)[0]
                            ind0 = np.where(cluster_labels != label)[0]
                            vst11 = vst1.iloc[ind1] # num_spots x num_genes
                            vst10 = vst1.iloc[ind0]

                            res = ranksums(vst11, vst10)
                            cohens = cohend(vst11, vst10).values
                            zscores = res.statistic
                            pvalues = res.pvalue

                            reject, pvals_corrected, alphacSidakfloat, alphacBonffloat = multipletests(pvalues, method='fdr_bh')

                            dfs[label] = pd.DataFrame({'zscores': zscores, 'pvalues': pvalues, 'pvalues_corrected': pvals_corrected.tolist(), 'cohensd': cohens.tolist()})

                            inds = np.argsort(pvals_corrected)
                            text = '{},{:.6f},{:.6f},{},{}\n'.format(
                                label, pvals_corrected.min(), pvals_corrected.max(),
                                ','.join(['{}({:.5f})'.format(gg, pp) for gg, pp in zip(
                                    gene_names[inds[:20]].tolist(), pvals_corrected[inds[:20]].tolist())]),
                                ','.join(['{}({:.5f})'.format(gg, pp) for gg, pp in zip(gene_names[inds[-20:]].tolist(), pvals_corrected[inds[-20:]].tolist())]))
                            texts.append(text)

                            inds = np.argsort(zscores)
                            text = '{},{:.6f},{:.6f},{},{}\n'.format(
                                label, zscores.min(), zscores.max(),
                                ','.join(['{}({:.5f})'.format(gg, pp) for gg, pp in zip(
                                    gene_names[inds[:20]].tolist(), zscores[inds[:20]].tolist())]),
                                ','.join(['{}({:.5f})'.format(gg, pp) for gg, pp in zip(gene_names[inds[-20:]].tolist(), zscores[inds[-20:]].tolist())]))
                            texts2.append(text)

                            inds = np.argsort(cohens)
                            text = '{},{:.6f},{:.6f},{},{}\n'.format(
                                label, cohens.min(), cohens.max(),
                                ','.join(['{}({:.5f})'.format(gg, pp) for gg, pp in zip(
                                    gene_names[inds[:20]].tolist(), cohens[inds[:20]].tolist())]),
                                ','.join(['{}({:.5f})'.format(gg, pp) for gg, pp in zip(gene_names[inds[-20:]].tolist(), cohens[inds[-20:]].tolist())]))
                            texts3.append(text)

                        with open(cluster_data_filename.replace('_cluster_data.pkl', '_cluster_tests.pkl'), 'wb') as fp:
                            pickle.dump(dfs, fp)
                        with open(cluster_data_filename.replace('_cluster_data.pkl', '_cluster_tests1.csv'), 'w') as fp:
                            fp.writelines(texts)
                        with open(cluster_data_filename.replace('_cluster_data.pkl', '_cluster_tests2.csv'), 'w') as fp:
                            fp.writelines(texts2)
                        with open(cluster_data_filename.replace('_cluster_data.pkl', '_cluster_tests3.csv'), 'w') as fp:
                            fp.writelines(texts3)

                all_patches_json_dict = []
                all_patch_filenames = {iii: []
                                       for iii in range(len(final_centroids))}
                for ind1, coord in enumerate(coords2):
                    reduced_feat = reduced_feats[ind1]
                    nn_label = nn_labels_j[ind1]
                    min_distance = distances_j[ind1][nn_label]
                    all_distances_for_each_cluster[nn_label] += min_distance

                    coord_in_vis_level = coords_in_vis_level[ind1]
                    coord_in_original = coords_in_original[ind1]
                    annoid_str = "annoid-{:d}-{:d}-{:.3f}".format(nn_label, ind1, min_distance)
                    all_patches_json_dict.append(new_web_annotation(nn_label, min_distance, coord_in_vis_level[0], coord_in_vis_level[1], patch_size_vis_level, patch_size_vis_level, annoid_str))

                    all_patch_filenames[nn_label].append(
                        '{:d},{:d},{:.6f},{:.6f},{:.6f},{:d},{:.6f},{:d},{:d},{:d},{:d}\n'.format(
                            coord_in_vis_level[0], coord_in_vis_level[1],
                            reduced_feat[0],
                            reduced_feat[1],
                            reduced_feat[2] if len(reduced_feat) == 3 else 0,
                            nn_label,
                            min_distance,
                            len(nn_labels_j),
                            len(jj_selected),
                            coord_in_original[0], coord_in_original[1]))

                    # add data for umap3d visualization
                    all_data_for_umap3dvis.append(
                        '{},{:.6f},{:.6f},{:.6f},{:d},{:.6f},{:d},{:d},{}.jpg,{:d},{:d}\n'.format(
                            svs_prefix,
                            reduced_feat[0],
                            reduced_feat[1],
                            reduced_feat[2] if len(reduced_feat) == 3 else 0,
                            nn_label,
                            min_distance,
                            coord_in_vis_level[0],
                            coord_in_vis_level[1],
                            annoid_str,
                            coord_in_original[0],
                            coord_in_original[1])
                    )

                cs_dict_data[svs_prefix] = {f'c{iii}': ''.join(all_patch_filenames[iii]) for iii, vvv in all_patch_filenames.items()}

                patch_annos_data[svs_prefix] = all_patches_json_dict

        final_centroids_lines = []
        for c_ind, center in enumerate(final_centroids):
            final_centroids_lines.append('c{},{:.6f},{:.6f},{:.6f},'
                                         '{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                                             c_ind, center[0], center[1], center[2] if len(
                                                 center) > 2 else 0,
                                             np.min(
                                                 all_distances_for_each_cluster[c_ind]),
                                             np.max(
                                                 all_distances_for_each_cluster[c_ind]),
                                             np.mean(
                                                 all_distances_for_each_cluster[c_ind]),
                                             np.median(all_distances_for_each_cluster[c_ind])))

        cluster_cols = ['c{}'.format(i) for i in range(len(final_centroids))]
        counts_df = pd.DataFrame(cluster_counts_per_case,
                                 columns=['filename', args.cluster_task_name,
                                          '{}Name'.format(
                                              args.cluster_task_name),
                                          'patch_level', 'vis_level',
                                          'patch_level_width', 'patch_level_height',
                                          'vis_level_width', 'vis_level_height', 'patient_id'] + cluster_cols)
        counts_df = counts_df.sort_values(args.cluster_task_name)

        for label, label_text in labels_dict[label_title].items():
            counts_df1 = counts_df[counts_df['{}Name'.format(args.cluster_task_name)] == label_text].reset_index(drop=True)
            counts_df1 = counts_df1[['filename', 'patient_id']]

        cluster_counts_per_case1 = counts_df[['filename', args.cluster_task_name, '{}Name'.format(args.cluster_task_name), 'patient_id'] + cluster_cols].copy().reset_index(drop=True)

        # get fractions
        cluster_counts_by_category = counts_df[['{}Name'.format(args.cluster_task_name)] + cluster_cols].copy().groupby('{}Name'.format(args.cluster_task_name)).sum()
        fractions_df = cluster_counts_by_category.copy()
        fractions_df = fractions_df.div(fractions_df.sum(axis=1), axis=0)
        cluster_names = fractions_df.columns.values.tolist()

        keys = list(labels_dict[args.cluster_task_name].keys())

        score_result = {}
        min_val = 1e10
        for ii, key0 in enumerate(keys):
            for jj, key1 in enumerate(keys):
                if ii >= jj:
                    continue

                # plot fraction box plots
                df0 = cluster_counts_per_case1[cluster_counts_per_case1['{}Name'.format(args.cluster_task_name)] == labels_dict[args.cluster_task_name][key0]][cluster_names + ['patient_id']]
                df1 = cluster_counts_per_case1[cluster_counts_per_case1['{}Name'.format(args.cluster_task_name)] == labels_dict[args.cluster_task_name][key1]][cluster_names + ['patient_id']]
                if len(df0) == 0 or len(df1) == 0:
                    continue
                df0 = df0.groupby('patient_id').sum()
                df1 = df1.groupby('patient_id').sum()
                df0_bak = df0.copy()
                df1_bak = df1.copy()

                df0 = df0.div(df0.sum(axis=1), axis=0)
                df1 = df1.div(df1.sum(axis=1), axis=0)

                colors = ['#D7191C', '#2C7BB6']
                data_a = df0.values.T.tolist()
                data_b = df1.values.T.tolist()
                ticks = cluster_names

                # draw circles
                xs = []
                ys = []
                zs = []
                rs = []
                for x1, y1, x2, y2 in zip(np.arange(len(data_a)) * 2.0 - 0.4, data_a, np.arange(len(data_b)) * 2.0 + 0.4, data_b):
                    xs.append((x1 + x2) / 2)
                    z, r = ranksums(y1, y2)
                    zs.append(z)
                    rs.append(r)
                    yy = y1 + y2
                    if len(yy) > 0:
                        ys.append(max(yy))
                    x1 = np.random.normal(x1, 0.04, size=len(y1))
                    x2 = np.random.normal(x2, 0.04, size=len(y2))
                data_a = np.array(data_a).T
                data_b = np.array(data_b).T
                res = ranksums(data_a, data_b)
                cohens = cohend(pd.DataFrame(data_a), pd.DataFrame(data_b)).values
                zscores = res.statistic
                pvalues = res.pvalue

                reject, pvals_corrected, alphacSidakfloat, alphacBonffloat = multipletests(pvalues, method='fdr_bh')

                if ii != jj:  # and min(rs) < 0.1:
                    score_result[f'{ii}_{jj}'] = {'zscores': zs, 'pvalues': rs, 'pvalues_corrected': pvals_corrected.tolist(), 'cohensd': cohens.tolist()}
                    min_val = min(min_val, min(rs))

        score_result['cluster_names'] = cluster_names
        data_dict = {
            'labels_dict': labels_dict,
            'final_centroids': final_centroids,
            'final_centroids_lines': final_centroids_lines,
            'cluster_counts_per_case': cluster_counts_per_case,
            'cs_dict_data': cs_dict_data,
            'patch_annos_data': patch_annos_data,
            'all_data_for_umap3dvis': all_data_for_umap3dvis,
            'all_cluster_data': all_cluster_data,
            'args': args
        }
        all_results.append({'save_root': save_root,
                            'score_result': score_result,
                            'min_val': min_val,
                            'data': data_dict})

    return all_results


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def generate_result_files_from_pkl(temp_data_filename):

    with open(temp_data_filename, 'rb') as fp:
        tmpdata = pickle.load(fp)

    for data_dict0 in tmpdata['alldata']:
        save_root = data_dict0['save_root']
        min_val = data_dict0['min_val']
        score_result = data_dict0['score_result']

        os.makedirs(save_root, exist_ok=True)
        data_dict = pickle.loads(data_dict0['data'])

        args = data_dict['args']
        labels_dict = data_dict['labels_dict']
        label_title = 'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype' if args.cluster_task_name == 'CLS_Molecular_Subtype' else args.cluster_task_name
        for svs_prefix, data in data_dict['all_cluster_data'].items():
            cluster_data_filename = '{}/{}_cluster_data.pkl'.format(save_root, svs_prefix)
            if svs_prefix in data_dict['all_cluster_data']:
                with open(cluster_data_filename, 'wb') as fp:
                    pickle.dump(data_dict['all_cluster_data'][svs_prefix], fp)
            if svs_prefix in data_dict['cs_dict_data']:
                with open('{}/{}_cs.json'.format(save_root, svs_prefix), 'w') as fp:
                    json.dump(data_dict['cs_dict_data'][svs_prefix], fp)
            if svs_prefix in data_dict['patch_annos_data']:
                with open('{}/{}_patches_annotations.json'.format(save_root, svs_prefix), 'w') as fp:
                    json.dump(data_dict['patch_annos_data']
                              [svs_prefix], fp, indent=2)
        if len(data_dict['all_data_for_umap3dvis']) > 0:
            with open('{}/all_data_for_umap3dvis.txt'.format(save_root), 'w') as fp:
                fp.writelines(data_dict['all_data_for_umap3dvis'])
        # with open('{}/{}_{}_final_centroids.txt'.format(save_root, args.cluster_task_name, args.cluster_subset), 'w') as fp:
        with open('{}/{}_{}_final_centroids.txt'.format(save_root, args.cluster_task_name, 'test'), 'w') as fp:
            fp.writelines(data_dict['final_centroids_lines'])

        cluster_cols = ['c{}'.format(i) for i in range(len(data_dict['final_centroids_lines']))]
        counts_df = pd.DataFrame(data_dict['cluster_counts_per_case'],
                                 columns=['filename', args.cluster_task_name, '{}Name'.format(args.cluster_task_name), 'patch_level', 'vis_level',
                                          'patch_level_width', 'patch_level_height', 'vis_level_width', 'vis_level_height', 'patient_id'] + cluster_cols)
        counts_df = counts_df.sort_values(args.cluster_task_name)
        counts_df.to_csv('{}/all_info.csv'.format(save_root))

        for label, label_text in labels_dict[label_title].items():
            counts_df1 = counts_df[counts_df['{}Name'.format(args.cluster_task_name)] == label_text].reset_index(drop=True)
            counts_df1 = counts_df1[['filename', 'patient_id']]
            counts_df1.to_csv('{}/counts_in_for_label_{}.csv'.format(save_root, label_text))

        cluster_counts_per_case = counts_df[['filename', args.cluster_task_name, '{}Name'.format(args.cluster_task_name), 'patient_id'] + cluster_cols].copy().reset_index(drop=True)
        cluster_counts_per_case.to_csv('{}/cluster_counts_per_case.csv'.format(save_root))

        # get fractions
        cluster_counts_by_category = counts_df[['{}Name'.format(args.cluster_task_name)] + cluster_cols].copy().groupby('{}Name'.format(args.cluster_task_name)).sum()
        fractions_df = cluster_counts_by_category.copy()
        fractions_df = fractions_df.div(fractions_df.sum(axis=1), axis=0)
        fractions_df.to_csv('{}/cluster_counts_by_category_fraction.csv'.format(save_root), float_format='%.6f')
        cluster_names = fractions_df.columns.values.tolist()
        keys = list(labels_dict[args.cluster_task_name].keys())

        score_result = {}
        for ii, key0 in enumerate(keys):
            for jj, key1 in enumerate(keys):
                df0 = cluster_counts_per_case[cluster_counts_per_case['{}Name'.format(args.cluster_task_name)] == labels_dict[args.cluster_task_name][key0]][cluster_names + ['patient_id']]
                df1 = cluster_counts_per_case[cluster_counts_per_case['{}Name'.format(args.cluster_task_name)] == labels_dict[args.cluster_task_name][key1]][cluster_names + ['patient_id']]
                if len(df0) == 0 or len(df1) == 0:
                    continue
                df0 = df0.groupby('patient_id').sum()
                df1 = df1.groupby('patient_id').sum()
                df0_bak = df0.copy()
                df1_bak = df1.copy()

                df0 = df0.div(df0.sum(axis=1), axis=0)
                df1 = df1.div(df1.sum(axis=1), axis=0)

                colors = ['#D7191C', '#2C7BB6']
                data_a = df0.values.T.tolist() # numSxnumC --> numCxnumS
                data_b = df1.values.T.tolist()
                ticks = cluster_names

                fig = plt.figure(figsize=(len(cluster_names), 8))
                bpl = plt.boxplot(data_a, positions=np.arange(len(data_a)) * 2.0 - 0.4, sym='',
                                    widths=0.6)
                bpr = plt.boxplot(data_b, positions=np.arange(len(data_b)) * 2.0 + 0.4, sym='',
                                    widths=0.6)
                # colors are from http://colorbrewer2.org/
                set_box_color(bpl, colors[0])
                set_box_color(bpr, colors[1])

                # draw circles
                xs = []
                ys = []
                zs = []
                rs = []
                for x1, y1, x2, y2 in zip(np.arange(len(data_a)) * 2.0 - 0.4, data_a, np.arange(len(data_b)) * 2.0 + 0.4, data_b):
                    xs.append((x1 + x2) / 2)
                    z, r = ranksums(y1, y2)
                    zs.append(z)
                    rs.append(r)
                    yy = y1 + y2
                    if len(yy) > 0:
                        ys.append(max(yy))
                    x1 = np.random.normal(x1, 0.04, size=len(y1))
                    x2 = np.random.normal(x2, 0.04, size=len(y2))
                    plt.scatter(x1, y1, c=colors[0], alpha=0.2)
                    plt.scatter(x2, y2, c=colors[1], alpha=0.2)

                data_a = np.array(data_a).T
                data_b = np.array(data_b).T
                res = ranksums(data_a, data_b)
                cohens = cohend(pd.DataFrame(data_a), pd.DataFrame(data_b))
                zscores = res.statistic
                pvalues = res.pvalue

                reject, pvals_corrected, alphacSidakfloat, alphacBonffloat = multipletests(pvalues, method='fdr_bh')

                plt.plot([], c=colors[0], label=labels_dict[args.cluster_task_name][key0] + '({})'.format(len(df0_bak)))
                plt.plot([], c=colors[1], label=labels_dict[args.cluster_task_name][key1] + '({})'.format(len(df1_bak)))
                plt.legend()

                ticks1 = ['{}\n{:.3f}\n{:.3f}\n{:.3f}'.format(xx, zz, pp, ccc) for xx, zz, pp, ccc in zip(ticks, zs, rs, cohens)]
                plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks1)
                plt.xlim(-2, len(ticks) * 2)
                plt.title(args.cluster_task_name)
                plt.xlabel('Clusters & Z-score & P-value & Cohens\'d', fontweight='bold', fontsize=15)
                plt.ylabel('Patch fractions', fontweight='bold', fontsize=15)
                plt.tight_layout()
                plt.savefig(os.path.join('{}/cluster_fractions_boxplot_between_{}_and_{}.png'.format(save_root, key0, key1)), bbox_inches='tight', dpi=600)
                plt.close(fig)

                with open(os.path.join('{}/cluster_fractions_boxplot_between_{}_and_{}.csv'.format(save_root, key0, key1)), 'w') as fp:
                    fp.writelines([','.join(ticks) + '\n', ','.join(['{:.4f}'.format(vvv) for vvv in zs]) + '\n', ','.join(['{:.4f}'.format(vvv) for vvv in rs]) + '\n'])
                score_result[f'{ii}_{jj}'] = {'zscores': zs, 'pvalues': rs, 'pvalues_corrected': pvals_corrected, 'cohensd': cohens}

        with open('{}/cluster_tests.pkl'.format(save_root), 'wb') as fp:
            pickle.dump(score_result, fp)


if __name__ == '__main__':
    main()
