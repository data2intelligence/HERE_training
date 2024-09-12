import sys
import os
import glob
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from PIL import Image


def main_kather100k():
    name = 'kather100k'
    save_root = f'./data/kather100k'
    os.makedirs(os.path.join(save_root, 'ALL'), exist_ok=True)

    data_root = './downloaded/KatherData/NCT-CRC-HE-100K-NONORM'
    labels = [label for label in os.listdir(data_root) if label[0] != '.']
    labels_dict = {label: i for i, label in enumerate(labels)}

    all_items = []
    for label_name, label in labels_dict.items():
        files = glob.glob(os.path.join(data_root, label_name, '*.tif'))
        for f in files:
            os.system('ln -sf "{}" "{}"'.format(f,
                      os.path.join(save_root, 'ALL')))
        all_items.extend([(os.path.basename(f), label) for f in files])

    df = pd.DataFrame(all_items, columns=['Patch Names', 'label'])
    df.to_csv(os.path.join(save_root, name + '_patch_label_file.csv'))


def main_bcss():
    name = 'bcss'
    save_root = './data/BCSS/'

    bcss_root = './downloaded/0_Public-data-Amgad2019_0.25MPP/'
    df = pd.read_csv(os.path.join(
        bcss_root, 'meta/gtruth_codes.tsv'), sep='\t')
    labels_dict = {int(label): label_name for label_name,
                   label in zip(df['label'].values, df['GT_code'].values)}
    palette_filename = os.path.join(bcss_root, 'palette.npy')
    palette = np.load(palette_filename)

    for BCSS_CROP_SIZE in [256, 512]:   # 256, 512
        for BCSS_TARGET_SIZE in [256]:  # 256,512,etc
            for BCSS_RATIO in [0.5, 0.8]:      # 0.5
                for BCSS_MIN in [50]:        # 50
                    for BCSS_OVERLAP in [False]:   # True or False
                        bcss_patches_filename = f"{bcss_root}/imgs_{BCSS_CROP_SIZE}_{BCSS_TARGET_SIZE}_{BCSS_RATIO}_{BCSS_MIN}_{BCSS_OVERLAP}.pth"
                        save_dir = os.path.join(
                            save_root, f'bcss_{BCSS_CROP_SIZE}_{BCSS_TARGET_SIZE}_{BCSS_RATIO}_{BCSS_MIN}_{BCSS_OVERLAP}')
                        os.makedirs(os.path.join(
                            save_dir, 'ALL'), exist_ok=True)
                        with open(bcss_patches_filename, 'rb') as fp:
                            tmpdata = torch.load(fp)
                            # 224x244 30000 pathes
                            all_patches = tmpdata['all_patches']
                            all_positions = tmpdata['all_positions']
                            # all_colored_patches = tmpdata['all_colored_patches']
                            all_ratios = tmpdata['all_ratios']

                        labels = all_positions[:, -1]
                        unique_labels = np.unique(labels)
                        all_items = []
                        for label in unique_labels:
                            label_name = labels_dict[label]
                            inds = np.where(labels == label)[0]
                            for ind in inds:
                                filename = '{}_patch_x{}_y{}.jpg'.format(
                                    label_name, all_positions[ind, 1], all_positions[ind, 2])
                                cv2.imwrite(os.path.join(
                                    save_dir, 'ALL', filename), all_patches[ind])
                                all_items.append((filename, label))

                        df = pd.DataFrame(all_items, columns=[
                                          'Patch Names', 'label'])
                        df.to_csv(os.path.join(
                            save_dir, 'patch_label_file.csv'))


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def main_PanNuke():

    root = './downloaded/PanNuke/Fold1'
    save_root = './data/PanNuke/'
    os.makedirs(os.path.join(save_root, 'ALL'), exist_ok=True)

    labels_dict = {0: 'Neoplastic cells', 1: 'Inflammatory',
                   2: 'Connective/Soft tissue cells', 3: 'Dead Cells', 4: 'Epithelial', 6: 'Background'}
    images = np.load(f"{root}/images/fold1/images.npy")
    masks = np.load(f"{root}/masks/fold1/masks.npy")

    print("Process images")
    all_items = []
    for i in tqdm(range(len(images)), total=len(images)):

        filename = 'fold1_patch_{}.jpg'.format(i)

        if True:
            out_img = images[i]
            im = Image.fromarray(out_img.astype(np.uint8))
            im.save(os.path.join(save_root, 'ALL', filename))

        # need to create instance map and type map with shape 256x256
        mask = masks[i]
        inst_map = np.zeros((256, 256))
        num_nuc = 0
        for j in range(5):
            # copy value from new array if value is not equal 0
            layer_res = remap_label(mask[:, :, j])
            # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc = num_nuc + np.max(layer_res)
        inst_map = remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for j in range(5):
            layer_res = (
                (j + 1) * np.clip(mask[:, :, j], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)

        if inst_map.max() < 1:
            continue
        counts = {k: 0 for k, v in labels_dict.items()}
        for inst_i in range(1, inst_map.max() + 1):
            inds = np.where(inst_map == inst_i)
            values = type_map[inds[0], inds[1]]
            a, b = np.unique(values, return_counts=True)
            binds = np.argsort(b)[::-1]
            label = a[binds[0]] - 1
            counts[label] += 1
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if counts[0][1] > 0:
            all_items.append((filename, counts[0][0]))

    df = pd.DataFrame(all_items, columns=['Patch Names', 'label'])
    df.to_csv(os.path.join(save_root, 'patch_label_file.csv'))


def main_NuCLS():

    root = './downloaded/NuCLS/Corrected single-rater dataset/QC'
    save_root = './data/NuCLS/'
    os.makedirs(os.path.join(save_root, 'ALL'), exist_ok=True)

    csv_files = glob.glob(os.path.join(root, 'csv', '*.csv'))
    labels = []
    filenames = []
    for f in csv_files:
        df = pd.read_csv(f)
        a = df['main_classification'].value_counts()
        b = list(a.keys())
        c = list(a.values)
        inds = np.argsort(c)[::-1]
        label = b[inds[0]]
        labels.append(label)
        prefix = os.path.basename(f).replace('.csv', '')
        filename = prefix + '.jpg'
        im = Image.open(os.path.join(root, 'rgb', '{}.png'.format(prefix)))
        im.save(os.path.join(save_root, 'ALL', filename))
        filenames.append(filename)

    labels_dict = {i: v for i, v in enumerate(np.unique(np.array(labels)))}
    labels_dict = {0: 'AMBIGUOUS',
                   1: 'lymphocyte',
                   2: 'macrophage',
                   3: 'nonTILnonMQ_stromal',
                   4: 'other_nucleus',
                   5: 'plasma_cell',
                   6: 'tumor_nonMitotic'}
    labels_dict = {v:k for k,v in labels_dict.items()}
    all_items = []
    for i in range(len(labels)):
        all_items.append((filenames[i], labels_dict[labels[i]]))
    df = pd.DataFrame(all_items, columns=['Patch Names', 'label'])
    df.to_csv(os.path.join(save_root, 'patch_label_file.csv'))


def check_patches():
    import sys,os,glob,shutil
    import pandas as pd
    import openslide
    from PIL import Image
    data_roots = {
        'BCSS': './data/BCSS/bcss_512_256_0.8_50_False',
        'PanNuke': './data/PanNuke/',
        'NuCLS': './data/NuCLS/',
        'Kather100K': './data/kather100k'
    }
    if 'CLUSTER_NAME' in os.environ and os.environ['CLUSTER_NAME'] == 'Biowulf':
            data_roots = {
                'BCSS': '/data/zhongz2/temp_BCSS/bcss_512_256_0.8_50_False',
                'PanNuke': '/data/zhongz2/temp_PanNuke/',
                'NuCLS': '/data/zhongz2/temp_NuCLS/',
                'Kather100K': '/data/zhongz2/temp_kather100k'
            }

    for proj_name,data_root in data_roots.items():
        save_dir = os.path.join('./tmp', proj_name)
        os.makedirs(save_dir, exist_ok=True)
        df = pd.read_csv(os.path.join(data_root, 'patch_label_file.csv'))
        for label1, count in df['label'].value_counts().items():
            print(label1, count)
            c = 0
            for rowind, row in df[df['label']==label1].sample(50, replace=True).iterrows():
                print(row['Patch Names'], row['label'])

                patch_name = row['Patch Names']
                label = row['label']
                if proj_name == 'Kather100K':
                    patch = openslide.open_slide(
                        os.path.join(data_root, 'ALL', patch_name))
                    patch_rescaled = patch.read_region(
                        (0, 0), 0, (224, 224)).convert('RGB')
                else:
                    patch = Image.open(os.path.join(data_root, 'ALL', patch_name))
                    patch_rescaled = patch.convert('RGB')
                patch_rescaled.save(os.path.join(save_dir, 'label{}_{}.jpg'.format(label1, c)))
                c += 1
