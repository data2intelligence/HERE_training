import os
import torch
from torch.utils.data import Dataset
from common import CLASSIFICATION_DICT, REGRESSION_LIST, LABEL_COL_DICT


class PatchDatasetV2(Dataset):
    def __init__(self, slide, coords, patch_level, patch_size):
        self.slide = slide
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        coord = self.coords[index]
        patch = self.slide.read_region(location=(int(coord[0]), int(coord[1])), level=self.patch_level,
                                       size=(self.patch_size, self.patch_size)).convert('RGB')  # BGR
        return {
            'pixel_values': patch,
            'coords': coord
            }


class HistoDataset(Dataset):
    def __init__(self, df, feats_dir):
        super().__init__()
        self.feats_dir = feats_dir  
        self.epoch = -1 

        slide_data = df.copy()

        valid_classification_dict = {}
        for cls_task_name in CLASSIFICATION_DICT.keys():
            if LABEL_COL_DICT[cls_task_name] in slide_data.columns:
                valid_classification_dict[cls_task_name] = CLASSIFICATION_DICT[cls_task_name]
        self.classification_dict = valid_classification_dict

        valid_regression_list = []
        for reg_task_name in REGRESSION_LIST:
            if reg_task_name in slide_data.columns:
                valid_regression_list.append(reg_task_name)
        self.regression_list = valid_regression_list

        self.slide_data = slide_data

    def __len__(self):
        return len(self.slide_data)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):

        item = self.slide_data.iloc[index]
        svs_filename = item['DX_filename']
        svs_prefix = os.path.basename(svs_filename).replace('.svs', '')

        patches = torch.load(os.path.join(self.feats_dir, svs_prefix + '.pt')).float()

        labels_dict = {
            'svs_filename': svs_filename,
            'PanCancerSiteID': int(item['PanCancerSiteID'])
        }

        for cls_task_name in CLASSIFICATION_DICT.keys():
            if LABEL_COL_DICT[cls_task_name] in item:
                labels_dict[cls_task_name] = int(item[LABEL_COL_DICT[cls_task_name]])
        for reg_task_name in REGRESSION_LIST:
            if reg_task_name in item:
                labels_dict[reg_task_name] = item[reg_task_name].astype('float32')

        return patches, labels_dict

