import os, glob, json, pickle, random, argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from common import CLASSIFICATION_DICT, REGRESSION_LIST, PAN_CANCER_SITES, IGNORE_INDEX_DICT, LABEL_COL_DICT
from utils import Accuracy_Logger, Regression_Logger
from model import AttentionModel
from dataset import HistoDataset



def get_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', default='False', type=str)
    parser.add_argument('--combine_train_val_splits', type=str, default="True")  
    parser.add_argument('--combine_train_val_test_splits', type=str, default="False") 
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--split_num', type=int, default=0) 
    parser.add_argument('--feats_dir', default='', type=str)
    parser.add_argument('--accum_iter', default=4, type=int)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--save_root', type=str, default='.')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--regu_loss_type', type=str, default='None')
    parser.add_argument('--backbone', type=str, default='PLIP')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--outside_test_filenames',  default="", type=str)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def ddp_setup():
    dist.init_process_group(backend="nccl")


def collect_results_gpu(part_tensor, size, world_size):
    shape = part_tensor.shape
    shape_tensor = torch.tensor(shape[0], device=part_tensor.device)
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    if len(shape) == 1:
        part_send = torch.zeros(shape_max, dtype=part_tensor.dtype, device=part_tensor.device)
        part_send[:shape_tensor] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        dist.all_gather(part_recv_list, part_send)
    if len(shape) == 2:
        part_send = torch.zeros((shape_max, shape[1]), dtype=part_tensor.dtype, device=part_tensor.device)
        part_send[:shape_tensor] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max, shape[1]) for _ in range(world_size)
        ]
        dist.all_gather(part_recv_list, part_send)
    return torch.cat(part_recv_list, axis=0)[:size]


class Trainer:
    def __init__(
            self,
            model,
            dataloaders,
            optimizer,
            class_weights_dict,
            args
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.model = model.to(self.gpu_id)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = os.path.join(args.save_root, 'snapshot.pt') 
        self.args = args
        self.accum_iter = args.accum_iter

        if self.gpu_id == 0:
            with open(os.path.join(args.save_root, 'args.txt'), 'w') as fp:
                json.dump(args.__dict__, fp, indent=2)

        cls_loss_fn_dict, reg_loss_fn_dict = \
            get_loss_fn_dict(class_weights_dict, device=self.gpu_id)
        self.cls_loss_fn_dict = cls_loss_fn_dict
        self.reg_loss_fn_dict = reg_loss_fn_dict

        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.loss_dicts = {subset: [] for subset in self.dataloaders.keys()}
        self.loss_dicts_per_cancer = {}
        for site_id, site_name in enumerate(PAN_CANCER_SITES):
            self.loss_dicts_per_cancer[site_name] = {subset: [] for subset in self.dataloaders.keys()}

        if self.gpu_id == 0:
            save_dirs = {}
            for subset in self.dataloaders.keys():
                save_dirs[subset] = os.path.join(args.save_root, subset)
                os.makedirs(save_dirs[subset], exist_ok=True)

                if self.dataloaders[subset] is not None:
                    self.dataloaders[subset].dataset.slide_data.to_csv(
                        os.path.join(save_dirs[subset], 'slide_data.csv'.format(subset)))

                for site_id, site_name in enumerate(PAN_CANCER_SITES):
                    os.makedirs(os.path.join(save_dirs[subset], site_name), exist_ok=True)
            self.save_dirs = save_dirs

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch, subset='train'):

        is_train = subset == 'train'
        if is_train:
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        b_sz = len(next(iter(self.dataloaders[subset]))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloaders[subset])}")
        self.dataloaders[subset].sampler.set_epoch(epoch)
        dataset = self.dataloaders[subset].dataset
        results = {}
        labels = {}
        all_svs_filenames = []
        for batch_idx, (patches, labeled_batch) in enumerate(self.dataloaders[subset]):
            patches = patches.to(self.gpu_id)
            # all_svs_filenames += labeled_batch['svs_filename']
            for k in labeled_batch.keys():
                if k != 'svs_filename':
                    labeled_batch[k] = labeled_batch[k].to(self.gpu_id)
                    if k in labels:
                        labels[k].append(labeled_batch[k].detach().clone())
                    else:
                        labels[k] = [labeled_batch[k].detach().clone()]

            if is_train:
                results_dicts = self.model(patches)
            else:
                with torch.no_grad():
                    results_dicts = self.model(patches)

            for k, v in results_dicts.items():
                if k in results:
                    results[k].append(v.detach().clone())
                else:
                    results[k] = [v.detach().clone()]

            classification_losses = {}
            for k in CLASSIFICATION_DICT.keys():
                if k in labeled_batch:
                    classification_losses[k] = self.cls_loss_fn_dict[k](results_dicts[k + '_logits'], labeled_batch[k])

            regression_losses = {}
            for k in REGRESSION_LIST:
                if k in labeled_batch:
                    regression_losses[k] = self.reg_loss_fn_dict[k](results_dicts[k + '_logits'], labeled_batch[k])

            total_loss = sum([v for vi, v in enumerate(classification_losses.values())]) + \
                         sum([v for vi, v in enumerate(regression_losses.values())])

            if is_train:

                total_loss = total_loss / self.accum_iter
                total_loss.backward()

                if (batch_idx + 1) % self.accum_iter == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

        for k in results.keys():
            results[k] = torch.cat(results[k])
        for k in labels.keys():
            labels[k] = torch.cat(labels[k])

        all_results = {}
        for k in results.keys():
            all_results[k] = collect_results_gpu(results[k], len(dataset), world_size=self.world_size)
        all_labels = {}
        for k in labels.keys():
            all_labels[k] = collect_results_gpu(labels[k], len(dataset), world_size=self.world_size)

        if self.gpu_id == 0:
            loss_dict = self._save_results(epoch, all_labels, all_results, dataset, save_dir=self.save_dirs[subset],
                                           writer=None, subset=subset)
            self.loss_dicts[subset].append(loss_dict)

            for site_id, site_name in enumerate(PAN_CANCER_SITES):
                inds = torch.where(all_labels['PanCancerSiteID'] == site_id + 1)[0]
                tmp_labels = {k: v[inds] for k, v in all_labels.items()}
                tmp_results = {k: v[inds] for k, v in all_results.items()}
                loss_dict = self._save_results(epoch, tmp_labels, tmp_results, dataset,
                                                save_dir=os.path.join(self.save_dirs[subset], site_name),
                                                writer=None, subset=subset)
                self.loss_dicts_per_cancer[site_name][subset].append(loss_dict)

            for subset, v in self.loss_dicts.items():
                if len(v) > 0:
                    val_log_df = pd.DataFrame(v)
                    val_log_df.to_csv(os.path.join(self.args.save_root, '{}_e{}_log.csv'.format(subset, epoch)))
            for site_id, site_name in enumerate(PAN_CANCER_SITES):
                for subset, v in self.loss_dicts_per_cancer[site_name].items():
                    if len(v) > 0:
                        val_log_df = pd.DataFrame(v)
                        val_log_df.to_csv(
                            os.path.join(self.save_dirs[subset], site_name, '{}_e{}_log.csv'.format(subset, epoch)))

        dist.barrier()

    def _save_results(self, epoch, all_labels, all_results, dataset, save_dir, writer=None, subset='val'):
        losses_dict = {
            'bce': 0., 'kld': 0.,
            'surv': 0., 'regu': 0.
        }
        for k in CLASSIFICATION_DICT.keys():
            losses_dict[k] = 0.
        for k in REGRESSION_LIST:
            losses_dict[k] = 0.

        loggers_dict = {}
        cls_invalids = {k: False for k in CLASSIFICATION_DICT.keys()}
        for k, v in CLASSIFICATION_DICT.items():
            if k not in dataset.classification_dict:
                cls_invalids[k] = True
                continue
            Y = all_labels[k]
            if len(Y[torch.where(Y != IGNORE_INDEX_DICT[k])[0]].unique()) < 2:
                cls_invalids[k] = True
                continue
            loggers_dict[k] = Accuracy_Logger(n_classes=len(v), task_name=k, label_names=v,
                                              ignore_label_ind=IGNORE_INDEX_DICT[k])
            Y_hat_k = torch.topk(all_results[k + '_logits'], 1, dim=1)[1].squeeze(1)
            Y_prob_k = F.softmax(all_results[k + '_logits'], dim=1)
            loggers_dict[k].log(Y_hat_k, Y, Y_prob_k)
            losses_dict[k] += self.cls_loss_fn_dict[k](all_results[k + '_logits'], Y).item()

        reg_loggers_dict = {}
        reg_invalids = {k: False for k in REGRESSION_LIST}
        for k in REGRESSION_LIST:
            if k not in dataset.regression_list:
                reg_invalids[k] = True
                continue
            Y = all_labels[k]
            if len(Y) < 2:
                reg_invalids[k] = True
                continue
            reg_loggers_dict[k] = Regression_Logger()
            reg_loggers_dict[k].log(all_results[k + '_logits'], Y)
            losses_dict[k] += self.reg_loss_fn_dict[k](all_results[k + '_logits'], Y).item()

        for name, labels in CLASSIFICATION_DICT.items():
            if name not in dataset.classification_dict:
                continue
            if cls_invalids[name]:
                continue
            loggers_dict[name].set_save_filename(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_data.txt'.format(epoch, name, subset)))

            for average in ['micro', 'macro', 'weighted']:
                score = loggers_dict[name].get_f1_score(average=average)
                losses_dict['{}_f1_{}'.format(name, average)] = score
                if writer:
                    writer.add_scalar('{}/{}_f1_{}'.format(subset, name, average), score, epoch)

            for average in ['macro', 'weighted']:
                auc = loggers_dict[name].get_auc_score(average=average)
                losses_dict['{}_auc_{}'.format(name, average)] = auc
                if writer:
                    writer.add_scalar('{}/{}_auc_{}'.format(subset, name, average), auc, epoch)

            loggers_dict[name].get_roc_curve(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_ROC.jpg'.format(epoch, name, subset)))

            loggers_dict[name].save_data(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_data.txt'.format(epoch, name, subset)))

            loggers_dict[name].get_classification_report(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_report.txt'.format(epoch, name, subset)))

            for j in range(len(labels)):
                acc, correct, count = loggers_dict[name].get_summary(j)
                losses_dict['{}_{}_acc'.format(name, CLASSIFICATION_DICT[name][j])] = acc
                losses_dict['{}_{}_correct'.format(name, CLASSIFICATION_DICT[name][j])] = correct
                losses_dict['{}_{}_count'.format(name, CLASSIFICATION_DICT[name][j])] = count
                if writer:
                    writer.add_scalar('{}/{}_{}_{}_acc'.format(subset, name, j, CLASSIFICATION_DICT[name][j]), acc,
                                      epoch)

        for name in REGRESSION_LIST:
            if name not in dataset.regression_list:
                continue
            if reg_invalids[name]:
                continue
            mse = reg_loggers_dict[name].mean_squared_error()
            losses_dict['{}_mse'.format(name)] = mse
            if writer:
                writer.add_scalar('{}/{}_mse'.format(subset, name), mse, epoch)

            reg_metrics_dict = reg_loggers_dict[name].compute_metrics()
            for metric_name, metric_val in reg_metrics_dict.items():
                losses_dict['{}_{}'.format(name, metric_name)] = metric_val
                if writer:
                    writer.add_scalar('{}/{}_{}'.format(subset, name, metric_name), metric_val, epoch)

        losses_dict['epoch'] = epoch
        losses_dict['c_index'] = 0
        losses_dict['surv_iAUC'] = 0

        return losses_dict

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path.replace('.pt', '_{}.pt'.format(epoch)))
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):

            for subset in self.dataloaders.keys():
                if self.dataloaders[subset] is not None:
                    self._run_epoch(epoch, subset)

            if self.gpu_id == 0 and epoch % args.save_every == 0:
                self._save_snapshot(epoch)

        if self.gpu_id == 0:
            for subset, v in self.loss_dicts.items():
                if len(v) > 0:
                    val_log_df = pd.DataFrame(v)
                    val_log_df.to_csv(os.path.join(self.args.save_root, '{}_log.csv'.format(subset)))
            for site_id, site_name in enumerate(PAN_CANCER_SITES):
                for subset, v in self.loss_dicts_per_cancer[site_name].items():
                    if len(v) > 0:
                        val_log_df = pd.DataFrame(v)
                        val_log_df.to_csv(os.path.join(self.save_dirs[subset], site_name, '{}_log.csv'.format(subset)))


def get_loss_fn_dict(class_weights_dict, device):
    cls_loss_fn_dict = {}
    for k in CLASSIFICATION_DICT.keys():
        cls_loss_fn_dict[k] = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_DICT[k],
                                                  weight=class_weights_dict[k].to(device))
    reg_loss_fn_dict = {k: nn.MSELoss() for k in REGRESSION_LIST}

    return cls_loss_fn_dict, reg_loss_fn_dict


def get_class_weights(train_datasets, num_classes, label_col):
    Ns = 0
    slide_cls_ids = [0 for _ in range(num_classes)]
    for train_dataset in train_datasets:
        Ns += float(len(train_dataset))
        for i in range(num_classes):
            slide_cls_ids[i] += len(np.where(train_dataset.slide_data[label_col] == i)[0])
    weight_per_class = [Ns / slide_cls_ids[c] if slide_cls_ids[c] > 0 else 0 for c in range(num_classes)]
    class_weights = torch.FloatTensor(weight_per_class)
    return class_weights


def clean_csv(slide_data, feats_dir):
    # remove invalid files
    invalid_CLIDs = {'TCGA-5P-A9KA', 'TCGA-5P-A9KC', 'TCGA-HT-7483', 'TCGA-UZ-A9PQ'}
    existed_prefixes = set([os.path.basename(f).replace('.pt', '') for f in glob.glob(os.path.join(feats_dir, '*.pt'))])
    invalid_inds = []
    for ind, DX_filename in enumerate(slide_data['DX_filename'].values):
        fileprefix = os.path.basename(DX_filename).replace('.svs', '')
        if fileprefix[:12] in invalid_CLIDs or fileprefix not in existed_prefixes:
            invalid_inds.append(ind)
    if len(invalid_inds) > 0:
        slide_data = slide_data.drop(invalid_inds).reset_index(drop=True)

    for k, v in CLASSIFICATION_DICT.items():
        if k in slide_data:
            slide_data[k] = slide_data[k].fillna(IGNORE_INDEX_DICT[k])
    for k in REGRESSION_LIST:
        if k in slide_data:
            slide_data = slide_data[slide_data[k].isnull() == False]

    slide_data = slide_data.reset_index(drop=True)

    return slide_data

def load_train_objs(args):

    trainset = pd.read_csv(args.trainset_filename, low_memory=False)
    valset = pd.read_csv(args.valset_filename, low_memory=False)
    testset = pd.read_csv(args.testset_filename, low_memory=False)

    outside_testset_csvs = {
        'outside_test{}'.format(ii): pd.read_csv(filename, low_memory=False)
        for ii, filename in enumerate(args.outside_test_filenames.split('|'))
    }

    trainset = clean_csv(trainset, args.feats_dir)
    valset = clean_csv(valset, args.feats_dir)
    testset = clean_csv(testset, args.feats_dir)

    if args.debug == 'True':
        trainset = valset
        valset = testset
    else:
        if args.combine_train_val_splits == 'True':  # combine train and val split for train the final model
            trainset = pd.concat([trainset, valset], axis=0).reset_index(drop=True)
            valset = testset.copy()
        if args.combine_train_val_test_splits == 'True':  # combine train/val/test splits for train the final model
            trainset = pd.concat([trainset, valset, testset], axis=0).reset_index(drop=True)
            valset = testset.copy()

    print('final_trainset: {}'.format(len(trainset)))
    print('final_valset: {}'.format(len(valset)))
    print('final_testset: {}'.format(len(testset)))

    train_dataset = HistoDataset(df=trainset, feats_dir=args.feats_dir)
    val_dataset = HistoDataset(df=valset, feats_dir=args.feats_dir)
    test_dataset = HistoDataset(df=testset, feats_dir=args.feats_dir)
    outside_test_datasets = {}
    for key, testset2 in outside_testset_csvs.items():
        outside_test_datasets[key] = HistoDataset(df=testset2, feats_dir=args.feats_dir)

    class_weights_dict = {k: get_class_weights([train_dataset], len(CLASSIFICATION_DICT[k]), LABEL_COL_DICT[k])
                            for k in CLASSIFICATION_DICT.keys()}

    model = AttentionModel(backbone=args.backbone, dropout=args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, \
        weight_decay=args.weight_decay if args.regu_loss_type == 'None' else 0)

    return train_dataset, val_dataset, test_dataset, outside_test_datasets, \
        model, optimizer, class_weights_dict


def train_main(args):
    ddp_setup()
    train_dataset, val_dataset, test_dataset, outside_test_datasets, model, optimizer, class_weights_dict = load_train_objs(args)

    dataloaders = {
        'train':
            DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False)),
        'val':
            DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False)),
        'test':
            DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(test_dataset, shuffle=False, drop_last=False))
    }
    for key, outside_test_dataset in outside_test_datasets.items():
        dataloaders[key] = DataLoader(outside_test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(outside_test_dataset, shuffle=False, drop_last=False))
    trainer = Trainer(model, dataloaders, optimizer, class_weights_dict, args)
    trainer.train(args.max_epochs)
    dist.destroy_process_group()


if __name__ == "__main__":

    args = get_args()

    setup_seed(args.seed)

    args.__dict__['trainset_filename'] = 'splits/train-{}.csv'.format(args.split_num)
    args.__dict__['valset_filename'] = 'splits/val-{}.csv'.format(args.split_num)
    args.__dict__['testset_filename'] = 'splits/test-{}.csv'.format(args.split_num)

    train_main(args)
