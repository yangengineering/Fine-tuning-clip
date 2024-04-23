import os.path as osp
from models.MyModel import MyModel
from datasets.datasets import Datasets, MergedDataset
import torch
import torch.nn.functional as F
from utils.utils import *
from datasets.transforms import image_augment
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import csv
from pandas import DataFrame
from cls_names_tmplates import *

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

# cls_name = ['is normal', 'occurs litters', 'occurs persons', 'occurs container broken', 'occurs pits', 
#                'misses tanker covers', 'occurs tarp ropes broken', 'occurs tarp broken', 'occurs side ropes decoupling', 'occurs side ropes broken',
#                'occurs standing water']
               
cls_name = ['normal carriage', 'carriage with litters', 'carriage with persons', 'carriage with broken container', 'carriage with pits', 'carriage without tanker covers', 'carriage with broken tarp ropes', 'carriage with broken tarp', 'carriage with decoupling side ropes ', 'carriage with broken side ropes ', 'carriage with water']
RESULTS_cn = ['无异常', '有杂物', '有闲杂人员', '箱体破损', '车厢凹陷', '顶盖丢失', '绳网破损', '篷布破损', '边绳脱钩', '边绳断裂', '车厢积水']

class MyTrainer():

    def __init__(self, args):
        self.args = args
        init_experiment(args, runner_name=['CLIP'])
        self.csv_saver_path = osp.join(self.args.log_dir, 'results.xlsx')
        self.csv_saver_path_train = osp.join(self.args.log_dir, 'train_results.xlsx')

        # prepare model
        self.model = self.prepare_model()  
        self.model.cuda()

        # encode cls name
        with torch.no_grad():
            tmplates        = ['carriage', 'flatcat', 'openwagon', 'tankcar', 'tanker']
            target_names    = ['{}', '{} with litters', '{} with persons', '{} with broken container',
                            '{} with pits', '{} without tanker covers', '{} with broken tarp ropes', '{} with broken tarp', '{} with decoupling side ropes',
                            '{} with broken side ropes', '{} with water']
            self.text_embs = []
            for tmpl in tmplates:
                cur_types = [cur_type.format(tmpl) for cur_type in target_names]
                text_embs = self.model.forward_text(cur_types)
                self.text_embs.append(text_embs)
            self.text_embs = torch.stack(self.text_embs)
            self.text_embs = self.text_embs.mean(dim=0)
            # self.text_embs = self.model.forward_text(cls_name3)

        # prepare dataloader
        self.train_datasets, self.test_datasets = self.prepare_datasets()
        self.train_loader = self.prepare_dataloader(self.train_datasets)
        self.test_loader = self.prepare_dataloader(self.test_datasets, state='test')
     

    def prepare_model(self):
        model = MyModel(self.args)
        return model


    def prepare_datasets(self):
        transform_train = image_augment()
        transform_test  = image_augment(state='test')
        trd1             = Datasets(file_path=self.args.annotation_train_exception_path, transform=transform_train)
        trd2             = Datasets(file_path=self.args.annotation_train_normal_path, transform=transform_train)
        ted1             = Datasets(file_path=self.args.annotation_test_exception_path, transform=transform_test)
        ted2             = Datasets(file_path=self.args.annotation_test_normal_path, transform=transform_test)
        train_datasets   = MergedDataset(trd1, trd2)
        test_datasets    = MergedDataset(ted1, ted2)
        return train_datasets, test_datasets

        
    def prepare_dataloader(self, datasets, state:str='train'):
        if state == 'train':
            dataloader = DataLoader(datasets, num_workers=self.args.num_workers, batch_size=self.args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        else:
            dataloader = DataLoader(datasets, num_workers=self.args.num_workers, batch_size=self.args.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        return dataloader


    def prepare_optimizer(self):
        optim_target = [{'params': self.model.parameters(),'lr':self.args.lr}]
        # freeze encoder
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        # select optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optim_target, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_target, weight_decay=self.args.wd)
        else:
            raise ValueError(f"Invalid optimizer {self.args.optimizer}")

        # select scheduler
        if self.args.scheduler == 'SLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                                    optimizer, step_size=self.args.steps, gamma=self.args.gamma)
        elif self.args.scheduler == 'MSLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epoch)
        elif self.args.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        else:
            raise ValueError(f"Invalid scheduler {self.args.scheduler}")
        
        return optimizer, scheduler


    def train_one_epoch(self, epoch):
        all_preds  = []
        all_gts    = []
        for i, batch in enumerate(tqdm(self.train_loader)):
            data, true_label = [_.cuda() for _ in batch] 
            logits = self.model(data, self.text_embs)
            # print(logits)
            # print(true_label)
            # print((self.args.temperature * logits))
            loss   = F.cross_entropy(self.args.temperature * logits, true_label)
            preds            = logits.argmax(dim=-1)
            all_preds.append(preds)
            all_gts.append(true_label)
            if i % 5 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.args.logger.info('lr is {}'.format(current_lr))
                self.args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}'.format(epoch, i, len(self.train_loader), loss.item()))
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        all_preds   = torch.cat(all_preds, dim=-1)
        all_gts     = torch.cat(all_gts, dim=-1)

        # overall accuracy
        overall_acc = 100 * all_preds.eq(all_gts).float().mean().item()
        self.args.logger.info('train accuracy is {}'.format(overall_acc))


    
    def train(self):
        # prepare optimizer
        self.optimizer, self.scheduler = self.prepare_optimizer()

        max_acc = 0
        for epoch in range(1, self.args.epoch+1):
            self.model.train()
            self.train_one_epoch(epoch)
            acc, all_preds, all_gts, write_csv = self.test()

            # save best model
            if acc > max_acc:
                max_acc = acc
                state_dict_best = {'model': self.model.state_dict()}
                torch.save(state_dict_best, self.args.best_model_path)
                self.args.logger.info('Best overall accuracy is {}'.format(acc))
                self.acc_of_each_type_of_train(all_preds, all_gts, write_csv=write_csv)
                self.acc_of_each_type(all_preds, all_gts, write_csv=write_csv)
            self.args.logger.info('Epoch:[{}]\tAccuracy:[{}]'.format(epoch, acc))

            # save checkpoint
            state_dict = {
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'lr_schedule':self.scheduler.state_dict(),
                            'epoch': epoch + 1,
                            'best_acc': max_acc,
                        }
            torch.save(state_dict, self.args.model_path)

            self.scheduler.step()

    
    @torch.no_grad()
    def test(self, load_best:bool=False, zero_shot:bool=False, write_csv:bool=False):
        if load_best:
            self.model = load_trained_paras(self.args.best_model_path, [self.model], ['model'])[0]
        self.model.eval()
        all_preds  = []
        all_gts    = []
        for i, batch in enumerate(tqdm(self.test_loader)):
            data, true_label = [_.cuda() for _ in batch] 
            if zero_shot:
                img_features     = self.model.forward_image(data)
                logits           = F.linear(F.normalize(img_features, dim=-1, p=2), F.normalize(self.text_embs, dim=-1, p=2))
            else:
                logits           = self.model(data, self.text_embs)

            preds            = logits.argmax(dim=-1)
            all_preds.append(preds)
            all_gts.append(true_label)
        all_preds   = torch.cat(all_preds, dim=-1)
        all_gts     = torch.cat(all_gts, dim=-1)

        # overall accuracy
        overall_acc = 100 * all_preds.eq(all_gts).float().mean().item()
        self.ovearall_acc=overall_acc

        # further analysis
        # self.acc_of_each_type(all_preds, all_gts, write_csv)
        
        return  overall_acc, all_preds, all_gts, write_csv
    

    def acc_of_each_type(self, all_preds, all_gts, write_csv:bool=False):
        """
        The col denotes the gt
        The row denotes the pred
        """
        write_csv = True
        path=''
        # transfer labels to numpy format
        all_preds_numpy     = all_preds.cpu().numpy()
        all_gts_numpy       = all_gts.cpu().numpy()

        # get unique lables
        uni_label           = np.unique(all_gts_numpy)
        type_len            = len(uni_label)

        # init a confusion matrix
        cls_matrix                      = np.zeros((type_len, type_len)).astype(int)
        instances_gts_overall_each_type = dict(zip(np.arange(type_len), np.zeros(type_len).astype(int)))

        # assign values to the confusion matrix based on the predctions and gts
        for i, pred in enumerate(all_preds_numpy):
            cord_x                      = int(pred)
            cord_y                      = int(all_gts_numpy[i])
            cls_matrix[cord_x, cord_y]  += 1
            instances_gts_overall_each_type[cord_y] += 1
        path='test_confusion_matrix.png'
        self.plot_confusion_matrix(cls_matrix, uni_label, path)
        # calc precision and recall
        precisions = []
        recalls    = []
        for i in range(type_len):
            precision = cls_matrix[i, i] / cls_matrix[i].sum()
            recall     = cls_matrix[i, i] / instances_gts_overall_each_type[i]
            precisions.append(precision)
            recalls.append(recall)
        
        # write results to csv file 
        if write_csv:
            data = {
                "type":RESULTS_cn,
                'precision':precisions,
                'recall': recalls
            }
            df = DataFrame(data)
            df.to_excel(self.csv_saver_path)
            # csv_saver = csv.writer(open(self.csv_saver_path, 'w', encoding='utf-8', newline=""))
            # csv_saver.writerow(['type', 'precision', 'recall'])
            # for i in range(type_len):
            #     type_name = RESULTS_cn[i]
            #     precision = precisions[i]
            #     recall = recalls[i]
            #     csv_saver.writerow([type_name, '{:.2f}'.format(precision), '{:.2f}'.format(recall)])
      
        return precisions, recalls

    def acc_of_each_type_of_train(self, all_preds, all_gts, write_csv:bool=False):
        """
        The col denotes the gt
        The row denotes the pred
        """
        write_csv = True
        # transfer labels to numpy format
        all_preds_numpy     = all_preds.cpu().numpy()
        all_gts_numpy       = all_gts.cpu().numpy()

        # get unique lables
        uni_label           = np.unique(all_gts_numpy)
        type_len            = len(uni_label)

        # init a confusion matrix
        cls_matrix                      = np.zeros((type_len, type_len)).astype(int)
        instances_gts_overall_each_type = dict(zip(np.arange(type_len), np.zeros(type_len).astype(int)))

        # assign values to the confusion matrix based on the predctions and gts
        for i, pred in enumerate(all_preds_numpy):
            cord_x                      = int(pred)
            cord_y                      = int(all_gts_numpy[i])
            cls_matrix[cord_x, cord_y]  += 1
            instances_gts_overall_each_type[cord_y] += 1
        path='train_confusion_matrix.png'
        self.plot_confusion_matrix(cls_matrix, uni_label, path)

        # calc precision and recall
        precisions = []
        recalls    = []
        for i in range(type_len):
            precision = cls_matrix[i, i] / cls_matrix[i].sum()
            recall     = cls_matrix[i, i] / instances_gts_overall_each_type[i]
            precisions.append(precision)
            recalls.append(recall)

        # write results to csv file 
        if write_csv:
            data = {
                "type":RESULTS_cn,
                'precision':precisions,
                'recall': recalls
            }
            df = DataFrame(data)
            df.to_excel(self.csv_saver_path_train)
            # csv_saver = csv.writer(open(self.csv_saver_path, 'w', encoding='utf-8', newline=""))
            # csv_saver.writerow(['type', 'precision', 'recall'])
            # for i in range(type_len):
            #     type_name = RESULTS_cn[i]
            #     precision = precisions[i]
            #     recall = recalls[i]
            #     csv_saver.writerow([type_name, '{:.2f}'.format(precision), '{:.2f}'.format(recall)])
      
        return precisions, recalls

    def plot_confusion_matrix(self, confusion_matrix, labels, path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt="g", xticklabels=labels, yticklabels=labels)
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("混淆矩阵")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        # 保存混淆矩阵图片
        plt.savefig(path)


    def save_chekpoints(self):
        pass
    

    def load_checkpoint(self):
        pass

    
    def write_csv(self):
        pass
    