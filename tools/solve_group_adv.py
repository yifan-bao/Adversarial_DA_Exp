import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import ceil, floor
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch.utils.data as data
from torch.autograd import Variable

import sys
sys.path.append(os.path.abspath('.'))
from utils.loss import *
from datasets.bewteencity_Dataset import Group_Dataset

from tools.train_source import *

from graphs.models.discriminator import FCDiscriminator

class UDATrainer(Trainer):
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)
        
        source_data_set = Group_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split=args.source_split,
                                    base_size=args.base_size,
                                    crop_size=args.crop_size,
                                    class_16=args.class_16,
                                    group=0) 

        self.source_dataloader = data.DataLoader(source_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        # split = 'val
        source_data_set = Group_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split='val',
                                    base_size=args.base_size,
                                    crop_size=args.crop_size,
                                    class_16=args.class_16,
                                    group=0)
        self.source_val_dataloader = data.DataLoader(source_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)

        target_data_set = Group_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.target_base_size,
                                crop_size=args.target_crop_size,
                                class_16=args.class_16,
                                group=1)
        self.target_dataloader = data.DataLoader(target_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        
        # 这里修改了下总的迭代次数-应该是按target来的把-也不明白为啥前面不行
        self.dataloader.num_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        # val 
        target_data_set = Group_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split='val',
                                base_size=args.target_base_size,
                                crop_size=args.target_crop_size,
                                class_16=args.class_16,
                                group=1)
        
        self.target_val_dataloader = data.DataLoader(target_data_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)

        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num


        # 对抗训练相关模型初始化
        # init D
        self.model_D1 = FCDiscriminator(num_classes=self.args.num_classes)
        self.model_D2 = FCDiscriminator(num_classes=self.args.num_classes)
        # 注意这里是按multi的形式-有两个输出

        # D的优化器初始化
        self.optimizer_D1 = torch.optim.Adam(self.model_D1.parameters(), lr=self.args.learning_rate_D, betas=(0.9, 0.99))
        self.optimizer_D2 = torch.optim.Adam(self.model_D2.parameters(), lr=self.args.learning_rate_D, betas=(0.9, 0.99))
        # 注意增加args.learning_rate_D
        
        # 判别器训练loss函数
        # if gan == 'Vanilla'
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # if gan == 'LS'
        # bce_loss = torch.nn.MSELoss()

        # labels for adversarial training
        self.source_label = 0
        self.target_label = 1
        
    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))
        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint - 一般是都要有的-之前预训练目的就是这个
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)
        
        if not self.args.continue_training:
            self.best_MIou = 0
            self.best_iter = 0
            self.current_iter = 0
            self.current_epoch = 0

        # 这个是当前的预适配的模型继续-前面的是预训练模型-在source domain上train的
        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))

        self.args.iter_max = self.dataloader.num_iterations*self.args.epoch_each_round*self.round_num
        print(self.args.iter_max, self.dataloader.num_iterations)

        self.train_round()

        self.writer.close()
    def train_round(self):
        for r in range(self.current_round, self.round_num):
            print("\n############## Begin {}/{} Round! #################\n".format(self.current_round+1, self.round_num))
            print("epoch_each_round:", self.args.epoch_each_round)
            
            self.epoch_num = (self.current_round+1)*self.args.epoch_each_round

            # generate threshold
            self.threshold = self.args.threshold

            self.train() # train的内容就是train one epoch - 然后validate部分应该也得修改下 
            # train部分要看下-可能要重载下

            self.current_round += 1
        
    
    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch+1, self.epoch_num))
        self.logger.info("Training one epoch...")
        self.Eval.reset()

        # Initialize your average meters
        # 这部分内容一个epoch输出一次 
        loss_seg_value1 = 0  # 分割loss
        loss_adv_target_value1 = 0
        loss_D_value1 = 0    # 判别器loss

        # 第二部分是multi部分-看情况有或没有-注意修改
        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0 

        # 分割模型训练模式
        self.model.train()    

        # 判别器训练模式
        self.model_D1.train()
        self.model_D1.to(self.device)
        self.model_D2.train()
        self.model_D2.to(self.device)
        
        iter_num = self.dataloader.num_iterations # 训练次数
        batch_idx = 0
        
        # zip 命令--一次可以拿两个域的数据
        for batch_s, batch_t in tqdm_epoch:
            # 学习率调整
            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)                
            self.poly_lr_scheduler(optimizer=self.optimizer_D1, init_lr=self.args.learning_rate_D)
            self.writer.add_scalar('learning_rate_D', self.optimizer_D1.param_groups[0]["lr"], self.current_iter) 
            
            # 梯度清零
            self.optimizer.zero_grad()
            self.optimizer_D1.zero_grad()
            self.optimizer_D2.zero_grad()

            # train G
            # 关闭D训练
            for param in self.model_D1.parameters():
                param.requires_grad = False

            for param in self.model_D2.parameters():
                param.requires_grad = False
            
                # train with source
            x, y, _ = batch_s
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)
            
            pred = self.model(x)
            # pred 为tuple 就是multi方式
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]
            y = torch.squeeze(y, 1)

            # 分割loss
            loss = self.loss(pred, y) # 最终结果的分割loss

            loss_ = loss
            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y) # middle output seg loss
                loss_ += loss_2
                loss_seg_value2 += loss_2.cpu().item() / iter_num   # middle output seg loss for one epoch 
            
            loss_.backward() # 整合了middle和最终输出的loss bp
            loss_seg_value1 += loss.cpu().item() / iter_num # output seg loss for one epoch
            ### 注意pred是output 而pred2是middle的-lambda_seg乘以的是middl部分-一般为0.1
            
            # train with target
            x, _, _ = batch_t
            if self.cuda:
                x = Variable(x).to(self.device)
            pred_target = self.model(x)
            if isinstance(pred_target, tuple):
                pred_target2 = pred_target[1]
                pred_target = pred_target[0]
                pred_P_2 = F.softmax(pred_target2, dim=1)

            pred_P = F.softmax(pred_target, dim=1)
            # pred_P是output 而pred_P_2是middle

            D_out1 = self.model_D1(pred_P)
            # 对抗loss
            loss_adv_target1 = self.bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(self.source_label)).to(self.device))
            
            loss_ = loss_adv_target1 * self.args.lambda_adv_target1 # 记得增加上去 
            if self.args.multi:
                D_out2 = self.model_D2(pred_P_2)
                loss_adv_target2 = self.bce_loss(D_out2, Variable(torch.FloatTensor(D_out1.data.size()).fill_(self.source_label)).to(self.device))
                loss_ += loss_adv_target2 * self.args.lambda_adv_target2 
                loss_adv_target_value2 += loss_adv_target2.cpu().item() / iter_num
            
            loss_adv_target_value1 += loss_adv_target1.cpu().item() / iter_num
            
            loss_.backward()  # loss_ = loss_adv_target1 * self.args.lambda_adv_target1 + loss_adv_target2 * self.args.lambda_adv_target2
            

            # train D
            # bring back requires_grad
            for param in self.model_D1.parameters():
                param.requires_grad = True

            for param in self.model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred.detach()  # 取消梯度传递分割模型
            D_out1 = self.model_D1(F.softmax(pred1,dim=1))
            loss_D1 = self.bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(self.source_label)).to(self.device))
            loss_D1 = loss_D1 / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.cpu().item() / iter_num

            if self.args.multi:
                pred2 = pred_2.detach()
                D_out2 = self.model_D2(F.softmax(pred2, dim=1))
                loss_D2 = self.bce_loss(D_out1, Variable(torch.FloatTensor(D_out2.data.size()).fill_(self.source_label)).to(self.device))
                loss_D2 = loss_D2 / 2
                loss_D2.backward()  
                loss_D_value2 += loss_D2.cpu().item() / iter_num 
                # 只有multi的时候才训练第二个判别器-否则只用第一个就行了

            # train with target
            pred_target = pred_target.detach()
            D_out1 = self.model_D1(F.softmax(pred_target,dim=1))
            loss_D1 = self.bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(self.source_label)).to(self.device))
            loss_D1 = loss_D1 / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.cpu().item() / iter_num

            if self.args.multi:    
                pred_target2 = pred_target2.detach()
                D_out2 = self.model_D2(F.softmax(pred_target2,dim=1))
                loss_D2 = self.bce_loss(D_out2, Variable(torch.FloatTensor(D_out1.data.size()).fill_(self.source_label)).to(self.device))
                loss_D2 = loss_D2 / 2
                loss_D2.backward()
                loss_D_value2 += loss_D2.cpu().item() / iter_num

            self.optimizer.step()
            self.optimizer_D1.step()
            if self.args.multi:
                self.optimizer_D2.step()
            # ok
            if batch_idx % 400 == 0:
                if self.args.multi:
                    self.logger.info("epoch{}-batch-{}:loss_seg1={:.3f}-loss_adv1={:.3f}-loss_D1={:.3f}; loss_seg_2={:.3f}-loss_adv2={:3f}-loss_D2={:.3f}".format(self.current_epoch, 
                    batch_idx,loss.item(),loss_adv_target1.item(),loss_D1.item(),loss_2.item(),loss_adv_target2.item(),loss_D2.item()))
                    
                else:
                    self.logger.info("epoch{}-batch-{}:loss_seg1={:.3f}-loss_adv1={:.3f}-loss_D1={:.3f}".format(self.current_epoch, 
                    batch_idx,loss.item(),loss_adv_target1.item(),loss_D1.item()))
            
            batch_idx += 1
            self.current_iter += 1
            # end of batch
        
        # end of epoch
        self.writer.add_scalar('train_loss',loss_seg_value1, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, loss_seg_value1))
        self.writer.add_scalar('adv_target_loss',loss_adv_target_value1, self.current_epoch)
        tqdm.write("The average adv_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_adv_target_value1))
        self.writer.add_scalar('D_loss',loss_D1,self.current_epoch)
        tqdm.write("The average D_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_D1))
        if self.args.multi:
            self.writer.add_scalar('train_loss_2', loss_seg_value2, self.current_epoch)
            tqdm.write("The average loss_2 of train epoch-{}-:{}".format(self.current_epoch, loss_seg_value2))
            self.writer.add_scalar('adv_loss_2', loss_adv_target_value2, self.current_epoch)
            tqdm.write("The average adv_loss_2 of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_adv_target_value2))
            self.writer.add_scalar('D_loss_2',loss_D2,self.current_epoch)
            tqdm.write("The average D_loss_2 of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_D2))
        # 一共6组loss显示数据 - 每个epoch显示平均值
        tqdm_epoch.close()
        
        self.validate_source()

def add_UDA_train_args(arg_parser):
    arg_parser.add_argument('--source_dataset', default='synthia', type=str,
                            choices=['group0', 'synthia'],
                            help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str,
                            help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0, 
                            help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=1,
                            help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2,
                            help="epoch each round")                 
    arg_parser.add_argument('--target_mode', type=str, default="maxsquare",
                            choices=['maxsquare', 'IW_maxsquare', 'entropy', 'IW_entropy', 'hard'],
                            help="the loss function on target domain")
    arg_parser.add_argument('--lambda_target', type=float, default=1,
                            help="lambda of target loss")
    arg_parser.add_argument('--gamma', type=float, default=0, 
                            help='parameter for scaled entorpy')
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2, 
                            help='the ratio of image-wise weighting factor')
    arg_parser.add_argument('--threshold', type=float, default=0.95,
                            help="threshold for Self-produced guidance")
    
    arg_parser.add_argument("--lambda-seg", type=float, default=0.1,
                        help="lambda_seg.")
    arg_parser.add_argument("--lambda-adv-target1", type=float, default=0.001,
                        help="lambda_adv for adversarial training.")
    arg_parser.add_argument("--lambda-adv-target2", type=float, default=0.0002,
                        help="lambda_adv for adversarial training.") 
    arg_parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
                  
    return arg_parser

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset)+"2"+str(args.target_dataset)+"_"+args.target_mode

    agent = UDATrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()