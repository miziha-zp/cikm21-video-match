# TODO:
'''
1. load trainset, valid, test #
2. test_fun#
3. train_fun#
4. loss 
4.1 infoNCEloss
4.2 Duel Softmax Loss
4.3 Binary CrossEntropy Loss#
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from e2econfig import base_config, transformer_vision_config
from transformers import AdamW
from torch.nn.functional import normalize, dropout
from termcolor import colored
from pprint import pprint

from utils import *
from video_dataset import *
from model import *
# from transformer_frame import *
from customloss import *

annotation_file = '/data03/yrqUni/Workspace/QQB/Data/pairwise/label.tsv'
output_dir = 'checkpoint_bertbase/'
history_model_path = 'checkpoint_bertbase/time_10:14:20:56pytorch_model.bin.align_tag_ich.0'#'checkpoint/time_10:12:08:56pytorch_model.bin.align_tag_ich.7'
#'checkpoint/time_10:11:21:25pytorch_model.bin.align_tag_ich.6'
#'checkpoint/time_10:11:10:18pytorch_model.bin.align_tag_ich.1'
TORCH_SEED = base_config['TORCH_SEED']
HIDDENSIZE = base_config['FUSION_HIDDENSIZE']
OUTSZ = base_config['OUTSZ']
TAGSIZE = 10000
BATCHSIZE = int(2048 * 0.5)
EPOCH = 100
BCELOSS = nn.BCELoss()
MATCHLoss = NCELoss(0.1)
ALIGNLoss = NCELoss(0.1)

def inference(epoch, valid_loader, NET):
    data_enumerator = enumerate(valid_loader)
    vid_embedding = {}
    bert_out_ich_embedding = {}
    frame_out_ich_embedding = {}
    ich_out_embedding = {}
    frame_out_embedding = {}
    bert_out_embedding = {}
    
    for step, (input_id, mask, frame_input, ids) in data_enumerator:
        input_id, mask, frame_input =  input_id.cuda(), mask.cuda(), frame_input.cuda()
        with torch.no_grad():
            frame_out, bert_out, concat_emb, tag_pred,\
                bert_out_ich, frame_out_ich, ich_out = \
                NET(frame_input, input_id, mask)
            
            for vid, embedding in zip(ids, concat_emb):
                vid_embedding[vid] = embedding.tolist()
                
            for vid, embedding in zip(ids, bert_out_ich):
                bert_out_ich_embedding[vid] = embedding.tolist()
                
            for vid, embedding in zip(ids, frame_out_ich):
                frame_out_ich_embedding[vid] = embedding.tolist()
                
            for vid, embedding in zip(ids, ich_out):
                ich_out_embedding[vid] = embedding.tolist()
                
#             for vid, embedding in zip(ids, frame_out):
#                 frame_out_embedding[vid] = embedding.tolist()
                
#             for vid, embedding in zip(ids, bert_out):
#                 bert_out_embedding[vid] = embedding.tolist()
                
    score = test_spearmanr(vid_embedding, annotation_file)
    cprint(f'epoch=>{epoch}: concat_emb_spearmanr:{score}', 'red')
    
    score = test_spearmanr(bert_out_ich_embedding, annotation_file)
    cprint(f'\nepoch=>{epoch}: bert_out_ich_embedding_spearmanr:{score}', 'blue')
    
    score = test_spearmanr(frame_out_ich_embedding, annotation_file)
    cprint(f'\nepoch=>{epoch}: frame_out_ich_embedding_spearmanr:{score}', 'blue')
    
    score = test_spearmanr(ich_out_embedding, annotation_file)
    cprint(f'\nepoch=>{epoch}: ich_out_embedding_spearmanr:{score}', 'red')
    

def train(epoch, train_dataloader, valid_loader, NET, optimizer):
    data_enumerator = enumerate(train_dataloader)
    losses = AverageMeter('Loss', ':.3')
    losses_tag = AverageMeter('TAG Loss', ':.3')
    losses_match = AverageMeter('MATCH Loss', ':.3')
    losses_align = AverageMeter('ALIGN Loss', ':.3')
    
    progress = ProgressMeter(len(train_dataloader),
        [losses, losses_tag, losses_match, losses_align],
        prefix="Epoch: [{}]".format(epoch))
    
    for step, (input_id, mask, frame_input, tag) in data_enumerator:
        optimizer.zero_grad()
        
        input_id, mask, frame_input =  input_id.cuda(), mask.cuda(), frame_input.cuda()
        tag = tag.float().cuda()
        frame_out, bert_out, concat_emb, tag_pred, \
            bert_out_ich, frame_out_ich, ich_out = \
            NET(frame_input, input_id, mask)
        
        frame_out_ich2, _, _, _, _, _, ich_out2 = \
            NET(frame_input, input_id, mask)
#         print(tag.shape)
        loss_tag = BCELOSS(tag_pred, tag) * TAGSIZE
        match_loss = MATCHLoss(bert_out_ich, frame_out_ich) + MATCHLoss(frame_out_ich2, bert_out_ich)
        align_loss = ALIGNLoss(frame_out_ich, frame_out_ich2) #+ ALIGNLoss(ich_out, ich_out)
        
        loss = loss_tag + match_loss + align_loss
        losses.update(loss.item())
        losses_tag.update(loss_tag.item())
        losses_match.update(match_loss.item())
        losses_align.update(align_loss.item())
        loss.backward()
        optimizer.step()
#         break
        if step % 30 == 0:
            now = get_str_time()
            progress.display(step)
#             print(f"{now}: epoch:{epoch} \t , step:{step}\t loss:{loss.item()}, match_loss:{match_loss.item()}, loss_tag:{loss_tag.item()}")
        if step % 1000 == 500:    
            inference(epoch, valid_loader, NET)

def main():
    seed_everything(TORCH_SEED)
    # model
    inputszlist = []
    NET = e2eNetwork(base_config, len(inputszlist),  inputszlist, hiddensz=HIDDENSIZE, outsz=OUTSZ)
    if history_model_path != '':
        NET = load_model(NET, history_model_path)
     # optimizer
    bert_parameters = list(map(id, NET.bert_net.parameters()))
    normal_parameters = filter(lambda x: id(x) not in bert_parameters, NET.parameters()) #在整个模型的参数中将CRF层的参数过滤掉（filter）
    optimizer = AdamW([
        {'params': NET.bert_net.parameters(), 'lr': base_config["BERT_LR"]},\
        {'params': normal_parameters, 'lr':base_config["BASE_LR"]}])
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        NET = nn.DataParallel(NET)
    NET = NET.cuda()
    
    # dataset
    train_set = PretrainDataSet('npy/train', training=True)
    valid_set = PretrainDataSet('npy/valid', training=False)
    test_set = PretrainDataSet('npy/test', training=False)
    
    train_dataloader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=False)
    valid_dataloader = DataLoader(valid_set, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=False)
    test_dataloader = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=False)
    # training
    for epoch in range(EPOCH):
        train(epoch, train_dataloader, valid_dataloader, NET, optimizer)
        inference(epoch, valid_dataloader, NET)
        save_model(epoch, output_dir, NET, type_name="align_tag_ich")

if __name__ == '__main__':
    main()
    
