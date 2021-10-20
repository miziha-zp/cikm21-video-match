import transformers as tfs
import torch
from torch import nn
from basemodel import *

import torch.nn as nn
import torch
from torch.nn.functional import normalize, dropout
from torch.nn import functional as F
from e2econfig import base_config

USEEMBONLY = base_config['USEEMBONLY']
USEMODELONLY = base_config['USEMODELONLY']

class MLP(nn.Module):
    def __init__(self, hiddenlist=[256,256,256], final_relu=False):
        '''
        hiddenlist must contain input size 
        '''
        super(MLP, self).__init__()
        modellist = [nn.BatchNorm1d(hiddenlist[0])]
        for idx, hdn in enumerate(hiddenlist[:-1]):

                modellist.append(nn.Linear(hiddenlist[idx], hiddenlist[idx+1])) 
                if idx == len(hiddenlist)-2:
                    if final_relu:
                        modellist.append(nn.ReLU())
                else:       
                    modellist.append(nn.ReLU())
                modellist.append(nn.BatchNorm1d(hiddenlist[idx+1]))
        self.model = nn.ModuleList(modellist)
    def forward(self, x):
        for l in  self.model:
            x = l(x)
        return x
        
class CNN(nn.Module):
    def __init__(self, outsz, seq_len=32, bert_embsz=768):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (size, bert_embsz)) for size in [3,4,5,6]])
        self.final_mlp = MLP([1024, 512, outsz])
        
    def forward(self, input_emb):
        cnn = input_emb.unsqueeze(1)#.cuda()
        cnn = [F.relu(conv(cnn)).squeeze(3) for conv in self.convs]
        # print(cnn[0].shape)
        cnn = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in cnn]
        # print(cnn[0].shape) # torch.Size([bs, 256])
        cnn = torch.cat(cnn, 1)
        out = self.final_mlp(cnn)
        # print(out.shape)
        return out

class SENet(nn.Module):
    def __init__(self, input_sz, ratio):
        super(SENet, self).__init__()
        self.channels = input_sz
        self.hidden_unit = input_sz // ratio
        self.BN = nn.BatchNorm1d(self.channels)
        self.channel_attention = nn.Sequential(
                nn.Linear(self.channels, self.hidden_unit, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_unit, self.channels, bias=False),
                nn.Sigmoid()
        )
    def forward(self, input_list):
        input_list = self.BN(input_list)
        atte_weight = self.channel_attention(input_list)
        final_emb = atte_weight * input_list
        return final_emb

class SE_MLP(nn.Module):
    def __init__(self, input_sz, ratio, hiddensz, outsz):
        super(SE_MLP, self).__init__()
        in_channel = input_sz
        self.senet = SENet(in_channel, ratio)
        self.mlp = nn.Sequential(
                nn.BatchNorm1d(in_channel),
                nn.Linear(in_channel, hiddensz),
                nn.ReLU(),
                nn.Linear(hiddensz, outsz, bias=True),
        )
    def forward(self, input):
        out = self.senet(input)
        out = self.mlp(out)
        return out
    
class ConCatSE(nn.Module):
    def __init__(self, input_sz, ratio, hiddensz, outsz):
        super(ConCatSE, self).__init__()
        in_channel = input_sz
        self.catdrop = nn.Dropout(0.2)
        self.fusion = nn.Linear(in_channel, hiddensz),
        self.senet = SENet(hiddensz, ratio)
        
    def forward(self, input):
        out = self.catdrop(input)
        out = self.fusion(out)
        out = self.senet(out)
        return out
    
    
class e2eNetwork(nn.Module):
    def __init__(self, config, KNOW_NUM, inputszlist, hiddensz=256, outsz=256):
        super(e2eNetwork, self).__init__()
        assert  KNOW_NUM == len(inputszlist)
        self.outsz = outsz
        if not USEMODELONLY:
            mlplist = []
            for shape in inputszlist:
                if len(shape) == 1:
                     mlplist.append(MLP([shape[0]]+[hiddensz, hiddensz, hiddensz]))
                else:
                     mlplist.append(CNN(hiddensz, seq_len=shape[0], bert_embsz=shape[1]))

            self.MLPlist = nn.ModuleList(mlplist)
#         +768+config['OUTPUT_SIZE']
#         in_channel = KNOW_NUM*hiddensz+config['OUTPUT_SIZE']#+768
        # finetune net
        if USEEMBONLY:
            in_channel = KNOW_NUM*hiddensz
        elif USEMODELONLY:
            in_channel = config['OUTPUT_SIZE'] + 768 
#             in_channel = 256 * 2
        else:
            in_channel = config['OUTPUT_SIZE'] + 768 + KNOW_NUM*hiddensz    
#         self.enhanceNet = SENet(in_channel, 8)
#         self.finalMLP = nn.Sequential(
#                 nn.BatchNorm1d(in_channel),
#                 nn.Linear(in_channel, hiddensz),
#                 nn.ReLU(),
#                 nn.Linear(hiddensz, outsz, bias=True),
#         )
        self.fusion_net = ConCatSE(in_channel, 8, hiddensz, outsz)
        # sub net
        self.frame_net = NeXtVLAD(config)
        self.bert_net = tfs.BertModel.from_pretrained(config['bert_path'])
        
        # pretrain tag head
        pretrain_fusion_channel = config['OUTPUT_SIZE']+768
#         self.pretrain_fusion = SENet(pretrain_fusion_channel, 8)
#         self.pretrain_mlp = nn.Sequential(
#                 nn.BatchNorm1d(pretrain_fusion_channel),
#                 nn.Linear(pretrain_fusion_channel, hiddensz),
#                 nn.ReLU(),
#                 nn.Linear(hiddensz, outsz, bias=True),
                
#         )
        self.pretrainfusion_net = ConCatSE(pretrain_fusion_channel, 8, hiddensz, outsz)
        self.tag_head = nn.Sequential(
            nn.Linear(outsz, 10000, bias=False),
            nn.Sigmoid()
        )
        # ICH head
        self.bert_fc = nn.Linear(768, config["ICH_SIZE"])
        self.frame_fc = nn.Linear(config['OUTPUT_SIZE'], config["ICH_SIZE"])
        
        ich_fusion_channel = config['ICH_SIZE'] * 2
#         self.ichpretrain_fusion = SENet(ich_fusion_channel, 8)
#         self.ichpretrain_mlp = nn.Sequential(
#                 nn.BatchNorm1d(ich_fusion_channel),
#                 nn.Linear(ich_fusion_channel, hiddensz),
#                 nn.ReLU(),
#                 nn.Linear(hiddensz, outsz, bias=True),
#         )
        self.ichfusion_net = ConCatSE(ich_fusion_channel, 8, hiddensz, outsz)

        
    def forward(self, frame_array, input_id, mask):
        # frame net
        frame_out = self.frame_net(frame_array)
        # bert
        bert_out = self.bert_net(input_ids=input_id[:,0,:], attention_mask=mask[:,0,:])[1]
        # fusion
        concat_emb = torch.cat([frame_out, bert_out], axis=1)
        concat_emb = self.pretrain_fusion(concat_emb)
        tag_pred = self.tag_head(concat_emb)
        # ich
        bert_out_ich = self.bert_fc(bert_out)
        frame_out_ich = self.frame_fc(frame_out)
        ich_input = torch.cat([bert_out_ich, frame_out_ich], axis=1)
#         ich_out = self.ichpretrain_fusion(ich_input)
#         ich_out = self.ichpretrain_mlp(ich_out)
        ich_out = self.ichfusion_net(ich_input)
    
        return frame_out, bert_out, concat_emb, tag_pred, bert_out_ich, frame_out_ich, ich_out
    
    def finetune_forward(self, input_list, frame_array, input_id, mask):
        if not USEMODELONLY:
        # embedding net
            know_out_list = []
            for idx, input in enumerate(input_list):
                know_out_list.append(self.MLPlist[idx](input))
        
        frame_out, bert_out, concat_emb, tag_pred, bert_out_ich, frame_out_ich, ich_out = \
            self.forward(frame_array, input_id, mask)
        # fusion
        if USEEMBONLY:
            knows = torch.cat(know_out_list, axis=1)
        elif USEMODELONLY:
            knows = torch.cat([frame_out, bert_out], axis=1)
#             knows = torch.cat([ich_out, concat_emb], axis=1)
        else:
            knows = torch.cat(know_out_list+[frame_out, bert_out], axis=1)
#         knows = self.enhanceNet(knows)
        out = self.fusion_net(knows)
        return out
        
        
if __name__ == '__main__':
    e2enet = e2eNetwork(base_config, 2, [(256, ), (256, )])
    print(e2enet)

        
        
