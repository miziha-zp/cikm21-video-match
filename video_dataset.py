import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
from torchvision.transforms import ToTensor

class PretrainDataSet(Dataset):
    def __init__(self, npy_path, training=True):
        self.npy_path = npy_path
        self.bert_npy_path = os.path.join(self.npy_path, 'bert_input')
        self.frame_npy_path = os.path.join(self.npy_path, 'frame_input')
        self.label_npy_path = os.path.join(self.npy_path, 'label')
        self.vid = os.listdir(self.label_npy_path)
        self.training = training
        
    def __len__(self):
        return len(self.vid)
    
    def get_bert_input(self, path):
        bert_input = np.load(path)
        input_id, mask = bert_input[:32], bert_input[32:]
        input_id = torch.from_numpy(np.array([input_id])).long()
        mask = torch.from_numpy(np.array([mask])).long()
        return input_id, mask
    
    def get_frame_input(self, path):
        npy_frame = np.load(path)
        return torch.from_numpy(npy_frame.astype(np.float32))
    
    def get_tag(self, path):
        tag = np.load(path)
        return torch.from_numpy(tag.astype(np.int8))
    
    def __getitem__(self, idx):
        fn = self.vid[idx]
        # bert input
        bert_input_path = os.path.join(self.bert_npy_path, fn)
        input_id, mask = self.get_bert_input(bert_input_path)
        # frame input
        frame_input_path = os.path.join(self.frame_npy_path, fn)
        frame_input = self.get_frame_input(frame_input_path)
        # tag 
        if self.training:
            tag_path = os.path.join(self.label_npy_path, fn)
            tag = self.get_tag(tag_path)
            return input_id, mask, frame_input, tag
        else:
            return input_id, mask, frame_input, fn.split('.')[0]
        

class FinetuneDataSetTrain(Dataset):
    def __init__(self, npy_path, knowledgePool, df):
        self.df = df
        self.knowledgePool = knowledgePool
        self.npy_path = npy_path
        self.bert_npy_path = os.path.join(self.npy_path, 'bert_input')
        self.frame_npy_path = os.path.join(self.npy_path, 'frame_input')
        
    def __len__(self):
        return len(self.df)
    
    def get_bert_input(self, path):
        bert_input = np.load(path)
        input_id, mask = bert_input[:32], bert_input[32:]
        input_id = torch.from_numpy(np.array([input_id])).long()
        mask = torch.from_numpy(np.array([mask])).long()
        return input_id, mask
    
    def get_frame_input(self, path):
        npy_frame = np.load(path)
        return torch.from_numpy(npy_frame.astype(np.float32))
        
    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        query, candidate, relevance = line['query'], line["candidate"], line["relevance"]
        
        # bert input
        query_bert_input_path = os.path.join(self.bert_npy_path, str(query)+'.npy')
        candidate_bert_input_path = os.path.join(self.bert_npy_path, str(candidate)+'.npy')
        query_input_id, query_mask = self.get_bert_input(query_bert_input_path)
        candidate_input_id, candidate_mask = self.get_bert_input(candidate_bert_input_path)
        # frame input 
        query_frame_input_path = os.path.join(self.frame_npy_path, str(query)+'.npy')
        candidate_frame_input_path = os.path.join(self.frame_npy_path, str(candidate)+'.npy')
        query_frame = self.get_frame_input(query_frame_input_path)
        candidate_frame = self.get_frame_input(candidate_frame_input_path)
        # embedding input
        queryemblist = []
        for knowledge in self.knowledgePool:
            queryemblist.append(torch.from_numpy(knowledge[query].astype(np.float32)))

        candidateemblist = []
        for knowledge in self.knowledgePool:
            candidateemblist.append(torch.from_numpy(knowledge[candidate].astype(np.float32)))
        queryemblist, candidateemblist = tuple(queryemblist), tuple(candidateemblist)
        
        query_data = (queryemblist, query_input_id, query_mask, query_frame)
        candidate_data = (candidateemblist, candidate_input_id, candidate_mask, candidate_frame)
        relevance = float(relevance)
        
        return query_data, candidate_data, relevance   
    


class FinetuneDataSetTest(Dataset):
    def __init__(self, knowledgePool, npy_path):
        self.knowledgePool = knowledgePool
        self.id = [id for id in self.knowledgePool[0]]
        self.npy_path = npy_path
        self.bert_npy_path = os.path.join(self.npy_path, 'bert_input')
        self.frame_npy_path = os.path.join(self.npy_path, 'frame_input')
        
    def __len__(self):
        return len(self.id)
    
    def get_bert_input(self, path):
        bert_input = np.load(path)
        input_id, mask = bert_input[:32], bert_input[32:]
        input_id = torch.from_numpy(np.array([input_id])).long()
        mask = torch.from_numpy(np.array([mask])).long()
        return input_id, mask
    
    def get_frame_input(self, path):
        npy_frame = np.load(path)
        return torch.from_numpy(npy_frame.astype(np.float32))
    
    def __getitem__(self, idx):
        id = self.id[idx]
        # embedding input
        emblist = []
        for knowledge in self.knowledgePool:
            emblist.append(torch.from_numpy(knowledge[id].astype(np.float32)))
        # bert input
        bert_input_path = os.path.join(self.bert_npy_path, str(id)+'.npy')
        input_id, mask = self.get_bert_input(bert_input_path)
        # frame input
        frame_input_path = os.path.join(self.frame_npy_path, str(id)+'.npy')
        frame_data = self.get_frame_input(frame_input_path)
        emblist = tuple(emblist)
        data = (emblist, input_id, mask, frame_data)
        return data, id
            
        