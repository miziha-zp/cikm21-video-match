import numpy as np
import pandas as pd
import json
import joblib
from zipfile import ZIP_DEFLATED, ZipFile
import os
import joblib
from utils import *
from sklearn.model_selection import train_test_split
# ?train_test_split
from video_dataset import *
from torch.utils.data import DataLoader
from e2econfig import base_config
# inputszlist = [k[0] for k in inputszlist]
from transformers import AdamW
import scipy
from torch.nn.functional import normalize, dropout
from model import *
from customloss import *
from termcolor import colored
from sklearn.model_selection import train_test_split
from pprint import pprint
TEMP = 0.2
ALLIN = False
OUTSZ = 256
EPOCH = 6
BATCHSIZE = 384
HIDDENSIZE = base_config['FUSION_HIDDENSIZE']
LOSS = SpearmanCorrelationLoss(TEMP) 
# LOSS = WhiteMSE()
# LOSS = wiseMSE()
# LOSS = SentenceBertLoss(OUTSZ).cuda()
    
pretrain_model_path = 'checkpoint/time_10:13:19:25pytorch_model.bin.align_tag_ich.8'
#'checkpoint/time_10:11:23:04pytorch_model.bin.align_tag_ich.2'
test_path = 'npy/test'
save_path = 'result/allin5fold/'

EMB_POOL_VALID = [
'/data03/yrqUni/Workspace/QQB/zp/code/model2/pkl/test_bitm_vision_transdp01_bertlast_valid.pkl',\
    '/data03/yrqUni/Workspace/QQB/zp/code/model2/pkl/test_robitm_vision_transdp01_bertlast_valid.pkl',\
    '/data03/yrqUni/Workspace/QQB/ZLH/pretrain_baseline/pkl/valid_two_stagedict_input_zlh1002.pkl',\
    '/data03/yrqUni/Workspace/QQB/zp/code/baseline/pkl/roberta_valid_seq.pkl',\
#     '/data03/yrqUni/Workspace/QQB/zp/ICH_valid.pkl'
]
EMB_POOL_TEST = [
'/data03/yrqUni/Workspace/QQB/zp/code/model2/pkl/test_bitm_vision_transdp01_bertlast_test.pkl',\
    '/data03/yrqUni/Workspace/QQB/zp/code/model2/pkl/test_robitm_vision_transdp01_bertlast_test.pkl',\
    '/data03/yrqUni/Workspace/QQB/ZLH/pretrain_baseline/pkl/test_two_stagedict_input_zlh1002.pkl',\
    '/data03/yrqUni/Workspace/QQB/zp/code/baseline/pkl/roberta_test_seq.pkl',\
#     '/data03/yrqUni/Workspace/QQB/zp/ICH_test.pkl'
]

if base_config['USEMODELONLY']:
    '''
    比较尴尬，，，，，必须有一个才能正确实例化dataset。。。
    '''
    EMB_POOL_VALID = ['/data03/yrqUni/Workspace/QQB/zp/code/model2/pkl/test_bitm_vision_transdp01_bertlast_valid.pkl'
]
    EMB_POOL_TEST = ['/data03/yrqUni/Workspace/QQB/zp/code/model2/pkl/test_bitm_vision_transdp01_bertlast_test.pkl'
]

def test_model(model, knowledgePool, test_path, save_path):
    print('test'*10)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    testset = FinetuneDataSetTest(knowledgePool, test_path)
    test_dataloader = DataLoader(testset, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=False)
    vid_embedding = {}
    for data1_cpu, ids in test_dataloader:
        data1 = tuple([data.cuda() for data in data1_cpu[0]])
        input_id, mask, frame =  data1_cpu[1].cuda(), data1_cpu[2].cuda(), data1_cpu[3].cuda()
        with torch.no_grad():
            out_emb = model.finetune_forward(data1, frame, input_id, mask).detach().cpu().numpy().astype(np.float16)
            for vid, embedding in zip(ids, out_emb):
                vid_embedding[vid] = embedding.tolist()
                
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    now = get_str_time()
    output_json = os.path.join(save_path, f'{now}result.json')
    output_zip = os.path.join(save_path, f'{now}result.zip')
    with open(output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)

        
def main():

    valid_knowledgePool_t = []
    test_knowledgePool_t = []
    for emb_fn_valid, emb_fn_test in zip(EMB_POOL_VALID, EMB_POOL_TEST):
        print(emb_fn_valid)
        baseline_valid = joblib.load(emb_fn_valid)
        baseline_test = joblib.load(emb_fn_test)
        valid_knowledgePool_t.extend(baseline_valid)
        test_knowledgePool_t.extend(baseline_test)
    
    annotation_file = '/data03/yrqUni/Workspace/QQB/Data/pairwise/label.tsv'

    valid_knowledgePool = []
    test_knowledgePool = []
    print(len(valid_knowledgePool_t))
    for k1,k2 in zip(valid_knowledgePool_t, test_knowledgePool_t):
        if len(k1['2345203561710400875'].shape) == 1:
    #         score = test_spearmanr(k1, annotation_file)
    #         print(k1['2345203561710400875'].shape, score)
            valid_knowledgePool.append(k1)
            test_knowledgePool.append(k2)

    for k1, k2 in zip(valid_knowledgePool_t, test_knowledgePool_t):
        if len(k1['2345203561710400875'].shape) == 2:
            print(k1['2345203561710400875'].shape)
            valid_knowledgePool.append(k1)
            test_knowledgePool.append(k2)
    del valid_knowledgePool_t, test_knowledgePool_t
    import gc
    gc.collect()
    
    inputszlist = [know['2345203561710400875'].shape for know in valid_knowledgePool]
    print(inputszlist)
    query_list = []
    candidate_list = []
    relevance_list = []
    data_path = '/data03/yrqUni/Workspace/QQB/Data'
    annotation_file = os.path.join(data_path, 'pairwise/label.tsv')

    with open(annotation_file, 'r') as f:
        for line in f:
            query, candidate, relevance = line.split()
            query_list.append(query)
            candidate_list.append(candidate)
            relevance_list.append(relevance)


    valid_df = pd.DataFrame({
        "query":query_list,
        "candidate":candidate_list,
        "relevance":relevance_list
    })

    valid_df['relevance'] =  valid_df['relevance'].apply(float)
    
    print(LOSS)

    print(len(inputszlist), inputszlist)
    pprint(base_config)
    if base_config['USEMODELONLY']:
        inputszlist = []
    

    from sklearn.model_selection import GroupKFold
    np.random.seed(42)
    group_kfold = GroupKFold(n_splits=5)

    y = valid_df['relevance']
    groups = valid_df['candidate']
    train_part, valid_part = 0, 0 

    for train_index, test_index in group_kfold.split(valid_df, y, groups):
            # print("TRAIN:", train_index, "TEST:", test_index)
        train_part, valid_part = valid_df.iloc[train_index], valid_df.iloc[test_index]
        print(train_part.shape)
        print(valid_part.shape)

        training_data = FinetuneDataSetTrain('npy/valid', valid_knowledgePool, train_part)
        validing_data = FinetuneDataSetTrain('npy/valid', valid_knowledgePool, valid_part)

        train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=False)
        valid_dataloader = DataLoader(validing_data, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=False)
        NET = e2eNetwork(base_config, len(inputszlist),  inputszlist, hiddensz=HIDDENSIZE, outsz=OUTSZ)
        NET = load_backbone(NET, pretrain_model_path)
    #     NET = load_model(NET, pretrain_model_path)
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         NET = nn.DataParallel(NET)
        NET = NET.cuda()  
        #     optimizer = AdamW(NET.parameters(), lr=5e-5, weight_decay=1e-5)
        # optimizer
        bert_parameters = list(map(id, NET.bert_net.parameters()))
        normal_parameters = filter(lambda x: id(x) not in bert_parameters, NET.parameters()) #在整个模型的参数中将CRF层的参数过滤掉（filter）
        optimizer = AdamW([
            {'params': NET.bert_net.parameters(), 'lr': base_config["BERT_LR"]*0.5},\
            {'params': normal_parameters, 'lr':base_config["BASE_LR"]*0.5}])

        for epoch in range(EPOCH):
            data_enumerator = enumerate(train_dataloader)
            valid_enumerator = enumerate(valid_dataloader)
            loss_epoch = 0
            train_ans = []
            y_list = []
            for step, (data1_cpu, data2_cpu, y) in data_enumerator:
                optimizer.zero_grad()
                data1 = tuple([data.cuda() for data in data1_cpu[0]])
                data2 = tuple([data.cuda() for data in data2_cpu[0]])

                query_input_id, query_mask, query_frame =  data1_cpu[1].cuda(), data1_cpu[2].cuda(), data1_cpu[3].cuda()
                candidate_input_id, candidate_mask, candidate_frame =  \
                    data2_cpu[1].cuda(), data2_cpu[2].cuda(), data2_cpu[3].cuda()

                y = y.float().cuda()
                out1 = NET.finetune_forward(data1, query_frame, query_input_id, query_mask)
                out2 = NET.finetune_forward(data2, candidate_frame,candidate_input_id, candidate_mask)
    #             out1 = out1.detach()
                loss, preds = LOSS(out1, out2, y, return_sim=True)

                train_ans.extend(list(preds.detach().cpu().numpy()))
                y_list.extend(list(y.detach().cpu().numpy()))
                loss_epoch += loss.item()
                if step % 50 == 0:
    #                 print(y)
                    print(f"epoch:{epoch} \t , step:{step}\t loss:", loss.item())
                loss.backward()
    #             torch.cuda.empty_cache()
                optimizer.step()

        #         break
            if not ALLIN:
                spearmanr = scipy.stats.spearmanr(train_ans, y_list).correlation
                info = f"epoch:{epoch}\t spearmanr:{spearmanr} \tloss_epoch:{loss_epoch / len(train_dataloader)}"
                print(colored(info, 'blue'))


            if not ALLIN:
                loss_epoch = 0
                val_ans = []
                y_list = []
                for step, (data1_cpu, data2_cpu, y) in valid_enumerator:
                    optimizer.zero_grad()
                    data1 = tuple([data.cuda() for data in data1_cpu[0]])
                    data2 = tuple([data.cuda() for data in data2_cpu[0]])

                    query_input_id, query_mask, query_frame =  data1_cpu[1].cuda(), data1_cpu[2].cuda(), data1_cpu[3].cuda()
                    candidate_input_id, candidate_mask, candidate_frame =  \
                        data2_cpu[1].cuda(), data2_cpu[2].cuda(), data2_cpu[3].cuda()

                    y = y.float().cuda()
                    with torch.no_grad():
                        out1 = NET.finetune_forward(data1, query_frame, query_input_id, query_mask)
                        out2 = NET.finetune_forward(data2, candidate_frame, candidate_input_id, candidate_mask)

                        loss, preds = LOSS(out1, out2, y, return_sim=True)
                        val_ans.extend(list(preds.detach().cpu().numpy()))
                        y_list.extend(list(y.detach().cpu().numpy()))
                        loss_epoch += loss.item()
                    if step % 10 == 0:
                        print(f"valid epoch:{epoch} \t , step:{step}\t loss:", loss.item())

                spearmanr = scipy.stats.spearmanr(val_ans, y_list).correlation
                info = f"epoch:{epoch}\t spearmanr:{spearmanr} \tloss_epoch:{loss_epoch / len(valid_dataloader)}"
                print(colored("TEMP:"+str(TEMP), 'red'))
                print(colored("OUTSZ:"+str(OUTSZ), 'red'))
                print(colored("BATCHSIZE:"+str(BATCHSIZE), 'red'))
                print(colored("HIDDENSIZE:"+str(HIDDENSIZE), 'red'))
                print(colored(info, 'green'))

        test_model(NET, test_knowledgePool, test_path, save_path)
if __name__ == '__main__':
    main()