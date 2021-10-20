'''
convert tfrecord to npy for pytorch easy loading
miziha-zp
zzhangpeng@zju.edu.cn
'''

import logging
import os
from pprint import pprint
import tensorflow as tf
import numpy as np
from config import parser
from data_helper import create_datasets
from data_helper import FeatureParser
from tqdm import tqdm
def write_dataset2npy(dataset, data_path):
    print(f'starting {data_path}')
    dir_list = [os.path.join(data_path, t_dir) for t_dir in ['bert_input', 'frame_input', 'label']]
    for t_dir in dir_list:
        if not os.path.exists(t_dir):
            os.makedirs(t_dir)
            
    for batch in tqdm(dataset):
        '''
        input_ids (64, 32) int32
        mask (64, 32) int32
        frames (64, 32, 1536) float32
        num_frames (64, 1) int32
        vid (64,) object
        labels (64, 10000) int8
        '''
        input_ids = batch['input_ids'].numpy().astype(np.int32)
        mask = batch['mask'].numpy().astype(np.int8)
        frames = batch['frames'].numpy().astype(np.float16)
        vid = batch['vid'].numpy().astype(str)
        labels = batch['labels'].numpy().astype(np.int8)
        
        for t_vid, labels in zip(vid, labels):
            dir_path = os.path.join(data_path, 'label')
            file_path = os.path.join(dir_path, t_vid+'.npy')
            np.save(file_path, labels)
#             break
        
        for t_vid, input_id, mask_ in zip(vid, input_ids, mask):
            dir_path = os.path.join(data_path, 'bert_input')
            file_path = os.path.join(dir_path, t_vid+'.npy')
            ob = np.concatenate([input_id, mask_])
            np.save(file_path, ob)
#             break
        
        for t_vid, frame in zip(vid, frames):
            dir_path = os.path.join(data_path, 'frame_input')
            file_path = os.path.join(dir_path, t_vid+'.npy')
            np.save(file_path, frame)
#             break
        

def parse_dataset(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    feature_parser = FeatureParser(args)
    test_dataset = feature_parser.create_dataset(args.test_b_file, training=False, batch_size=args.test_batch_size)
    
    write_dataset2npy(val_dataset, "npy/valid")
    write_dataset2npy(test_dataset, "npy/test")
    write_dataset2npy(train_dataset, "npy/train")
    
        
def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    pprint(vars(args))
    parse_dataset(args)


if __name__ == '__main__':
    main()
