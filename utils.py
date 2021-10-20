import os
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import random
import torch 
from termcolor import colored

class AverageMeter(object):
    def __init__(self, name, fmt=':.2f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(get_str_time(), '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def cprint(x, color='green'):
    print(colored(x, color))
    
def get_str_time():
    import time
    output_file='time_'+time.strftime("%m:%d:%H:%M", time.localtime())
    return output_file
            
def save_model(epoch, output_dir, model, type_name=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Only save the model it-self
    now=get_str_time()
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        output_dir, now+"pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    
    cprint(f"{now}: Model saved to {output_model_file}")
    return output_model_file

def load_model(NET, pretrain_model_path):
    if os.path.exists(pretrain_model_path) and len(pretrain_model_path)>0:
        model_state_dict = torch.load(pretrain_model_path, map_location='cpu')
        NET.load_state_dict(model_state_dict, strict=False)
        print('-'*100)
        print(pretrain_model_path, "loaded.")
        print('-'*100)
    else:
        print('-'*100)
        print(pretrain_model_path, "not exist.")
        print('-'*100)
    return NET.cuda()

def load_backbone(model, pretrain_model_path):
    if os.path.exists(pretrain_model_path) and len(pretrain_model_path)>0:
        save_model = torch.load(pretrain_model_path, map_location='cpu')
        model_dict =  model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if 'bert_net' in k or 'frame_net' in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
  
        print('-'*100)
        print('load model_state_dict number:', len(state_dict))
        print(pretrain_model_path, "loaded. only backbone !")
        print('-'*100)
    else:
        print('-'*100)
        print(pretrain_model_path, "not exist.")
        print('-'*100)
    return model.cuda()

# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
    
def test_spearmanr(vid_embedding, annotation_file):
    relevances, similarities = [], []
    with open(annotation_file, 'r') as f:
        for line in tqdm(f):
            query, candidate, relevance = line.split()
            if query not in vid_embedding:
                print(f'ERROR: {query} NOT found')
                continue
                # raise Exception(f'ERROR: {query} NOT found')
            if candidate not in vid_embedding:
                print(f'ERROR: {candidate} NOT found')
                continue
                # raise Exception(f'ERROR: {candidate} NOT found')
            # print('pass')
            query_embedding = vid_embedding.get(query)
            candidate_embedding = vid_embedding.get(candidate)
            similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
            similarities.append(similarity)
            relevances.append(float(relevance))

    spearmanr = scipy.stats.spearmanr(similarities, relevances).correlation
    return spearmanr