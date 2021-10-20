base_config = {
    # nextvlad:
    "FEATURE_SIZE": 1536,
    "OUTPUT_SIZE": 1536,
    "EXPANSION_SIZE": 2,
    "CLUSTER_SIZE": 64, 
    "NUM_GROUPS": 8, 
    "DROPOUT_PROB": 0.2,
    "FUSION_HIDDENSIZE":1536,
    "TORCH_SEED":42,
    "OUTSZ":256,
    'USEBERT':True,
    "USERNEXTVLAD":True,
    "USEEMB":True,
    "bert_path": 'bert-base-chinese',
    "ICH_SIZE":256,
    "USEEMBONLY": False,
    "USEMODELONLY": False,
    "BERT_LR":1e-5,
    "BASE_LR":3e-4
#     hfl/chinese-roberta-wwm-ext
#     hfl/chinese-roberta-wwm-ext
#     peterchou/simbert-chinese-base ******vocab not match
#     bert-base-chinese
#     bert  
}
transformer_vision_config = {
    "d_model":1536,
    "nhead":12,
    "dim_feedforward":base_config['OUTPUT_SIZE'],
    "dropout":0.1, 
    "activation":'relu', 
    "layer_norm_eps":1e-05, 
    "num_layers": 3,
}
