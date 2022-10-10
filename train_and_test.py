import os
from toolkit import framework
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

np.set_printoptions(threshold=sys.maxsize)


def train_and_test(B, N, K, Q, dataset, training_epoch, word_embedding_dim, seed, early_stop=True, patience=3, Em=4,
                   max_len=None, threshold=None, shuffle=True, trick=None, alpha=0.1, temp=0.1):
    # TODO: What's the value of Em we used? since in train_and_test.py /  framework.py they are all default to be 4,
    #  however in main() we set Em=2 ---> for consistency, maybe we need to change 4-->2 ? --- Yuchen Shen
    if trick is None or 'AWATT' in trick:
        model_name = dataset + '_AWATT_seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(
            trick) + '_' + str(word_embedding_dim) + '_d'
    else:
        model_name = dataset + '_HATT_seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(
            trick) + '_' + str(word_embedding_dim) + '_d'

    chasis = framework.Framework(B=B, N=N, K=K, Q=Q, dataset=dataset, model_name=model_name, Em=Em, gate=threshold,
                                 max_len=max_len, training_epoch=training_epoch, early_stop=early_stop,
                                 patience=patience, shuffle=shuffle, trick=trick, alpha=alpha, temp=temp)
    print('training...')
    best_eval_f1, best_eval_AUC, one_seed_dict = chasis.train()
    test_f1, test_AUC = chasis.test(tasks=600)

    d = {'dataset': dataset, 'B': chasis.B, 'N': N, 'K': K, 'seed': seed,
         'max len': max_len, 'early stop': early_stop, 'patience': patience, 'Em': Em, 'threshold': threshold,
         'trick': str(trick),
         'best_eval_f1': best_eval_f1, 'best_eval_AUC': best_eval_AUC, 'test_f1': test_f1, 'test_AUC': test_AUC}
    if 'CL' in trick or 'LDF' in trick:
        d['alpha'] = alpha
    return d, one_seed_dict


if __name__ == '__main__':
    extra_parameter = {'alpha': 0.1, 'temp': 0.1}
    # alpha means the λ in our paper, which is the weight of the CL loss
    # temp is the temperature used in the label-weighted contrastive loss
    seed_list = [5, 10, 15, 20, 25]
    # we take 5 runs where seed are set to 5, 10, 15, 20, 25
    model_list = ['LDF_AWATT']
    # you can choose one or multiple methods at one time
    # code:             corresponding model:
    #  None             the original AWATT model
    # 'AWATT_LAS'             AWATT+LAS
    # 'AWATT_LCL'             AWATT+LCL
    # 'AWATT_SCL'             AWATT+SCL
    # 'LDF_AWATT'             LDF-AWATT
    # 'HATT'           the original HATT model
    # 'HATT_LAS'               HATT+LAS
    # 'HATT_LCL'               HATT+LCL
    # 'HATT_SCL'               HATT+SCL
    # 'LDF-HATT'               LDF-HATT

    dataset_list = ['FewAsp', 'FewAsp(single)', 'FewAsp(multi)']
    # you can choose one or multiple datasets at one time
    config_list = [[2, 5, 5], [1, 5, 10], [1, 10, 5], [1, 10, 10]]
    # you can choose one or multiple configs at one time
    # [2, 5, 5] stands for: two(2) '5'-way-'5'-shot meta-tasks for one batch
    # [1, 5, 10] stands for: one(1) '5'-way-'10'-shot meta-task for one batch
    # [1, 10, 5] stands for: one(1) '10'-way-'5'-shot meta-task for one batch
    # [1, 10, 10] stands for: one(1) '10'-way-'10'-shot meta-task for one batch

    result_list = []
    whole_result = []
    for dataset in dataset_list:
        dataset_result = {'dataset': dataset, 'results': []}
        for config in config_list:
            config_result = {'B': config[0], 'N': config[1], 'K': config[2], 'data': []}
            for i in seed_list:
                seed_result = {'seed': i, 'seed_data': []}
                for trick in model_list:
                    seed = i
                    os.environ['PYTHONHASHSEED'] = str(seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    tf.random.set_seed(seed)

                    B = config[0]
                    N = config[1]
                    K = config[2]
                    print('N: {}, K: {}, dataset: {}, seed: {} '.format(N, K, dataset, seed))
                    result, one_seed_dict = train_and_test(B=B, N=N, K=K, Q=5, dataset=dataset, training_epoch=30,
                                                           word_embedding_dim=50, seed=seed, early_stop=True,
                                                           patience=3, Em=2,
                                                           max_len=100, shuffle=True, threshold=0.3, trick=trick,
                                                           alpha=extra_parameter['alpha'],
                                                           temp=extra_parameter['temp'])
                    one_seed_dict['trick'] = trick
                    seed_result['seed_data'].append(one_seed_dict)
                    result_list.append(result)
                config_result['data'].append(seed_result)
            dataset_result['results'].append(config_result)
        whole_result.append(dataset_result)
        pd.DataFrame(result_list).to_excel("/data1/zhaof/LDF/" + 'result.xlsx')
