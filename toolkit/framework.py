import numpy as np
from keras.optimizers import Adam
from keras import models
from sklearn import metrics
from model import AWATT, AWATT_LAS, AWATT_SCL, LDF_AWATT, AWATT_LCL, \
    HATT, HATT_SCL, HATT_LCL, LDF_HATT, HATT_LAS

from toolkit import data_loader
import tensorflow as tf
import sys

np.set_printoptions(threshold=sys.maxsize)


def threshold(x, gate):
    y = tf.greater_equal(x, gate)
    y = tf.cast(y, dtype=float)
    return y


def make_argmax(x):
    top_vals, _ = tf.nn.top_k(x, 1)
    output = tf.cast(tf.greater_equal(x, top_vals), tf.float64)
    return output


class Framework(object):
    def __init__(self, B, N, K, Q, model_name, dataset, training_epoch, early_stop=True, patience=3, Em=4,
                 word_embedding_dim=50, gate=None, max_len=None, shuffle=True, trick=None, alpha=0.1,
                 temp=0.1):
        """
        :param B: batch_size (how many N-way K-shot tasks are there)
        :param N: number of classes (N-way K-shot)
        :param K: number of instances per class (N-way K-shot)
        :param Q: number of test instance per class ("queries" in meta-task)
        :param model_name: used as the name for saving directory
        :param dataset, training_epoch: self-explaining
        :param early_stop, patience: whether to use early-stopping , patience of early-stopping
        :param Em: parameter for AWATT (https://aclanthology.org/2021.acl-long.495.pdf, eq(3))
        :param gate: a statistic threshold (https://aclanthology.org/2021.acl-long.495.pdf, "Effects of Thresholds")
        :param word_embedding_dim, max_len, shuffle: parameters for data processing
        :param trick: which model / framework to use
        :param alpha: the weight λ of contrastive loss in ours, since "lambda" is reserved, we use alpha instead
        :param temp: the temperature used in contrastive loss
        """
        self.model_name = model_name + '.hd5'
        self.B = B
        self.N = N
        self.K = K
        self.Q = Q
        self.epoch = training_epoch
        self.dataset = dataset
        self.patience = patience
        self.early_stop = early_stop
        self.trick = trick
        self.use_label_flag = 0
        # this flag specifies whether label information is introduced
        print('the trick is {}'.format(self.trick))
        if N == 5:
            self.gate = 0.3  # 0.3
        elif N == 10:
            self.gate = 0.2  # 0.2
        elif gate is None:
            raise Exception('since it\'s not 5-way or 10-way, you need to specify the gate your own for computing '
                            'macro f1')
        else:
            self.gate = gate
        if self.dataset != 'FewAsp(single)':
            print(self.gate)
        self.dataloader = data_loader.JSONFileDataLoader(dataset=self.dataset, word_embedding_dim=word_embedding_dim,
                                                         max_len=max_len, shuffle=shuffle)
        self.word_2_vec_matrix = tf.convert_to_tensor(self.dataloader.word_2_vec_matrix)
        if self.trick is None:
            self.model = AWATT.make_model(N=self.N, K=self.K, Q=self.Q, Em=Em, max_len=self.dataloader.max_len)
        if self.trick == 'AWATT_LAS':
            self.model = AWATT_LAS.make_model(N=self.N, K=self.K, Q=self.Q, Em=Em, max_len=self.dataloader.max_len)
            self.use_label_flag = 1
        if self.trick == 'AWATT_SCL':
            self.model = AWATT_SCL.make_model(N=self.N, K=self.K, Q=self.Q, Em=Em, max_len=self.dataloader.max_len,
                                              alpha=alpha, temp=temp)
        if self.trick == 'AWATT_LCL':
            self.model = AWATT_LCL.make_model(N=self.N, K=self.K, Q=self.Q, Em=Em, max_len=self.dataloader.max_len,
                                              alpha=alpha, temp=temp)
            self.use_label_flag = 1
        if self.trick == 'LDF_AWATT':
            self.model = LDF_AWATT.make_model(N=self.N, K=self.K, Q=self.Q, Em=Em, max_len=self.dataloader.max_len,
                                              alpha=alpha, temp=temp)
            self.use_label_flag = 1

        if self.trick == 'HATT':
            self.model = HATT.make_model(N=self.N, K=self.K, Q=self.Q, max_len=self.dataloader.max_len)
        if self.trick == 'HATT_LAS':
            self.model = HATT_LAS.make_model(N=self.N, K=self.K, Q=self.Q, max_len=self.dataloader.max_len)
            self.use_label_flag = 1
        if self.trick == 'HATT_SCL':
            self.model = HATT_SCL.make_model(N=self.N, K=self.K, Q=self.Q, max_len=self.dataloader.max_len,
                                             alpha=alpha, temp=temp)
        if self.trick == 'HATT_LCL':
            self.model = HATT_LCL.make_model(N=self.N, K=self.K, Q=self.Q, max_len=self.dataloader.max_len, alpha=alpha,
                                             temp=temp)
            self.use_label_flag = 1
        if self.trick == 'LDF_HATT':
            self.model = LDF_HATT.make_model(N=self.N, K=self.K, Q=self.Q, max_len=self.dataloader.max_len, alpha=alpha,
                                             temp=temp)
            self.use_label_flag = 1

    def train(self, train_tasks=800, eval_tasks=600):
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        best_AUC = 0.0
        best_macro_f1 = 0.0
        eval_epoch = 0
        epoch = 1
        patience = self.patience

        train_loss_recorder = []
        eval_AUC_recorder = []
        eval_f1_recorder = []

        result_dict = {'seed': 0, 'eval_auc': eval_AUC_recorder, 'eval_f1': eval_f1_recorder, 'epoch': 0,
                       'loss': train_loss_recorder}
        while epoch <= self.epoch:
            print('epoch {}: '.format(epoch))
            loss_train = self.train_one_epoch(tasks=train_tasks)
            train_loss_recorder.append(loss_train)

            AUC_eval, macro_f1_eval = self.eval(tasks=eval_tasks)
            eval_AUC_recorder.append(AUC_eval)
            eval_f1_recorder.append(macro_f1_eval)
            print('eval macro f1: {}, AUC: {} for this epoch'.format(macro_f1_eval, AUC_eval))

            if AUC_eval > best_AUC:
                print('saving...')
                self.model.save(self.model_name)
                print('saved')
                best_AUC = AUC_eval
                best_macro_f1 = macro_f1_eval
                eval_epoch = epoch
                patience = self.patience
            elif self.early_stop:
                patience -= 1
            else:
                patience = self.patience

            if patience == 0:
                break
            print('patience now: {}'.format(patience))
            epoch += 1
        print('best macro f1: {} and best AUC: {} at epoch {} during evaluating'.format(best_macro_f1, best_AUC,
                                                                                        eval_epoch))
        result_dict['epoch'] = len(train_loss_recorder)
        return best_macro_f1, best_AUC, result_dict

    def train_one_epoch(self, tasks):
        loss_train = 0.0
        for step in range(int(tasks / self.B)):
            support_set, query_set = self.dataloader.next_batch(B=self.B, N=self.N, K=self.K,
                                                                Q=self.Q, phrase='train')
            s_sentence = support_set['sentence']
            q_sentence = query_set['sentence']

            s_mask = support_set['mask']
            q_mask = query_set['mask']
            label = query_set['label']  # B, N, Q, N

            s_sentence_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, s_sentence)
            q_sentence_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, q_sentence)

            x_input_list = [s_sentence_embedded, q_sentence_embedded, s_mask, q_mask]
            if self.use_label_flag == 1:
                class_name = support_set['class']  # B, N, max len=10
                class_name_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, class_name)
                x_input_list.append(class_name_embedded)
                # adds the label information to the model input

            metrics_dict = self.model.train_on_batch(x=x_input_list, y=label, return_dict=True)

            # print('step: {}----------metrics are : {}'.format(step, metrics_dict))
            loss_train += metrics_dict[list(metrics_dict.keys())[-1]]
            # TODO: might need to check whether "list(metrics_dict.keys())[-1]" corresponds to the "loss" term
            #  based on the above commented "print", this might be right ---Yuchen Shen
        loss_train /= int(tasks / self.B)
        print('training loss in this epoch {}'.format(loss_train))
        return loss_train

    def eval(self, tasks):
        # print('evaluating...')
        AUC_eval = 0.0
        macro_f1_eval = 0.0
        for step in range(int(tasks / self.B)):
            support_set, query_set = self.dataloader.next_batch(B=self.B, N=self.N, K=self.K,
                                                                Q=self.Q, phrase='eval')
            s_sentence = support_set['sentence']
            q_sentence = query_set['sentence']

            s_mask = support_set['mask']
            q_mask = query_set['mask']
            label = query_set['label']
            s_sentence_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, s_sentence)
            # B, N, K, max len, word embedding dim
            q_sentence_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, q_sentence)
            # B, N, Q, max len, word embedding dim
            x_input_list = [s_sentence_embedded, q_sentence_embedded, s_mask, q_mask]
            if self.use_label_flag == 1:
                class_name = support_set['class']  # B, N max len=10
                class_name_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, class_name)
                x_input_list.append(class_name_embedded)

            pred = self.model(x_input_list)
            if self.dataset == 'FewAsp(single)':
                threshold_pred = make_argmax(pred)
                # only make 1 entry prediction as such: [0.5, 0.4, 0.1] ---> [1, 0, 0]
            else:
                threshold_pred = threshold(pred, gate=self.gate)
                # make prediction with threshold, as such: gate=0.3, [0.5, 0.4, 0.1] ---> [1, 1, 0]
            auc_sklearn = metrics.roc_auc_score(np.array(label).reshape((-1, self.N)),
                                                pred.numpy().reshape((-1, self.N)), multi_class='ovo')
            f1_score = metrics.f1_score(label.reshape((-1, self.N)), threshold_pred.numpy().reshape((-1, self.N)),
                                        average='macro')

            AUC_eval += auc_sklearn
            macro_f1_eval += f1_score
        AUC_eval /= int(tasks / self.B)
        macro_f1_eval /= int(tasks / self.B)
        return AUC_eval, macro_f1_eval

    def test(self, tasks):
        print('testing...')
        loaded_model = models.load_model(self.model_name)
        print('model loaded')
        AUC_test = 0.0
        macro_f1_test = 0.0
        for step in range(int(tasks / self.B)):
            support_set, query_set = self.dataloader.next_batch(B=self.B, N=self.N, K=self.K,
                                                                Q=self.Q, phrase='test')
            s_sentence = support_set['sentence']
            q_sentence = query_set['sentence']

            s_mask = support_set['mask']
            q_mask = query_set['mask']
            label = query_set['label']  # B, N, Q, N
            s_sentence_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, s_sentence)
            q_sentence_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, q_sentence)

            x_input_list = [s_sentence_embedded, q_sentence_embedded, s_mask, q_mask]
            if self.use_label_flag == 1:
                class_name = support_set['class']  # B, N max len=10
                class_name_embedded = tf.nn.embedding_lookup(self.word_2_vec_matrix, class_name)
                x_input_list.append(class_name_embedded)

            pred = loaded_model.predict_on_batch(x_input_list)
            if self.dataset == 'FewAsp(single)':
                threshold_pred = make_argmax(pred)
                # only make 1 entry prediction as such: [0.5, 0.4, 0.1] ---> [1, 0, 0]
            else:
                threshold_pred = threshold(pred, gate=self.gate)
                # make prediction with threshold, as such: gate=0.3, [0.5, 0.4, 0.1] ---> [1, 1, 0]
            auc_sklearn = metrics.roc_auc_score(np.array(label).reshape((-1, self.N)),
                                                pred.reshape((-1, self.N)), multi_class='ovo')
            f1_score = metrics.f1_score(label.reshape((-1, self.N)), threshold_pred.numpy().reshape((-1, self.N)),
                                        average='macro')

            AUC_test += auc_sklearn
            macro_f1_test += f1_score
        AUC_test /= int(tasks / self.B)
        macro_f1_test /= int(tasks / self.B)
        print('test macro f1: {}, AUC: {} for {} meta tasks'.format(macro_f1_test, AUC_test, tasks))
        return macro_f1_test, AUC_test
