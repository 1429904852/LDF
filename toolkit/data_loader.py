import random
import re

import numpy as np
import json


# Logic of call:
# 1. initialize necessary parameters
# 2. repeatedly call the "next_batch()" method to give us new meta-tasks

class JSONFileDataLoader(object):
    def __init__(self, dataset=None, word_embedding_dim=None, max_len=None, shuffle=True):
        if dataset is None:
            raise Exception('please specify a dataset, options: {}, {}, {}'.format('FewAsp',
                                                                                   'FewAsp(multi)',
                                                                                   'FewAsp(single)'))
        self.shuffle = shuffle
        # whether to shuffle the query instances
        self.train_path = 'dataset/' + dataset + '/train.json'
        self.val_path = 'dataset/' + dataset + '/val.json'
        self.test_path = 'dataset/' + dataset + '/test.json'
        with open(self.train_path, 'r') as f:
            self.train_JSON_dict = json.load(f)
        with open(self.val_path, 'r') as f:
            self.val_JSON_dict = json.load(f)
        with open(self.test_path, 'r') as f:
            self.test_JSON_dict = json.load(f)

        if word_embedding_dim is None:
            raise Exception('please specify a word embedding dimension, options: {}'.format(50))
        self.word_embedding_path = 'word_embedding/glove.6B.' + str(word_embedding_dim) + 'd.json'
        self.word_2_index, self.word_2_vec_matrix, number_of_words = self.word_2_vec(file_path=self.word_embedding_path)

        self.unk_index = number_of_words
        self.padding_index = number_of_words + 1

        unk = list(np.random.normal(size=word_embedding_dim))
        padding = [0.0] * word_embedding_dim
        self.word_2_vec_matrix.append(unk)
        self.word_2_vec_matrix.append(padding)
        self.target_classes = []
        # keep track of the sampled classes

        if max_len is None:
            self.max_len = max(self.find_max_len(file_path=self.train_path),
                               self.find_max_len(file_path=self.val_path),
                               self.find_max_len(file_path=self.test_path))
            # gets the possible maximum input text length over all data ---> 99
        else:
            self.max_len = max_len

    @staticmethod
    def find_max_len(file_path=None):
        with open(file_path, 'r') as f:
            JSON_dict = json.load(f)
        JSON_Value_list = list(JSON_dict.values())
        max_len = 0
        for i in range(len(JSON_Value_list)):
            for j in range(len(JSON_Value_list[i])):
                max_len = max(max_len, len(JSON_Value_list[i][j][0]))
        return max_len

    @staticmethod
    def word_2_vec(file_path=None):
        # this is used when the Glove is in .jsonl form
        with open(file_path, 'r') as f:
            JSON_list = json.load(f)
        word_2_index = {}
        word_2_vec = []
        for i in range(len(JSON_list)):
            word = list(JSON_list[i].values())[0]
            vec = list(JSON_list[i].values())[1]
            # TODO, what's the key of each element in JSON_list here? ---Yuchen Shen
            word_2_index[word] = i
            word_2_vec.append(vec)
        return word_2_index, word_2_vec, len(JSON_list)

    @staticmethod
    def word_2_vec_txt(file_path=None):
        # this is used when the Glove is in .txt form
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        word_2_index = {}
        word_2_vec = []
        i = 0
        for line in lines:
            terms = line.strip().split()
            word = terms.pop(0)
            vec = list(map(float, list([float(num) for num in terms])))
            if len(vec) != 50:
                print('wrong at {}, word {}'.format(i, word))
                print(vec)
                print(len(vec))
            word_2_index[word] = i
            word_2_vec.append(vec)
            i += 1
        return word_2_index, word_2_vec, len(lines)

    def make_index_2_word(self):
        # TODO: this might be used in the visualization as well --- Yuchen Shen
        index_2_word = dict(zip(self.word_2_index.values(), self.word_2_index.keys()))
        index_2_word[self.unk_index] = 'unk'
        index_2_word[self.padding_index] = 'padding'
        return index_2_word

    def next_one(self, N, K, Q, JSON_dict):
        support_set = {'sentence': [], 'mask': [], 'class': []}
        # "sentence": text input in support set / "mask": corresponding text mask; "class": label-text information
        query_set = {'sentence': [], 'label': [], 'mask': []}
        # "sentence": text input in support set / "mask": corresponding text mask; "label": the ground-truth

        target_classes = random.sample(JSON_dict.keys(), N)
        self.target_classes.append(target_classes)
        for i, class_name in enumerate(target_classes):  # this loops for N times
            class_words = list(set(re.split('_', class_name)))  # get label text (redundancies removed)
            class_words_sentence, _ = self.make_sentence_and_mask(class_words, max_len=10)
            indices = np.random.choice(list(range(len(JSON_dict[class_name]))), size=K + Q, replace=False)
            # choose K+Q instances for current class
            sentences, masks, labels = [], [], []
            for j in range(len(indices)):  # this loops for K+Q times
                index = indices[j]
                words = JSON_dict[class_name][index][0]
                aspects = JSON_dict[class_name][index][1]

                sentence, mask = self.make_sentence_and_mask(words=words)
                label = self.make_label(classes=target_classes, aspects=aspects)

                sentences.append(sentence)
                masks.append(mask)
                labels.append(label)
            support_sentences, query_sentences, _ = np.split(sentences, [K, K + Q])
            support_masks, query_masks, _ = np.split(masks, [K, K + Q])
            _, query_labels, _ = np.split(labels, [K, K + Q])

            support_set['sentence'].append(support_sentences)
            support_set['mask'].append(support_masks)
            support_set['class'].append(class_words_sentence)
            query_set['sentence'].append(query_sentences)
            query_set['mask'].append(query_masks)
            query_set['label'].append(query_labels)

        support_set['sentence'] = np.stack(support_set['sentence'], axis=0)  # N, K, max_len
        support_set['mask'] = np.stack(support_set['mask'], axis=0)  # N, K, max_len
        support_set['class'] = np.stack(support_set['class'], axis=0)  # N, max_len=10
        query_set['sentence'] = np.concatenate(query_set['sentence'], axis=0)  # N*Q, max_len
        query_set['mask'] = np.concatenate(query_set['mask'], axis=0)  # N*Q, max_len
        query_set['label'] = np.concatenate(query_set['label'], axis=0)  # N*Q, N
        if self.shuffle:
            perm = np.random.permutation(N * Q)
            query_set['sentence'] = query_set['sentence'][perm]
            query_set['mask'] = query_set['mask'][perm]
            query_set['label'] = query_set['label'][perm]

        query_set['sentence'] = np.reshape(query_set['sentence'], (N, Q, -1))
        query_set['mask'] = np.reshape(query_set['mask'], (N, Q, -1))
        query_set['label'] = np.reshape(query_set['label'], (N, Q, -1))

        return support_set, query_set

    def make_sentence_and_mask(self, words, max_len=None, mask_words=None):
        # mask = 1: valid words; = 0: words need to be ignored
        sentence = []
        mask = []
        if max_len is None:
            length = self.max_len
        else:
            length = max_len
        for i in range(length):
            if i >= len(words):
                mask.append(0)
                sentence.append(self.padding_index)
            else:
                if words[i] in self.word_2_index.keys():
                    index = self.word_2_index[words[i]]
                    sentence.append(index)
                    if mask_words is not None and words[i] in mask_words:
                        # some tokens we don't want, e.g., punctuations
                        mask.append(0)
                    else:
                        mask.append(1)
                else:
                    sentence.append(self.unk_index)
                    mask.append(1)
        return sentence, mask

    @staticmethod
    def make_label(classes, aspects):
        # classes: N class names
        # aspects: class names (<=N) contained in current instance
        label = [0] * len(classes)
        # a label of length N
        for i in range(len(classes)):
            if classes[i] in aspects:
                label[i] = 1
        return label

    def next_batch(self, B, N, K, Q, phrase='train'):
        JSON_dict = {}
        if phrase == 'train':
            JSON_dict = self.train_JSON_dict
        if phrase == 'eval':
            JSON_dict = self.val_JSON_dict
        if phrase == 'test':
            JSON_dict = self.test_JSON_dict

        batch_support_set = {'sentence': [], 'mask': [], 'class': []}
        batch_query_set = {'sentence': [], 'label': [], 'mask': []}
        for one_batch in range(B):
            current_support_set, current_query_set = self.next_one(N=N, K=K, Q=Q, JSON_dict=JSON_dict)
            # "next_one" method gives us 1 (batch_size=1) meta-task
            batch_support_set['sentence'].append(current_support_set['sentence'])
            batch_support_set['mask'].append(current_support_set['mask'])
            batch_support_set['class'].append(current_support_set['class'])
            batch_query_set['sentence'].append((current_query_set['sentence']))
            batch_query_set['mask'].append((current_query_set['mask']))
            batch_query_set['label'].append((current_query_set['label']))
        batch_support_set['sentence'] = np.stack(batch_support_set['sentence'], axis=0)  # B, N, K, max len
        batch_support_set['mask'] = np.stack(batch_support_set['mask'], axis=0)  # B, N, K, max len
        batch_support_set['class'] = np.stack(batch_support_set['class'], axis=0)  # B, N, max len=10
        batch_query_set['sentence'] = np.stack(batch_query_set['sentence'], axis=0)  # B, N, Q, max len
        batch_query_set['mask'] = np.stack(batch_query_set['mask'], axis=0)  # B, N, Q, max len
        batch_query_set['label'] = np.stack(batch_query_set['label'], axis=0)  # B, N, Q, N

        return batch_support_set, batch_query_set
