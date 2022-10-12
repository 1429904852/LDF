import os
from keras import models
import tensorflow as tf
from sklearn import metrics
import numpy as np
from toolkit import data_loader
import matplotlib.pyplot as plt
import random
import re


# sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
def make_class_image(K, top_k, words_value_for_one_class, title, sorted_list, save_path):
    plt.figure(figsize=(11, 6.5), dpi=150)
    grid = plt.GridSpec(1, top_k + 1, wspace=0.5)
    # first figure
    plt.subplot(grid[0, 0:top_k])
    plt.title(title)
    size = top_k  # top_k
    x = np.arange(size)  # 0, 1, 2, ..., size-1
    total_width = 0.8
    width = total_width / K  # K sentences
    x = x - (total_width - width) / 2  # central symmetric, that's why /2
    for i in range(K):  # K
        data = words_value_for_one_class[i][1]  # corresponding values, len = top_k
        label = words_value_for_one_class[i][0]  # corresponding words, len = top_k
        plt.bar(x + i * width, data, width=width, label=str(i + 1))
        for j, x_axis in enumerate(x + i * width):
            plt.text(x_axis - 0.05, 0, '---' + label[j], ha='left', rotation=-60, wrap=True, fontsize=8)
            plt.text(x_axis + 0.03, 0.01, format(data[j], '.6f'), ha='left', rotation=90, wrap=True, fontsize=8)

    plt.ylim([0.0, 1.0])  # strict range of y between 0 and 1
    plt.xticks([])  # turn off the scale mark
    plt.legend(loc='upper right', fontsize=5, ncol=1, framealpha=0.1)  # show legend
    plt.plot()

    # second figure
    plt.subplot(grid[0, top_k])
    num_of_words = 10
    y = np.arange(-total_width / 2, total_width / 2, total_width / num_of_words)
    label_y = [element[0] for element in sorted_list]  # element 0 is word, 1 is corresponding value
    data_y = [element[1] for element in sorted_list]
    if len(label_y) < num_of_words:
        for _ in range(num_of_words - len(label_y)):
            label_y.append('None')
            data_y.append(0.0)
    for k in range(num_of_words):
        plt.bar(y[k], data_y[k], width=total_width / num_of_words, label=format(data_y[k], '.6f'))
        plt.text(y[k] - 0.03, 0, '---' + label_y[k], ha='left', rotation=-60, wrap=True, fontsize=8)
    plt.xticks([])
    plt.legend(loc='upper right', fontsize=5, ncol=1, framealpha=0.1)  # show legend
    plt.plot()
    path = save_path + '/' + title + '.png'
    plt.savefig(path)
    # plt.show()


# words_value_N = []
# elements of classes_with_frequent_words: sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
def make_query_image(N, top_k, words_value_N, classes, sentence, label, classes_with_frequent_words, sentence_num, save_path):
    plt.figure(figsize=(22, 6.5), dpi=200)
    grid = plt.GridSpec(1, 3 + N, wspace=0.5)
    plt.suptitle(str(sentence_num) + ': ' + sentence + '\n')
    # first figure
    plt.subplot(grid[0, 0:3])
    plt.title(label)
    # size = top_k  # top_k
    # x = np.arange(size)  # 0, 1, 2, ..., size-1
    total_width = 0.8
    # width = total_width / N  # K sentences
    # x = x - (total_width - width) / 2  # central symmetric, that's why /2
    x = np.arange(- total_width / 2, total_width / 2, total_width / top_k)
    for i in range(N):
        # print(x)
        # print(words_value_N[i][1])
        plt.bar(x, np.array(words_value_N[i][1]), width=total_width / top_k, label=classes[i])
        for j in range(top_k):
            plt.text(x[j] - 0.05, 0, '---' + words_value_N[i][0][j]+'-'+str(words_value_N[i][2][j]), ha='left', rotation=-60, wrap=True, fontsize=8)
            plt.text(x[j] + 0.03, 0.01, words_value_N[i][1][j], ha='left', rotation=90, wrap=True, fontsize=8)
        x = x + 1
    plt.ylim([0.0, 1.0])  # strict range of y between 0 and 1
    plt.xticks([])  # turn off the scale mark
    plt.legend(loc='upper right', fontsize=5, ncol=1, framealpha=0.1)  # show legend
    plt.plot()

    # following figure
    for k in range(N):
        plt.subplot(grid[0, 3 + k])
        plt.title(classes[k])
        num_of_words = 10
        y = np.arange(-total_width / 2, total_width / 2, total_width / num_of_words)
        label_y = [element[0] for element in
                   classes_with_frequent_words[k]]  # element 0 is word, 1 is corresponding value
        data_y = [element[1] for element in classes_with_frequent_words[k]]
        if len(label_y) < num_of_words:
            for _ in range(num_of_words - len(label_y)):
                label_y.append('None')
                data_y.append(0.0)
        for k in range(num_of_words):
            plt.bar(y[k], data_y[k], width=total_width / num_of_words, label=format(data_y[k], '.6f'))
            plt.text(y[k] - 0.03, 0, '---' + label_y[k], ha='left', rotation=-60, wrap=True, fontsize=8)
        plt.xticks([])
        plt.legend(loc='upper right', fontsize=5, ncol=1, framealpha=0.1)  # show legend
        plt.plot()
    path = save_path + '/' + str(sentence_num) + '.png'
    plt.savefig(path)
    # plt.show()


def threshold(x, gate):
    y = tf.greater_equal(x, gate)
    y = tf.cast(y, dtype=float)
    # print('inside threshold')
    return y


def make_argmax(x):
    top_vals, _ = tf.nn.top_k(x, 1)
    output = tf.cast(tf.greater_equal(x, top_vals), tf.float64)
    return output


def convert_to_sentence(index_2_word_dict, numeric_sentence):
    # take in one numeric sequence then convert to original words
    sentence = []
    for i in range(len(numeric_sentence)):
        sentence.append(index_2_word_dict[numeric_sentence[i]])
    return sentence


def find_top_ks(att, sentence, top_k_values):
    top_values, top_indices = tf.nn.top_k(att, top_k_values)
    top_k_words = [sentence[i] for i in top_indices]
    top_values = top_values.numpy().tolist()
    top_indices = top_indices.numpy().tolist()
    # top_values = [int(i*1e4)/1e4 for i in top_values]
    return top_values, top_k_words, top_indices


def visualize_att_and_label(N, K, seed, gate_specified=0.0, trick=None, top_k=3, Q=5, dataset='FewAsp', tasks=600, max_len=100):
    total_dir = dataset + '_visualization/seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(trick)
    if not os.path.exists(total_dir):
        os.makedirs(total_dir)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    print('{}-way-{}-shot-seed-{}'.format(N, K, seed))
    B = 1
    gate = gate_specified
    if N == 5:
        gate = 0.3
    if N == 10:
        gate = 0.2
    if gate == 0.0:
        raise Exception('please specify a gate value since your setting is neither 5 way nor 10 way')

    if trick is None or 'AWATT' in trick:
        model_name = dataset + '_AWATT_seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(
            trick) + '_50_d'
        if N == 5 and K == 5:
            B = 2
    else:
        model_name = dataset + '_HATT_seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(
            trick) + '_50_d'
    print('gate is {}, with trick: {}'.format(gate, str(trick)))
    print('testing...')
    loaded_model = models.load_model(model_name + '.hd5')
    layer_name = []
    for layer in loaded_model.layers:
        if re.match('tf_op_layer_Softmax', layer.name) is not None:
            layer_name.append(layer.name)

    beta_layer = layer_name[0]
    beta_model = models.Model(inputs=loaded_model.input, outputs=loaded_model.get_layer(beta_layer).output)

    print('model loaded')

    dataloader = data_loader.JSONFileDataLoader(dataset=dataset, word_embedding_dim=50, max_len=max_len)
    word_2_vec_matrix = tf.convert_to_tensor(dataloader.word_2_vec_matrix)
    index_2_word = dataloader.make_index_2_word()  # dict whose keys are indices and values are words

    true_label, pred_label = [], []
    s_sentences, q_sentences, beta_att = [], [], []
    for step in range(int(tasks / B)):
        support_set, query_set = dataloader.next_batch(B=B, N=N, K=K, Q=Q, phrase='test')
        s_sentence = support_set['sentence']  # B, N, K, max len
        q_sentence = query_set['sentence']  # B, N, Q, max len
        s_mask = support_set['mask']
        q_mask = query_set['mask']
        label = query_set['label']  # B, N, Q, N

        s_sentences.append(s_sentence)  # meta tasks/B of (B, N, K, max len)
        q_sentences.append(q_sentence)  # meta tasks/B of (B, N, Q, max len)
        true_label.append(label)  # meta tasks/B of (B, N, Q, N)

        s_sentence_embedded = tf.nn.embedding_lookup(word_2_vec_matrix, s_sentence)
        # B, N, Q, max len, conv dim
        q_sentence_embedded = tf.nn.embedding_lookup(word_2_vec_matrix, q_sentence)
        # B, N, Q, max len, conv dim

        x_input_list = [s_sentence_embedded, q_sentence_embedded, s_mask, q_mask]
        if 'LCL' in trick or 'LDF' in trick:
            class_name = support_set['class']  # B, N max len=10
            class_name_embedded = tf.nn.embedding_lookup(word_2_vec_matrix, class_name)
            x_input_list.append(class_name_embedded)
            class_mask = support_set['class_mask']
            x_input_list.append(class_mask)

        # pred = loaded_model(inputs=x_input_list)  # B, N, Q, N
        beta = beta_model(inputs=x_input_list)
        # B, N, K, max len
        beta_att.append(beta)  # meta tasks/B of (B, N, K, max len)

    target_classes = dataloader.target_classes  # meta tasks/B, B, N
    target_classes = np.array(target_classes).reshape(-1, N)  # meta tasks, N

    s_sentences = np.concatenate(s_sentences, axis=0)  # meta tasks, N, K, max len
    beta_att = np.concatenate(beta_att, axis=0)  # meta tasks, N, K, max len

    np.set_printoptions(linewidth=np.inf)
    while True:
        msg = input('Enter an int value between [{}, {}]'.format(0, tasks - 1))
        if msg == 'exit':
            exit(0)
        if msg == 'continue':
            new_seed = input('input the new seed for this setting: ')
            visualize_att_and_label(N=N, K=K, seed=int(new_seed), trick=trick, dataset=dataset)
        classes_with_frequent_words = []  # N, K, max len
        classes = []
        support_set_dir_path = total_dir + '/meta_task_' + str(msg)
        if not os.path.exists(support_set_dir_path):
            os.makedirs(support_set_dir_path)
        for i in range(N):
            print('the {} sentences of class -\'{}\'- from the support set of meta task {}'.format(K, target_classes[
                int(msg)][i], int(msg)))
            classes.append(target_classes[int(msg)][i])
            word_dict = {}
            words_value_for_one_class, sorted_list = [], []
            # of len K, each element (list) is of form: [list of top_k words, list of top_k values]
            for j in range(K):
                index = np.where(s_sentences[int(msg)][i][j] == 400001)[0][0]  # 400001 is the index for padding
                if index < top_k:
                    index = top_k
                sentence_with_words = convert_to_sentence(index_2_word_dict=index_2_word,
                                                          numeric_sentence=s_sentences[int(msg)][i][j])
                att = beta_att[int(msg)][i][j][:index]

                values, words, _ = find_top_ks(att=att, sentence=sentence_with_words, top_k_values=top_k)
                words_value = [words, values]
                words_value_for_one_class.append(words_value)
                for k in range(top_k):
                    if words[k] in word_dict.keys():
                        word_dict[words[k]] += values[k]
                    else:
                        word_dict[words[k]] = values[k]
                print('---sentence: \n{}'.format(' '.join(x for x in sentence_with_words[:index])))  # max len
                print('---attention: \n{}'.format(att))  # max len
                print('---words and their attention of top {}: \n{}\n{}\n'.format(top_k, words, values))
                sorted_list = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
            print(
                'top {} words from {} sentences of class -\'{}\'-: \n{}\n'.format(top_k, K, target_classes[int(msg)][i],
                                                                                  sorted_list))
            make_class_image(K=K, top_k=top_k, words_value_for_one_class=words_value_for_one_class,
                             title=target_classes[int(msg)][i], sorted_list=sorted_list, save_path=support_set_dir_path)
            classes_with_frequent_words.append(sorted_list)


if __name__ == '__main__':
    visualize_att_and_label(N=5, K=10, seed=5, trick='AWATT_LCL', dataset='FewAsp')