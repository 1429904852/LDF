from sklearn import manifold
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
import keras
from toolkit import data_loader
import re
import numpy as np
import json


def visualize_proto(dataset, B, N, K, trick, use_label, seed, times=50):  # use_label = 1 means using the label
    dataloader = data_loader.JSONFileDataLoader(dataset=dataset, word_embedding_dim=50, max_len=100)
    if trick is None or 'AWATT' in trick:
        model_name = dataset + '_AWATT_seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(
            trick) + '_50_d'
        model_type = 'AWATT'
    else:
        model_name = dataset + '_HATT_seed_' + str(seed) + '_' + str(N) + '_way_' + str(K) + '_shot_trick_' + str(
            trick) + '_50_d'
        model_type = 'HATT'
    loaded_model = keras.models.load_model(model_name + '.hd5')
    print('model loaded')

    for layer in loaded_model.layers:
        if model_type == 'AWATT':
            if len(layer.output.shape) == 3 and layer.output.shape[-2] == N and layer.output.shape[-1] == 50:
                print('layer name:\n {} \n and the output shape:\n {}'.format(layer.name, layer.output))
        else:
            if len(layer.output.shape) == 4 and layer.output.shape[-2] == N and layer.output.shape[-1] == 50:
                print('layer name:\n {} \n and the output shape:\n {}'.format(layer.name, layer.output))

    msg_layer = input('choose the layer you want to visualize')

    matrix = dataloader.word_2_vec_matrix
    matrix = tf.cast(matrix, dtype='float')

    with open('dataset/separate.json', 'r') as f:
        classes = json.load(f)['test']
    for i, class_name in enumerate(classes):
        print('the {}-th class is \'{}\''.format(i, class_name))
    msg_classes = input('the {} classes you want to visualize'.format(N))
    choose_N = [int(x) for x in re.findall(r"\d+\.?\d*", msg_classes)]

    visualize_model = keras.Model(inputs=loaded_model.input, outputs=loaded_model.get_layer(msg_layer).output)
    proto_vectors = []
    for step in range(times):
        support_set, query_set = dataloader.next_batch(B=B, N=N, K=K, Q=5, phrase='test', choose_N=choose_N)
        s_sentence = support_set['sentence']
        q_sentence = query_set['sentence']

        s_mask = support_set['mask']
        q_mask = query_set['mask']

        s_sentence_embedded = tf.nn.embedding_lookup(matrix, s_sentence)
        q_sentence_embedded = tf.nn.embedding_lookup(matrix, q_sentence)

        x_input_list = [s_sentence_embedded, q_sentence_embedded, s_mask, q_mask]
        if use_label == 1:
            class_name = support_set['class']  # B, N max len=10
            class_name_embedded = tf.nn.embedding_lookup(matrix, class_name)
            x_input_list.append(class_name_embedded)

        pred = visualize_model.predict_on_batch(x_input_list)
        if model_type == 'HATT':  # pred is B, {NQ}, N, conv dim
            pred = tf.reduce_mean(pred, axis=1)
        proto_vectors.append(pred)
    # print(proto_vectors[:5])
    proto_vectors = np.concatenate(proto_vectors, axis=0)
    # print(proto_vectors[:5])
    proto_vectors = np.reshape(proto_vectors, (-1, 50))
    # print(proto_vectors[:5])
    """
    if mode == 'choose':
        msg = input('please input the boundaries (e.g. 1, 200), max number: {}'.format(proto_vectors.shape[0]))
        numbers = re.findall(r"\d+\.?\d*", msg)
        proto_vectors = proto_vectors[int(numbers[0]):int(numbers[1])]

    if mode == 'random':
        msg = input('please input the boundaries (e.g. 200), max number: {}'.format(proto_vectors.shape[0]))
        number = re.findall(r"\d+\.?\d*", msg)
        row_rand_array = np.arange(proto_vectors.shape[0])
        np.random.shuffle(row_rand_array)
        proto_vectors = proto_vectors[row_rand_array[ :int(number[0])]]
    """
    # prototype be like: B, N, conv dim
    # https://mortis.tech/2019/11/program_note/664/
    t_SNE = manifold.TSNE(n_components=2, init='pca', verbose=1)
    x_SNE = t_SNE.fit_transform(proto_vectors)
    x_min, x_max = x_SNE.min(0), x_SNE.max(0)
    X_norm = (x_SNE - x_min) / (x_max - x_min)  # normalized
    # print(proto_vectors.shape[0])
    df = pandas.DataFrame(
        {' ': X_norm[:, 0], '  ': X_norm[:, 1], 'label': [x % N for x in range(proto_vectors.shape[0])]})
    df.plot(x=' ', y='  ', kind='scatter', c='label', marker='1', colormap='viridis', s=3)
    # the x and y should be the same for the keys for dict when constructing the dataframe
    plt.show()


visualize_proto(dataset='FewAsp(single)', B=1, N=5, K=10, trick='HATT_LCL', use_label=1, seed=5)