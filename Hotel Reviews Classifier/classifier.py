import collections
import glob
import os
import pickle
import sys
import random
import re
import numpy
import tensorflow as tf
import IPython
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib


def GetInputFiles():
    return glob.glob(os.path.join("op_spam_training_data", '*/*/*/*.txt'))


VOCABULARY = collections.Counter()


# ** TASK 1.
def Tokenize(comment):
    """Receives a string (comment) and returns array of tokens."""
    # words = comment.split()
    comment = re.sub(r'\b\w{1,1}\b', '', comment)
    comment = comment.lower()
    words = re.split('[^a-zA-Z]', comment)
    words = list(filter(None, words))
    return words


# ** TASK 2.
VARS = dict()
def FirstLayer(net, l2_reg_val, is_training):
    """First layer of the neural network.

    Args:
        net: 2D tensor (batch-size, number of vocabulary tokens),
        l2_reg_val: float -- regularization coefficient.
        is_training: boolean tensor.A

    Returns:
        2D tensor (batch-size, 40), where 40 is the hidden dimensionality.
    """

    net = tf.nn.l2_normalize(net, axis=1)

    net = tf.contrib.layers.fully_connected(net, 40, activation_fn=None, weights_regularizer=None,
                                            biases_initializer=None)
    # VARS[2] = net
    tf.losses.add_loss(tf.reduce_sum(l2_reg_val * tf.square(net)), tf.GraphKeys.REGULARIZATION_LOSSES)

    net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    net = tf.nn.tanh(net)
    # VARS[3] = net

    return net


# ** TASK 2 ** BONUS part 1
def EmbeddingL2RegularizationUpdate(embedding_variable, net_input, learn_rate, l2_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    global VARS

    net = tf.nn.l2_normalize(net_input, axis=1)
    net1 = tf.matmul(net, embedding_variable)
    net1 = tf.transpose(net1)
    net2 = tf.matmul(net1, net)
    net2 = tf.transpose(net2)
    gradient= 2 * learn_rate* l2_reg_val * net2
    return embedding_variable.assign(embedding_variable - gradient)


# ** TASK 2 ** BONUS part 2
def EmbeddingL1RegularizationUpdate(embedding_variable, net_input, learn_rate, l1_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.

    # net = tf.nn.l2_normalize(net_input, axis=1)
    # net = tf.transpose(net)
    # gradient = learn_rate * l1_reg_val * net
    # gradient = tf.math.sign(gradient)
    # embedding_variable.assign(embedding_variable - gradient)
    # return embedding_variable
    net = tf.nn.l2_normalize(net_input, axis=1)
    net1 = tf.transpose(net)
    net2 = tf.matmul(net, embedding_variable)
    embedding_variable1 = tf.math.sign(net2)
    gradient = l1_reg_val* tf.matmul(net1, embedding_variable1)
    # gradient = tf.gradients(tf.math.abs(l1_reg_val * tf.matmul(tf.transpose(net), embedding_variable)), embedding_variable)
    return embedding_variable.assign(embedding_variable - learn_rate*gradient)


# ** TASK 3
def SparseDropout(slice_x, keep_prob=0.3):
    """Sets random (1 - keep_prob) non-zero elements of slice_x to zero.

    Args:
        slice_x: 2D numpy array (batch_size, vocab_size)

    Returns:
        2D numpy array (batch_size, vocab_size)
    """
    # D1 = numpy.random.rand(slice_x[slice_x != 0].shape[0])
    # D1 = D1 < keep_prob
    # slice_x[slice_x != 0] = numpy.multiply(slice_x[slice_x != 0], D1)
    # return slice_x



    # indices_with_no_zero contain row,col index of non-zero slice_x values with row in [0] index and col in [1] index
    indices_with_no_zero = numpy.transpose(numpy.nonzero(slice_x))

    random_row = numpy.random.choice(numpy.arange(indices_with_no_zero.shape[0]), replace=False, size=int(indices_with_no_zero.shape[0] * (1 - keep_prob)))
    slice_x_index = indices_with_no_zero[random_row]
    slice_x[slice_x_index[:, 0], slice_x_index[:, 1]] = 0
    return slice_x


# ** TASK 4
# TODO(student): YOU MUST SET THIS TO GET CREDIT.
# You should set it to tf.Variable of shape (vocabulary, 40).
EMBEDDING_VAR = None


# ** TASK 5
# This is called automatically by VisualizeTSNE.
def ComputeTSNE(embedding_matrix):
    """Projects embeddings onto 2D by computing tSNE.

    Args:
        embedding_matrix: numpy array of size (vocabulary, 40)

    Returns:
        numpy array of size (vocabulary, 2)
    """

    embedding_matrix = TSNE(n_components=2, learning_rate=50.0, n_iter=500).fit_transform(embedding_matrix)
    return embedding_matrix


# ** TASK 5
# This should save a PDF of the embeddings. This is the *only* function marked
# marked with "** TASK" that will NOT be automatically invoked by our grading
# script (it will be "stubbed-out", by monkey-patching). You must run this
# function on your own, save the PDF produced by it, and place it in your
# submission directory with name 'tsne_embeds.pdf'.
def VisualizeTSNE(sess):
    global TERM_INDEX
    if EMBEDDING_VAR is None:
        print('Cannot visualize embeddings. EMBEDDING_VAR is not set')
        return
    embedding_mat = sess.run(EMBEDDING_VAR)
    tsne_embeddings = ComputeTSNE(embedding_mat)

    class_to_words = {
        'positive': [
            'relaxing', 'upscale', 'luxury', 'luxurious', 'recommend', 'relax',
            'choice', 'best', 'pleasant', 'incredible', 'magnificent',
            'superb', 'perfect', 'fantastic', 'polite', 'gorgeous', 'beautiful',
            'elegant', 'spacious'
        ],
        'location': [
            'avenue', 'block', 'blocks', 'doorman', 'windows', 'concierge', 'living'
        ],
        'furniture': [
            'bedroom', 'floor', 'table', 'coffee', 'window', 'bathroom', 'bath',
            'pillow', 'couch'
        ],
        'negative': [
            'dirty', 'rude', 'uncomfortable', 'unfortunately', 'ridiculous',
            'disappointment', 'terrible', 'worst', 'mediocre'
        ]
    }

    # TODO(student): Visualize scatter plot of tsne_embeddings, showing only words
    # listed in class_to_words. Words under the same class must be visualized with
    # the same color. Plot both the word text and the tSNE coordinates.
    print('visualization should generate now')
    # positive_word_list = class_to_words['positive']
    # location_word_list = class_to_words['location']
    # furniture_word_list = class_to_words['furniture']
    # negative_word_list = class_to_words['negative']
    # positive_indices = []
    # for d in positive_word_list:
    #     positive_indices.append(TERM_INDEX[d])
    # print(positive_indices)


    word_list = []
    label_list = []
    index_list = []
    colors = ['blue', 'green', 'red', 'orange']
    for key, val in TERM_INDEX.items():
        if key in class_to_words.get('positive'):
            label_list.append(1)
            index_list.append(val)
            word_list.append(key)
        elif key in class_to_words.get('location'):
            label_list.append(2)
            index_list.append(val)
            word_list.append(key)
        elif key in class_to_words.get('furniture'):
            label_list.append(3)
            index_list.append(val)
            word_list.append(key)
        elif key in class_to_words.get('negative'):
            label_list.append(4)
            index_list.append(val)
            word_list.append(key)
    tsne_embeddings = tsne_embeddings[index_list]
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=label_list, cmap=matplotlib.colors.ListedColormap(colors))
    for i in range(0, len(word_list)):
        plt.text(tsne_embeddings[i][0], tsne_embeddings[i][1], word_list[i])
    plt.savefig('tsne_embeds.pdf')


CACHE = {}


def ReadAndTokenize(filename):
    """return dict containing of terms to frequency."""
    global CACHE
    global VOCABULARY
    if filename in CACHE:
        return CACHE[filename]
    comment = open(filename).read()
    words = Tokenize(comment)

    terms = collections.Counter()
    for w in words:
        VOCABULARY[w] += 1
        terms[w] += 1

    CACHE[filename] = terms
    return terms


TERM_INDEX = None


def MakeDesignMatrix(x):
    global TERM_INDEX
    if TERM_INDEX is None:
        print('Total words: %i' % len(VOCABULARY.values()))
        min_count, max_count = numpy.percentile(list(VOCABULARY.values()), [50.0, 99.8])
        TERM_INDEX = {}
        for term, count in VOCABULARY.items():
            if count > min_count and count <= max_count:
                idx = len(TERM_INDEX)
                TERM_INDEX[term] = idx
    #
    x_matrix = numpy.zeros(shape=[len(x), len(TERM_INDEX)], dtype='float32')
    for i, item in enumerate(x):
        for term, count in item.items():
            if term not in TERM_INDEX:
                continue
            j = TERM_INDEX[term]
            x_matrix[i, j] = count  # 1.0    # Try count or log(1+count)
    return x_matrix


def GetDataset():
    """Returns numpy arrays of training and testing data."""
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    classes1 = set()
    classes2 = set()
    for f in GetInputFiles():
        class1, class2, fold, fname = f.split('/')[-4:]
        classes1.add(class1)
        classes2.add(class2)
        class1 = class1.split('_')[0]
        class2 = class2.split('_')[0]

        x = ReadAndTokenize(f)
        y = [int(class1 == 'positive'), int(class2 == 'truthful')]
        if fold == 'fold4':
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)

    ### Make numpy arrays.
    x_test = MakeDesignMatrix(x_test)
    x_train = MakeDesignMatrix(x_train)
    y_test = numpy.array(y_test, dtype='float32')
    y_train = numpy.array(y_train, dtype='float32')

    dataset = (x_train, y_train, x_test, y_test)
    with open('dataset.pkl', 'wb') as fout:
        pickle.dump(dataset, fout)
    return dataset


def print_f1_measures(probs, y_test):
    y_test[:, 0] == 1  # Positive
    positive = {
        'tp': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
        'fp': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
        'fn': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
    }
    negative = {
        'tp': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
        'fp': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
        'fn': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
    }
    truthful = {
        'tp': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
        'fp': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
        'fn': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
    }
    deceptive = {
        'tp': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
        'fp': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
        'fn': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
    }

    all_f1 = []
    for attribute_name, score in [('truthful', truthful),
                                  ('deceptive', deceptive),
                                  ('positive', positive),
                                  ('negative', negative)]:
        precision = float(score['tp']) / float(score['tp'] + score['fp'])
        recall = float(score['tp']) / float(score['tp'] + score['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        all_f1.append(f1)
        print('{0:9} {1:.2f} {2:.2f} {3:.2f}'.format(attribute_name, precision, recall, f1))
    print('Mean F1: {0:.4f}'.format(float(sum(all_f1)) / len(all_f1)))


def BuildInferenceNetwork(x, l2_reg_val, is_training):
    """From a tensor x, runs the neural network forward to compute outputs.
    This essentially instantiates the network and all its parameters.

    Args:
        x: Tensor of shape (batch_size, vocab size) which contains a sparse matrix
             where each row is a training example and containing counts of words
             in the document that are known by the vocabulary.

    Returns:
        Tensor of shape (batch_size, 2) where the 2-columns represent class
        memberships: one column discriminates between (negative and positive) and
        the other discriminates between (deceptive and truthful).
    """
    global EMBEDDING_VAR
    EMBEDDING_VAR = None  # ** TASK 4: Move and set appropriately.

    ## Build layers starting from input.
    net = x

    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_val)
    learning_rate = tf.placeholder_with_default(
        numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')

    ## First Layer
    net = FirstLayer(net, l2_reg_val, is_training)
    EMBEDDING_VAR = tf.trainable_variables()[0]
    # l_rate = 0.01
    # EMBEDDING_VAR = EmbeddingL1RegularizationUpdate(EMBEDDING_VAR, x, l_rate, l2_reg_val)

    ## Second Layer.
    net = tf.contrib.layers.fully_connected(
        net, 10, activation_fn=None, weights_regularizer=l2_reg)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.nn.relu(net)

    net = tf.contrib.layers.fully_connected(
        net, 2, activation_fn=None, weights_regularizer=l2_reg)

    return net


def main(argv):
    ######### Read dataset
    x_train, y_train, x_test, y_test = GetDataset()

    ######### Neural Network Model
    x = tf.placeholder(tf.float32, [None, x_test.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='y')
    is_training = tf.placeholder(tf.bool, [])

    l2_reg_val = 1e-6  # Co-efficient for L2 regularization (lambda)
    net = BuildInferenceNetwork(x, l2_reg_val, is_training)

    ######### Loss Function
    tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=net)

    ######### Training Algorithm
    learning_rate = tf.placeholder_with_default(
        numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # IPython.embed()

    def evaluate(batch_x=x_test, batch_y=y_test):
        probs = sess.run(net, {x: batch_x, is_training: False})
        print_f1_measures(probs, batch_y)

    def batch_step(batch_x, batch_y, lr):
        sess.run(train_op, {
            x: batch_x,
            y: batch_y,
            is_training: True, learning_rate: lr,
        })

    def step(lr=0.01, batch_size=100):
        indices = numpy.random.permutation(x_train.shape[0])
        for si in range(0, x_train.shape[0], batch_size):
            se = min(si + batch_size, x_train.shape[0])
            slice_x = x_train[indices[si:se]] + 0  # + 0 to copy slice
            slice_x = SparseDropout(slice_x)
            batch_step(slice_x, y_train[indices[si:se]], lr)

    lr = 0.05
    print('Training model ... ')
    for j in range(300): step(lr)
    for j in range(300): step(lr / 2)
    for j in range(300): step(lr / 4)
    print('Results from training:')
    evaluate()
    VisualizeTSNE(sess)


if __name__ == '__main__':
    tf.random.set_random_seed(0)
    main([])
