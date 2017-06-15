import numpy as np
from numpy import array
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Dropout, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import backend as K

def norm(vectors):
    """Takes the L2-norm of each row in vectors
    Args:
        vectors: a 2-D Tensor with shape (M, N)
    Returns:
        a 1-D Tensor of length M
    """
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=1))

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Dense(20, input_shape=(input_dim,), activation='linear'))
    seq.add(Dropout(0.1))
    seq.add(Dense(20, activation='linear'))
    seq.add(Dropout(0.1))
    seq.add(Dense(20, activation='softmax'))

    return seq

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

def nn(train_data, epochs=20):
    train_data = array(train_data)
    input_dim = 30

    tr_l = np.zeros(30)
    tr_r = np.zeros(30)
    for i in range(train_data.shape[0]):
            tr_l = np.vstack([tr_l, train_data[i][0]])
            tr_r = np.vstack([tr_r, train_data[i][1]])
           

    tr_l = np.delete(tr_l, 1, 0)
    tr_r = np.delete(tr_r, 1, 0)
    tr_y = train_data[:, 2]

    # Build The Model
    base_network = create_base_network(input_dim)

    input_a = Input(shape=(input_dim, ))
    input_b = Input(shape=(input_dim, ))

    # Simease Network so we're doing two layer stuff
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
   
    model = Model([input_a, input_b], distance)

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)

    model.fit([tr_l, tr_r], tr_y,
          batch_size=128,
          epochs=epochs,
          )
    # compute final accuracy on training and test sets
    pred = model.predict([tr_l, tr_r])
    tr_acc = compute_accuracy(pred, tr_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
