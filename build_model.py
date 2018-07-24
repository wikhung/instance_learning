from keras.models import Model
from keras.layers import Input, Embedding, TimeDistributed, Dense, Lambda, Concatenate, Multiply
from keras.layers import Conv1D, MaxPooling1D, MaxPooling2D, Dropout, GlobalMaxPooling1D
from keras.losses import binary_crossentropy
from keras import regularizers
import keras.backend as K

import tensorflow as tf

# convenient convolution wrapper for modeling
def Dropout_Conv1D(layer, n_layer, ks, padding, activation, dropout_prob):
    x = Conv1D(256, ks, padding=padding, activation=activation, name="channel_{}".format(n_layer))(layer)
    x = Dropout(dropout_prob)(x)
    return x

def build_sent_encoder(max_num_words, max_num_sent, vocab_size, dropout_prob, embedding_dim, embedding_mat,
                     embedding_trainable):
    sent = Input((max_num_words,), name="sent_input")
    # vocab size + 2 for 0 padding and UNKNOWN TOKEN
    embed = Embedding(vocab_size + 2, embedding_dim, weights=[embedding_mat], trainable=embedding_trainable,
                      name="sent_embed")(sent)
    channels = [Dropout_Conv1D(embed, i, ks, "same", "relu", dropout_prob) for i, ks in enumerate([2, 3, 4, 6])]
    x = Concatenate()(channels)
    x = Conv1D(512, 5, padding="same", activation="relu")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout_prob)(x)
    x = Conv1D(512, 5, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout_prob)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(dropout_prob)(x)
    sent_encode = x

    sent_encoder = Model(inputs=sent, outputs=sent_encode)
    return sent_encoder

def build_model(sent_encoder, max_num_words, max_num_sent, dropout_prob):
    review = Input((max_num_sent, max_num_words))
    mask = Input((max_num_sent, 1))

    encoded_reviews = TimeDistributed(sent_encoder)(review)
    encoded_reviews = Dropout(dropout_prob)(encoded_reviews)

    # predictions on sentence sentiments
    y_hat = Dense(1, activation="sigmoid", name="sent_sentiment")(encoded_reviews)
    y_hat = Multiply(name="masked_sent_sentiment")([y_hat, mask])

    sent_avg_out = Lambda(lambda x: K.sum(x[0], axis=[-2]) / K.sum(x[1], axis=[-2]),
                          name="sent_agg_pred")([y_hat, mask])

    model = Model(inputs=[review, mask], outputs=sent_avg_out)

    return model, encoded_reviews, y_hat

# Pairwise sentence similarity for a batch
def pairwise_dist(x):
    new_shape = K.int_shape(x)[2:]
    x = K.reshape(x, (-1,) + new_shape)
    x1 = K.expand_dims(x, len(new_shape) - 1)
    x2 = K.expand_dims(x, len(new_shape))
    sq_diff = K.square(x1 - x2)
    c = K.sqrt(K.sum(sq_diff, axis = -1))
    c = c / (2 * (0.5 ** 2))
    sims = K.exp(-c)
    return tf.matrix_band_part(sims, -1, 0)

# Pairwise prediction difference for a batch
def y_derivative(y):
    new_shape = K.int_shape(y)[2:]
    y = K.reshape(y, (-1,) + new_shape)
    y1 = K.expand_dims(y, len(new_shape) - 1)
    y2 = K.expand_dims(y, len(new_shape))
    sq_diff = K.square(y1 - y2)
    return tf.matrix_band_part(sq_diff, -1, 0)

# similarity loss function
def custom_sim_loss(encoded_reviews, y_hat, batch_size):
    sims, pred_sims = pairwise_dist(encoded_reviews), y_derivative(y_hat)
    loss = K.sum(K.dot(sims, pred_sims)) / (batch_size ** 2)
    return loss

# Full custom loss function
def custom_loss_wrapper(encoded_reviews, y_hat, batch_size, l, a):
    def loss(y_true, y_pred):
        ent_loss = binary_crossentropy(y_true, y_pred)
        ent_loss = K.reshape(ent_loss, (-1, 1))
        sim_loss = custom_sim_loss(encoded_reviews, y_hat, batch_size)
        sim_loss = K.reshape(ent_loss, (-1, 1))
        return (l * ent_loss) +  (a * sim_loss)
    return loss