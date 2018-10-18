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

    # Append multiple n-gram like filters together
    channels = [Dropout_Conv1D(embed, i, ks, "same", "relu", dropout_prob) for i, ks in enumerate([2, 3, 4, 6])]
    x = Concatenate()(channels)

    # Couple Convolution-MaxPooling-Dropout block
    x = Conv1D(512, 5, padding="same", activation="relu")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout_prob)(x)
    x = Conv1D(512, 5, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout_prob)(x)
    # Global MaxPooling for the last layer
    x = GlobalMaxPooling1D()(x)
    x = Dropout(dropout_prob)(x)
    sent_encode = x

    # return the sentence encoder
    sent_encoder = Model(inputs=sent, outputs=sent_encode)
    return sent_encoder

def build_model(sent_encoder, max_num_words, max_num_sent, dropout_prob):
    review = Input((max_num_sent, max_num_words))
    mask = Input((max_num_sent, 1))

    # Stitch the sentence encoder together
    encoded_reviews = TimeDistributed(sent_encoder)(review)
    encoded_reviews = Dropout(dropout_prob)(encoded_reviews)

    # predictions on sentence sentiments
    y_hat = Dense(1, activation="sigmoid", name="sent_sentiment")(encoded_reviews)
    # multiply the mask to remove padded sentences from prediction
    y_hat = Multiply(name="masked_sent_sentiment")([y_hat, mask])

    # use average sentence predictions as the overall review predictions
    sent_avg_out = Lambda(lambda x: K.sum(x[0], axis=[-2]) / K.sum(x[1], axis=[-2]),
                          name="sent_agg_pred")([y_hat, mask])

    model = Model(inputs=[review, mask], outputs=sent_avg_out)
    return model, encoded_reviews, y_hat

# pairwise difference
def pairwise_dist(x):
    encoding_len = K.int_shape(x)[-1]
    x1 = K.reshape(x, (-1, encoding_len))
    x2 = K.reshape(x1, (-1, 1, encoding_len))
    l2_norm = K.sqrt(K.maximum(K.sum(K.square(x1 - x2), axis=-1), K.epsilon()))
    c = l2_norm
    sims = K.exp(-c)
    return sims

# prediction difference
def y_derivative(y):
    y1 = K.reshape(y, (-1,))
    y2 = K.reshape(y1, (-1, 1))
    sq_diff = K.maximum(K.square(y1 - y2), K.epsilon())
    return sq_diff

# similarity loss function
def custom_sim_loss(encoded_reviews, y_hat, batch_size, num_sent):
    sims, pred_sims = pairwise_dist(encoded_reviews), y_derivative(y_hat)
    loss = K.sum(tf.multiply(sims, pred_sims))  / ((batch_size * num_sent) ** 2)
    return loss

# Full custome loss function
def custom_loss_wrapper(encoded_reviews, y_hat, batch_size, num_sent, l, a):
    def loss(y_true, y_pred):
        ent_loss = binary_crossentropy(y_true, y_pred)
        sim_loss = custom_sim_loss(encoded_reviews, y_hat, batch_size, num_sent)
        return sim_loss + (l * ent_loss)
    return loss
t_loss) +  (a * sim_loss)
    return loss