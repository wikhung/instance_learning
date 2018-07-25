import pandas as pd
import numpy as np
import os
import utils
from collections import Counter
from nltk import tokenize
from build_model import build_sent_encoder, build_model, custom_loss_wrapper
import logging

from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

class InstanceLearning(object):

    def __init__(self, vocab_size=None, embedding_dim=None, max_sent=None, max_len=None, batch_size=128, lr=0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sent = max_sent
        self.max_len = max_len
        self.batch_size = batch_size
        self.pretrained_embeddings = None
        self.learning_rate = lr

    def read_review_data(self, nrows = None):
        logger.info("Read in {} yelp reviews".format(nrows if nrows is not None else "all"))
        df = pd.read_csv(os.path.join("data", "yelp_reviews", "yelp_review.csv"), nrows=nrows)

        df = df[["text", "stars"]]
        # Remove neutral reviews
        df = df[df['stars'] != 3]

        self.reviews = df["text"]
        self.labels = df["stars"]

    def process_data(self):
        logger.info("Prep the training data")
        # Remove all unneeded symbols from reviews
        self.reviews = self.reviews.apply(lambda s: utils.remove_symbols(s))

        # Count the number of words in the reviews
        vocab = Counter([word for review in self.reviews for word in tokenize.word_tokenize(review)])

        # Only keep the top n words to build a word2idx dictionary
        # 0 will be reserved for padding
        self.word2idx = {k: i + 1 for i, (k, _) in enumerate(vocab.most_common(self.vocab_size))}

        # tokenized the reviews; UNKNOWN_TOKEN = vocab_size + 1
        self.tokenized_reviews = self.reviews.apply(lambda r: utils.review_tokenizer(r, self.word2idx, self.max_sent,
                                                            self.max_len, self.vocab_size + 1))
        # reshape the data
        self.tokenized_reviews = np.stack(self.tokenized_reviews.values, axis=0)

        # generate a mask for padded sentences
        self.mask_mat = np.sum(self.tokenized_reviews, axis=-1).reshape(-1, self.max_sent, 1)
        self.mask_mat[self.mask_mat > 0] = 1

        # recode the data for training
        self.labels = self.labels.apply(lambda x: 1 if x > 3 else 0).values

    def get_pretrained_embeddings(self):
        logger.info("Prep the pretrained word embeddings")
        f = open(os.path.join("data", "embeddings", "wiki-news-300d-1M.vec"), encoding="utf8")
        contents = f.readlines()
        self.pretrained_embeddings = utils.parse_word_vectors(contents, self.word2idx,
                                                              self.vocab_size, self.embedding_dim)
        f.close()

    def get_models(self, dropout_prob, embedding_trainable, grp_w=0.5, sim_w=1):
        sent_encoder = build_sent_encoder(self.max_len, self.max_sent, self.vocab_size, dropout_prob,
                                          self.embedding_dim, self.pretrained_embeddings, embedding_trainable)

        model, encoded_reviews, y_hat = build_model(sent_encoder, self.max_len, self.max_sent, dropout_prob)

        sent_sentiment_layer = model.get_layer("masked_sent_sentiment")
        sent_sentiment_model = Model(inputs=model.input,
                                     outputs=sent_sentiment_layer.output)

        model.compile(Adam(),
                      loss=custom_loss_wrapper(encoded_reviews, y_hat, self.batch_size, grp_w, sim_w),
                      metrics=["accuracy"])
        self.sent_encoder = sent_encoder
        self.model=model
        self.sent_sentiment_model = sent_sentiment_model

    def train(self, epochs, save_weights = False):
        lr_scheduler = LearningRateScheduler(self.decaying_lr)
        earlystopping = EarlyStopping(patience=2)

        callbacks = [lr_scheduler, earlystopping]
        if save_weights:
            model_check_pt = ModelCheckpoint(os.path.join("checkpoint", "weights.{epoch:02d}-{val_loss:.2f}.hdf5"))
            callbacks.append(model_check_pt)

        self.model.fit([self.tokenized_reviews, self.mask_mat],
                       self.labels, self.batch_size, validation_split=0.2,
                       epochs=epochs, callbacks=callbacks)

    def decaying_lr(self, epoch):
        return self.learning_rate / (2 ** epoch)

    def demo(self):
        review = input("Enter the reviews here:")
        processed_review = utils.remove_symbols(review)
        tokenized_review = utils.review_tokenizer(processed_review, self.word2idx, self.max_sent,
                                                  self.max_len, self.vocab_size + 1)
        mask_mat = np.sum(tokenized_reviews, axis=-1).reshape(-1, self.max_sent, 1)
        mask_mat[mask_mat > 0] = 1

        sent_sentiment_pred = self.sent_sentiment_model.predict([tokenized_review, mask_mat])


        #print("Review ratings: {}".format(labels.iloc[i]))
        for s_i, sent in enumerate(tokenized_review[:self.max_sent]):
            sent_pred = sent_sentiment_pred[s_i]

            if sent_pred == 0:
                break
            elif sent_pred > 0.7:
                print('\033[1;42m{}\033[1;m'.format(sent))
            elif sent_pred < 0.3:
                print('\033[1;41m{}\033[1;m'.format(sent))
            else:
                print('\033[1;47m{}\033[1;m'.format(sent))
        print("\n")