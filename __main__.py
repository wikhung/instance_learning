from model import InstanceLearning

if __name__ == "__main__":
    instance_learning_model = InstanceLearning(vocab_size=30, max_sent=5, max_len=30, embedding_dim=300)
    instance_learning_model.read_review_data(nrows=1000)
    instance_learning_model.process_data()
    instance_learning_model.get_pretrained_embeddings()
    instance_learning_model.get_models(dropout_prob=0.4, embedding_trainable=True)
    print(instance_learning_model.sent_encoder.summary())

    #instance_learning_model.train(1)