from model import InstanceLearning

if __name__ == "__main__":
    instance_learning_model = InstanceLearning(vocab_size=3000, max_sent=7, max_len=30, embedding_dim=300)
    instance_learning_model.read_review_data(nrows=200000)
    instance_learning_model.process_data()
    instance_learning_model.get_pretrained_embeddings()
    instance_learning_model.get_models(dropout_prob=0.4, embedding_trainable=True)
    instance_learning_model.train(epochs=10, save_weights=True)