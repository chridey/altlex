import nltk

class NaiveBayes:
    def train(self, training):
        self.model = nltk.NaiveBayesClassifier.train(training)
        return self.model
    
    def classify(self, features):
        return self.model.classify(features)

    def accuracy(self, testing):
        return nltk.classify.accuracy(self.model, testing)

    def show_most_informative_features(self, n=50):
        return self.model.show_most_informative_features(n)
