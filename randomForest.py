from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=10)
        self.featureMap = None

    def _transform(self, features):
        if self.featureMap is None:
            self.featureMap = {}
            counter = 0
            for feature in features:
                for featureName in feature:
                    if featureName not in self.featureMap:
                        self.featureMap[featureName] = counter
                        counter += 1
        else:
            counter = len(self.featureMap)

        X = []
        for feature in features:
            x = [0] * counter
            for featureName in feature:
                if featureName in self.featureMap:
                    x[self.featureMap[featureName]] = float(feature[featureName])
            X.append(x)
            
        return X
        
    def train(self, training):
        features, Y = zip(*training)
        X = self._transform(features)
        self.model = self.classifier.fit(X, Y)
        return self.model
    
    def classify(self, features):
        assert(type(features) == dict)
        X = self._transform([features])
        result = self.model.predict(X)
        return result[0]
    
    def accuracy(self, testing):
        features, Y = zip(*testing)
        X = self._transform(features)
        return self.model.score(X,Y)

    def show_most_informative_features(self, n=50):
        pass
