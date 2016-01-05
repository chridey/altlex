from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()

def lemmatize(words, poses):
    lemmas = []
    for index,word in enumerate(words):
        if poses[index].startswith('RB'):
            pos = wn.ADV
        elif poses[index].startswith('JJ'):
            pos = wn.ADJ
        elif poses[index].startswith('NN'):
            pos = wn.NOUN
        elif poses[index].startswith('VB'):            
            pos = wn.VERB
        else:
            lemmas.append(word)
            continue
        lemmas.append(lemmatizer.lemmatize(word, pos=pos))
    return lemmas
