from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet  as wn

class WordNetManager:
    @property
    def wordNetPOS(self):
        return {'V': wn.VERB,
                'N': wn.NOUN,
                'J': wn.ADJ,
                'R': wn.ADV}

    def wordCategory(self, lemma, pos):
        try:
            synsets = wn.synsets(lemma,
                                 pos=self.wordNetPOS[pos[0]])
        except UnicodeDecodeError:
            synsets = []
            
        if len(synsets):
            return synsets[0].lexname
        else:
            return None

    def distance(self, word1, word2, pos=None):
        if pos is None:
            items = self.wordNetPOS.items()
        else:
            items = [self.wordNetPOS[pos]]
            
        maxy = 0
        for pos in items:
            try:
                word1synset = wn.synset('{}.{}.01'.format(word1, pos))
            except WordNetError:
                continue

            try:
                word2synset = wn.synset('{}.{}.01'.format(word2, pos))
            except WordNetError:
                continue

            r = word1synset.path_similarity(word2synset)
            if r is not None and r > maxy:
                maxy = r

        return maxy

