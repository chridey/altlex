class MultiWordCache:
    def __init__(self):
        self._cache = {}

    def lookup(self, *words):
        l = ''
        for word in words:
            l += '_' + word

        return self._cache.get(l)

    def update(self, value, *words):
        l = ''
        for word in words:
            l += '_' + word

        self._cache[l] = value
        return True

    def clear(self):
        self._cache = {}

class MultiLevelCache:
    def __init__(self):
        self._dictLookup = []
        self._cache = []

    def lookup(self, *words):
        curr = self._cache
        for index,word in enumerate(words):
            try:
                wordIndex = self._dictLookup[index].get(word)
            except IndexError:
                return None
            if wordIndex is None:
                return None

            curr = curr[wordIndex]

        return curr

    def update(self, value, *words):
        curr = self._cache
        for index,word in enumerate(words):
            try:
                wordIndex = self._dictLookup[index].get(word)
            except IndexError:
                self._dictLookup.append({})
                wordIndex = None
            print(wordIndex)
            if wordIndex is None:
                self._dictLookup[index][word] = len(curr)
                wordIndex = self._dictLookup[index][word]
                curr.append([])

            print(index, word, curr, wordIndex)
            prev = curr
            curr = curr[wordIndex]

        prev[wordIndex] = value
        return True
