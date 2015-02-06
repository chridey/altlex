class TaggedDataPoint(tuple):
    #data should be a dictionary
    def addData(self, data):
        self._data = data

    @property
    def data(self):
        return self._data
