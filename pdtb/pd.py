import sqlite3

class PDTB:
    def __init__(self,file='pdtb.sqlite3'):
        self.conn = sqlite3.connect(file)
        self.conn.text_factory = str
        self.cursor = self.conn.cursor()

    def get_causal(self, section=None, file=None):
        query = "select relation, firstsemfirst, arg1rawtext,arg2rawtext from annotations_test where firstsemfirst like '%Cause%'"
        if section is not None and file is not None:
            query += " and section = ? and file = ? "
        query += " order by section,file"

        #print(query)
        if section is not None and file is not None:
            self.cursor.execute(query, (section, file))
        else:
            self.cursor.execute(query)

        return [Tag(*i) for i in self.cursor]

class WSJ:
    def __init__(self, file):
        self.file = file

    def read_relations(self):
        sentences = []
        with open(self.file) as f:
            for line in f:
                if '.START' in f or f == '\n':
                    continue
                sentences.append(line)
        return sentences

class TaggedPDTB:
    def __init__(self, file):
        self.file = file

    def read_relations(self):
        intra_tags = []
        adjacent_tags = []
        tags = intra_tags
        with open(self.file) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line == 'INTRA_SENTENCE':
                    continue
                if line == 'ADJACENT_SENTENCES':
                    tags = adjacent_tags

                r = line.split('\t')
                tags.append(Tag(None, r[0], '\t'.join(r[1:])))
        return intra_tags,adjacent_tags

class Tag:
    def __init__(self, relation, tag, text, text2=None):
        self.relation = relation
        self.tag = tag
        self.text = text
        self.text2 = text2

    def set_alt_tag(self, tag):
        self.alt_tag = tag

    def __repr__(self):
        return str(self.__dict__)
