import os
import tarfile

#dont care if its well-formatted xml
class FastParsedGigawordReader:
    def __init__(self, gigaword_dir):
        self.gigaword_dir = gigaword_dir

    def iterFiles(self):
        for t in os.listdir(self.gigaword_dir):
            #print(t)
            with tarfile.open(os.path.join(self.gigaword_dir,t), "r:gz") as tf:
                try:
                    for ti in tf:
                        print(ti.name)
                        if ti.name.endswith('.xml'):
                            try:
                                f = tf.extractfile(ti)
                            except IOError as e:
                                print(e)
                                continue
                            inRoot = False
                            inDocument = False
                            document = []
                            #try:
                            #    line = f.readline()
                            #except IOError as e:
                            #    print(e)
                            #    continue
                            #if '<root>' in line:
                            #    inRoot = True
                            try:
                                for line in f:
                                    if type(line) == bytes:
                                        line = line.decode('latin-1')
                                    #if '<root>' in line:
                                    #    inRoot = True
                                    #el
                                    if '<document>' in line:
                                        inDocument = True
                                    elif '</document>' in line:
                                        inDocument = False
                                        yield document
                                        document = []
                                    elif inDocument:
                                        document.append(line)
                            except IOError as e:
                                print(e)
                                continue
                            f.close()
                except IOError:
                    print(e)
