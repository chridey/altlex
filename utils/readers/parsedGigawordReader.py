import os
import tarfile
import xml.etree.ElementTree as ET

class ParsedGigawordReader:
    def __init__(self, gigaword_dir):
        self.gigaword_dir = gigaword_dir

    def iterFiles(self):
        for t in os.listdir(self.gigaword_dir):
            #print(t)
            with tarfile.open(os.path.join(self.gigaword_dir,t), "r:gz") as tf:
                for ti in tf:
                    print(ti.name)
                    if ti.name.endswith('.xml'):
                        f = tf.extractfile(ti)
                        try:
                            root = ET.parse(f).getroot()
                        except IOError as e:
                            print(e)
                            continue
                        assert(root.tag == 'root')
                        for document in root:
                            yield document
                        f.close()
