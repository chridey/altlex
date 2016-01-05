import xml.etree.ElementTree as ET
import gzip

class XMLReader:
    def __init__(self, xml_file_name):
        self.file_name = xml_file_name
        self.root = 'root'
        if xml_file_name.endswith('.gz'):
            self.open = gzip.open
            self.flags = 'rb'
        else:
            self.open = open
            self.flags = 'r'
    def iterFiles(self):
        with self.open(self.file_name, self.flags) as f:
            root = ET.parse(f).getroot()
            assert(root.tag == self.root)
            for document in root:
                yield document

