import xml.etree.ElementTree as ET

class XMLReader:
    def __init__(self, xml_file_name):
        self.file_name = xml_file_name

    def iterFiles(self):
        with open(self.file_name) as f:
            root = ET.parse(f).getroot()
            assert(root.tag == 'root')
            for document in root:
                yield document

