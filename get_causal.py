import sys

output = ''
with open(sys.argv[1]) as f:
    for line in f:
        #print(line)
        section,fi = line.split('|')
        filename = 'Discourse3/data/wsj/{0}/wsj_{0}{1}'.format(section,fi).strip()
        with open(filename) as f2:
            output += f2.read()
print(output)
