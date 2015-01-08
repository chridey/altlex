import sqlite3
import sys

import chnlp.pdtb.pd

if __name__ == '__main__':
    p = pd.PDTB()
    t = pd.TaggedPDTB(sys.argv[1])
    if len(sys.argv) > 2:
        causes = p.get_causal(sys.argv[2], sys.argv[3])
    else:
        causes = p.get_causal()
    #print(len(causes))
    intra_tags,adjacent_tags = t.read_relations()
    
    i=0
    j=0
    tagged_causes = []
    tags = intra_tags
    #print(len(tags))
    while True:
        #print(i,j)
        #print('{}:{}'.format(causes[i].text, tags[j].text))
        
        #for each causal relation, iterate over the text until we find a match
        if causes[i].text in tags[j].text and causes[i].text in tags[j].text.split('\t')[0] and causes[i].text2 in tags[j].text:
            #might get false hit, make sure that if its two sentences, it occurs in the first sentence
            print('Match {}:{}'.format(causes[i].text, tags[j].text))
            tag = pd.Tag(**tags[j].__dict__)
            tag.set_alt_tag(causes[i].tag)
            tag.relation = causes[i].relation
            i += 1
            j+=1
            #j=0
            tagged_causes.append(tag)
        else:
            j += 1
        
        if j >= len(tags) or i >= len(causes):
            #print("ALSO TRUE")
            if tags==intra_tags:
                #print("TRUE!!!")
                tags = adjacent_tags
                i=0
                j=0
            else:
                break

    for t in tagged_causes:
        print(t)
