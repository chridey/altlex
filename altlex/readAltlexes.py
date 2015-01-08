def readAltlexes(altlexFile):
    with open(altlexFile) as f:
        altlexes = f.read().splitlines()

    for i in range(len(altlexes)):
        altlexes[i] = altlexes[i].replace("'s", " 's")
        altlexes[i] = altlexes[i].replace(":", " :")
        altlexes[i] = altlexes[i].replace(",", " ,")
        altlexes[i] = altlexes[i].replace('"', ' "')

    return altlexes

def matchAltlexes(sentence, altlexes):
    mx = 0
    argmax = None

    for a in altlexes:
        alist = a.split()
        if sentence[:len(alist)] == alist:
            if len(alist) > mx:
                mx = len(alist)
                argmax = alist

    return argmax
