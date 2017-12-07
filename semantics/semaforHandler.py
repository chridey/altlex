import json
import socket

def to_conll(index, word, pos, head, dep):
    '''
    Create a string for a token and metadata in CONLL format:
    1       My      _       PRP$    PRP$    _       2       NMOD    _       _
    2       kitchen _       NN      NN      _       5       SBJ     _       _
    3       no      _       RB      RB      _       5       ADV     _       _
    4       longer  _       RB      RB      _       3       AMOD    _       _
    5       smells  _       VBZ     VBZ     _       0       ROOT    _       _
    6       .       _       .       .       _       5       P       _       _
    
    '''
    return u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'.format(index+1,
                                                                       word,
                                                                       '_',
                                                                       pos,
                                                                       pos,
                                                                       '_',
                                                                       head+1,
                                                                       dep,
                                                                       '_',
                                                                       '_')

def spacy_sent_to_conll(sent, offset=0):
    conll_string = ''
    for idx, token in enumerate(sent):
        word = token.orth_
        pos = token.tag_
        dep = token.dep_
        head = token.head.i-offset
        if token.dep_ == u'ROOT':
          head = -1
        conll_string += to_conll(idx, word, pos, head, dep)
    # conll_string += '\n'
    return conll_string[:-1]

class SemaforHandler:

    def __init__(self):
        raise NotImplementedError

    def parse(self, conll_string):
        raise NotImplementedError
    
    def get_frames(self, sent):
        offset = list(sent)[0].i
        return self.get_frames_from_conll_string(spacy_sent_to_conll(sent, offset))
    
    def get_frames_from_conll_string(self, conll_string):
        frames = self.parse(conll_string)
        sas = [SemaforAnnotation(**frame) for frame in frames]
        sa = sas[0]
        sent_frame = [None for i in range(len(sa.tokens))]
        for frame, j in sa.iter_targets():
            sent_frame[j] = frame
        return sent_frame

class TCPClientSemaforHandler(SemaforHandler):
    
    def __init__(self, host='localhost', port=8888): #host='128.59.10.130', port=8898):
        self.host = host
        self.port = port

    def parse(self, conll_string):
        
        sock = socket.create_connection((self.host, self.port))
        sock.sendall(conll_string.encode('utf-8'))
        #sleep?
        sock.shutdown(socket.SHUT_WR)
        frame_string = ''
        while True:
            data = sock.recv(1024)
            # print 'data: ', data
            if data == u'':
                break
            frame_string += data
        ret = []
        
        # print 'frame string: ', frame_string
        tokens = [i.split('\n') for i in conll_string.split('\n\n')]
        # print 'tokens: ', tokens
        
        for index, line in enumerate(frame_string.splitlines()):
            # print index, line
            frames = json.loads(line)
            if type(frames) == dict:
                ret.append(frames)
                continue
            # tmp = tokens[index][:-2]
            # print tmp
            frames_new = {'tokens': [i.split('\t')[1] for i in tokens[index]],
                          'frames': []}
            for frame in frames:
                spans = []
                span = {}
                for location in sorted(frame['first']):
                    if not len(span):
                        span['start'] = location
                        span['end'] = location+1
                        span['text'] = frames_new['tokens'][location]
                    else:
                        if location > span['end']:
                            spans.append(span)
                            span = {'start': location,
                                    'end': location+1,
                                    'text': frames_new['tokens'][location]}
                        else:
                            span['end'] = location+1
                            span['text'] += ' ' + frames_new['tokens'][location]
                    
                spans.append(span)
                frames_new['frames'].append({'target': {'name': frame['second'],
                                                        'spans': spans}})
            ret.append(frames_new)
        return ret

class SemaforAnnotation:
    
    def __init__(self, frames=None, tokens=None, error=None):
        self.frames = []
        if frames is not None:
            self.frames = frames
            
        self.tokens = []
        if tokens is not None:
            self.tokens = tokens
            
        self.error = error
        # print frames
        # print tokens
        
    def iter_targets(self):
        for frame in self.frames:
            target = frame['target']
            frame_name = target['name']
            for span in target['spans']:
                for r in range(span['start'], span['end']):
                    yield frame_name, r

