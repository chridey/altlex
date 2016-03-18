from __future__ import print_function

import sys
import os
import collections
import re
import math

from altlex.utils import wordUtils

r = re.compile('^\s+$')

class FrameNetManager:
    def __init__(self,
                 framescoresDir=os.path.join(os.path.join(os.path.dirname(__file__),
                                                          '..'),
                                             'config'),
                 binary=True,
                 verbose=False):

        self.framenetScores = None
        self.framescoresDir = framescoresDir
        self.binary = binary
        self.verbose = verbose

    def loadFramenetScores(self):
        self.framenetScores = {}
        
        if self.verbose:
            print('loading framenet scores...')
            
        for pos in os.listdir(os.path.join(self.framescoresDir,
                                           'framenet')):
            if pos not in self.framenetScores:
                self.framenetScores[pos] = collections.defaultdict(float)
                with open(os.path.join(os.path.join(self.framescoresDir,
                                                    'framenet'), pos)) as f:
                    for line in f:
                        try:
                            p,word,count1,count2,score,entropy = line.split()
                        except Exception:
                            print(line)
                        score = float(entropy)
                        if score > 0.0:
                            self.framenetScores[pos][word] += score

    def score(self, stems, poses, suffix=''):
        if self.framenetScores is None:
            self.loadFramenetScores()
        
        #sum of weights of encoding causality for words for different parts of speech
        #stem and lowercase
        score = collections.defaultdict(float)

        for i in range(len(stems)):
                pos = poses[i][:2].lower()
                stem = stems[i]
                
                if self.binary:
                    full_pos = pos + '_causal'
                    if full_pos in self.framenetScores and stem in self.framenetScores[full_pos]:
                        score[suffix + '_causal'] += self.framenetScores[full_pos][stem]
                else:
                    for causalType in ('_reason', '_result'):
                        if pos + causalType in self.framenetScores and stem in self.framenetScores[pos + causalType]:
                            score[suffix + causalType] += self.framenetScores[pos + causalType][stem]
                        
                pos += '_anticausal'
                if pos in self.framenetScores and stem in self.framenetScores[pos]:
                    score[suffix + '_anticausal'] += self.framenetScores[pos][stem]

        return score

    @staticmethod
    def isCausalFrame(frame):
        frame = frame.lower()
        if 'caus' in frame or frame in ('reason', 'explanation', 'effect', 'trigger') or 'purpose' in frame or 'required' in frame or 'consequence' in frame or 'result' in frame or 'response' in frame or 'enabled' in frame:
            return True
        else:
            return False

    @staticmethod
    def isReasonFrame(frame):
        frame = frame.lower()
        if 'caus' in frame in ('reason', 'explanation', 'trigger') or 'purpose' in frame or 'required' in frame:
            return True
        else:
            return False

    @staticmethod
    def isResultFrame(frame):
        frame = frame.lower()
        if 'effect' in frame or 'consequence' in frame or 'result' in frame or 'response' in frame or 'enabled' in frame:
            return True
        else:
            return False

    @staticmethod
    def isAntiCausalFrame(frame):
        if frame in {
            'Requirements',
            'Have_as_requirement',
            'Statement',
            'Occasion',
            'Supplier',
            'Thriving',
            'Seeking',
            'Expectation',
            'Being_necessary',
            'Clarity_of_resolution',
            'Circumstances',
            'Relative_time',
            'Idiosyncrasy',
            'Compliance',
            'Aggregate',
            'Estimation',
            'Bringing',
            'Perception_active',
            'Responding_entity',
            'Evaluative_comparison',
            'Justifying',
            'Waiting',
            'Taking_sides',
            'Questioning',
            'Proposed_action',
            'Evaluee',
            'Descriptor',
            'Evidence',
            'Defend',
            'Activity_ongoing',
            'Boundary',
            'Coming_up_with',
            'Interval',
            'Explaining_the_facts',
            'Communicate_categorization',
            'Terms_of_agreement',
            'Requirement',
            'Process',
            'State',
            'Remembering_experience',
            'Defender',
            'Categorization',
            'Indicating',
            'Appointing',
            'Consideration',
            'Depictive',
            'Coming_to_believe',
            'Proposition',
            }:
            return True
        else:
            return False

    @staticmethod
    def makeFramenetScores(corpusdir, outdir, verbose=False):
        counts = {}
        for pos in ('nn', 'vv', 'rb', 'jj'):
            counts[pos] = collections.defaultdict(int)
            counts[pos + '_causal'] = collections.defaultdict(int)
            counts[pos + '_reason'] = collections.defaultdict(int)
            counts[pos + '_result'] = collections.defaultdict(int)
            counts[pos + '_anticausal'] = collections.defaultdict(int)

        #look for nn/vv/rb/jj
        files = os.listdir(corpusdir)
        if verbose:
            print(len(files))
        for fi in files:
            with open(os.path.join(corpusdir, fi)) as f:
                if verbose:
                    print(fi)
                for line in f:
                    if r.match(line) or line.startswith('SKIPPED') or line.startswith('#'):
                        continue

                    cols = line.split()
                    word = wordUtils.snowballStemmer.stem(wordUtils.replaceNonAscii(cols[2].lower()))
                    pos = cols[3].lower()

                    if pos.startswith('n'): #'nn' in pos or 'np' in pos:
                        pos = 'nn'
                    elif pos.startswith('v'): #'vv' in pos or 'vb' in pos:
                        pos = 'vv'
                    elif 'rb' in pos:
                        pos = 'rb'
                    elif 'jj' in pos:
                        pos = 'jj'
                    else:
                        continue

                    for role in cols[11:]:
                        if FrameNetManager.isCausalFrame(role):
                            counts[pos + '_causal'][word] += 1
                        if FrameNetManager.isReasonFrame(role):
                            counts[pos + '_reason'][word] += 1
                            break
                        if FrameNetManager.isResultFrame(role):
                            counts[pos + '_result'][word] += 1
                            break
                        if FrameNetManager.isAntiCausalFrame(role):
                            counts[pos + '_anticausal'][word] += 1
                            break
                    
                    counts[pos][word] += 1

        for outputType in ('_causal', '_reason', '_result', '_anticausal'):
            for pos in ('nn', 'vv', 'rb', 'jj'):
                if verbose:
                    print ('{}{}'.format(pos, outputType))
                full_pos = pos + outputType
                with open(os.path.join(outdir, full_pos), 'w') as f:
                    for word in sorted(counts[pos],
                                       key=lambda x:counts[full_pos][x]*1.0/counts[pos][x]*math.log(counts[pos][x]),
                                       reverse=True):
                        if counts[pos][word] > 0:
                            total = counts[pos][word]
                            total_pos = counts[full_pos][word]
                            fraction = total_pos*1.0/total
                            print(pos,
                                  word,
                                  total,
                                  total_pos,
                                  fraction,
                                  fraction*math.log(total),
                                  file=f)
                if verbose:
                    print ('Total: {}'.format(sum(counts[full_pos][word] for word in counts[full_pos])))
                        
        
if __name__ == '__main__':
    indir = sys.argv[1]
    outdir = sys.argv[2]

    FrameNetManager.makeFramenetScores(indir, outdir, True)
