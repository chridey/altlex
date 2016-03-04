import os
import collections

from altlex.utils import wordUtils

class FrameNetManager:
    def __init__(self,
                 framescoresDir=os.path.join(os.path.join(os.path.dirname(__file__),
                                                          '..'),
                                             'config'),
                 verbose=False):

        self.framenetScores = None
        self.framescoresDir = framescoresDir
        self.verbose = verbose
        
    def makeFramenetScores(self, corpus, outdir):
        with open(os.path.join(os.path.join(outdir,
                                            'framenet'), pos)) as f:
            pass

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
                            self.framenetScores[pos][wordUtils.snowballStemmer.stem(word.lower())] \
                                += score

    def score(self, stems, poses, binary=True, trinary=False, suffix=''):
        if self.framenetScores is None:
            self.loadFramenetScores()
        
        #sum of weights of encoding causality for words for different parts of speech
        #stem and lowercase
        score = collections.defaultdict(float)

        for i in range(len(stems)):
                pos = poses[i][:2].lower()
                stem = stems[i]
                
                if binary:
                    if pos in self.framenetScores and stem in self.framenetScores[pos]:
                        score[suffix] += self.framenetScores[pos][stem]

                if trinary:
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
        
if __name__ == '__main__':
    pass
    #make framenet scores
    #TODO
