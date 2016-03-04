defaultConfig = {
    'framenetSettings' : None,
    'featureSettings': {'kld_score': True,
                        'framenet': True,
                        'head_word_cat_curr': True,
                        'head_word_cat_prev': True,
                        'head_word_cat_altlex': True,
                        'head_word_verbnet_curr': True,
                        'head_word_verbnet_prev': True,
                        'head_word_verbnet_altlex': True,
                        'arguments_cat_curr': True,
                        'arguments_cat_prev': True,
                        'arguments_verbnet_curr': True,
                        'arguments_verbnet_prev': True
                        },
    'kldSettings': None,
    }

class Config:
    def __init__(self, settings=defaultConfig):
        self.settings = settings

    @property
    def KLDSettings(self):
        return self.settings['kldSettings']

    @property
    def framenetSettings(self):
        return self.settings['framenetSettings']

    @property
    def featureSettings(self):
        return self.settings['featureSettings']
