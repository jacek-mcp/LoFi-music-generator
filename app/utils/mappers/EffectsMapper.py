import os


class EffectsMapper:

    def __init__(self):
        effects = os.listdir(os.getcwd() + '/data/effects/')

        lof = {}
        for e in effects:
            lof[e] = e
        self.list_of_effects = lof

    def get_effect_file_name(self, key):
        return self.list_of_effects[key]

    def get_all_effect_names(self):
        return list(self.list_of_effects.keys())
