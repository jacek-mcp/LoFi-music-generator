class EffectsMapper:

    def __init__(self):
        self.list_of_effects = {
            "rain with thunder": "rain_thunder.wav",
            "slow jam beat": "slow_jam_beat.wav",
            "lofi-hip-hop-tutorial-bass.wav": "lofi-hip-hop-tutorial-bass.wav",
            "lofi-hip-hop-tutorial-voice.wav": "lofi-hip-hop-tutorial-voice.wav"
        }

    def get_effect_file_name(self, key):
        return self.list_of_effects[key]

    def get_all_effect_names(self):
        return list(self.list_of_effects.keys())
