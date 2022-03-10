import os

class InstrumentsMapper:

    def __init__(self):
        instruments = os.listdir(os.getcwd() + '/app/data/sf2/')

        loi = {}
        for i in instruments:
            loi[i] = i
        self.list_of_instruments = loi
        # self.list_of_instruments = {
        #     "CTK-230_SoundFont.sf2": "CTK-230_SoundFont.sf2",
        #     "rhodes.sf2": "rhodes.sf2",
        #     "Super_Mario_Advance_4.sf2": "Super_Mario_Advance_4.sf2",
        #     "Super_Nintendo_Unofficial_update.sf2": "Super_Nintendo_Unofficial_update.sf2"
        # }

    def get_instrument_file_name(self, key):
        return self.list_of_instruments[key]

    def get_all_instruments_names(self):
        return list(self.list_of_instruments.keys())
