# Importing from therpy is its main else using .
if __name__ == '__main__':
    import os.path
    import sys
    path2therpy = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library')
    sys.path.append(path2therpy)
    from therpy import io
else:
    from . import io



class qTextEditDictIO:
    '''
    Data format for settings
    qSett (dict)
        keys: name of the setting
        vals: list of QTextEdit
    sett (dict)
        keys: name of the setting
        vals: float value for the setting

    '''
    def __init__(self, qSett=None, func=None, filepath=None):
        self.qSett = qSett
        self.func = func
        self.filepath = filepath
        self.sett = self.setupValuesFloatFromFile(func,filepath)
        self.connectEditSignals()


    def getValuesFloat(self):
        for key in self.qSett.keys():
            self.sett[key] = float(self.qSett[key].text())

    def setupValuesFloatFromFile(self, func=None, filepath=None):
        self.sett = io.dictio(filepath=filepath)        # Get default values from file
        # Change the text to default value
        for key in self.qSett.keys():
            self.qSett[key].setText(str(self.sett.get(key,'nan')))

    def connectEditSignals(self):
        for key in self.qSett.keys():
            self.qSett[key].editingFinished.connect(self.getValuesFloat)

