import os

class mcst:
    def __init__(self):
        self._ServerBasePath()
        self._dbNames()

    @property
    def dbPath(self):
        return self.dbServerFileLocation

    @property
    def dbTableName(self):
        return 'test'

    def _dbNames(self):
        self.dbFileName = 'MetadataBEC1.db'
        self.dbServerFileLocation = os.path.join(self.ServerBasePath,'Processed Data','Database')

    def _ServerBasePath(self):
        # server base path
        from sys import platform as _platform
        if _platform == 'darwin':
            # Mac OS X
            self.ServerBasePath = '/Volumes'
        elif _platform == 'win32' or _platform == 'cygwin':
            # Windows
            self.ServerBasePath = '\\\\18.62.1.253'
        else:
            # Unknown platform
            self.ServerBasePath = None

if __name__ == '__main__':
    t = mcst()
    print(os.path.exists(t.dbServerFileLocation))
