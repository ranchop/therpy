import sqlite3
from . import metadata_constants


class SqliteComm:
    def __init__(self):
        self.mcst = metadata_constants.mcst()

    def queryWriter(self):
        pass

    def queryReader(self):
        pass

    def begin(self):
        self._db = sqlite3.connect(self.mcst.dbPath)
        self._db.row_factory = sqlite3.Row

    def end(self):
        self._db.close()

    def write(self,query=None):
        self.begin()
        self._db.execute(query)
        self._db.commit()
        self.end()

    def read(self,query):
        self.begin()
        cursor = self._db.execute(query)
        self.end()
        return cursor

