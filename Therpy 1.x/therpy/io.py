import time

# Importing from therpy is its main else using .
if __name__ == '__main__':
    import os.path
    import sys
    path2therpy = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library')
    sys.path.append(path2therpy)
    from therpy import roots1
else:
    from . import roots1


current_time_str = time.strftime("%m-%d-%Y_%H_%M_%S_dictioOutput.txt")
default_dictio_filepath = roots1.getpath('Default Storage','dictio',current_time_str)

class dictio:
    def __init__(self,*args,**kwargs):
        # Get data from the inputs
        if len(args) is 1 and type(args[0]) is dict: self.data = args[0]
        elif len(args) is 1 and type(args[0]) is str: self.data = self.fromfile(filepath=args[0])
        elif "filepath" in kwargs.keys(): self.data = self.fromfile(filepath=kwargs['filepath'])
        else: self.data = kwargs

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __iter__(self):
        return iter(self.data.keys())

    def tofile(self,filepath=None):
        if filepath is None: filepath = default_dictio_filepath
        fid = open(filepath,'w')
        for key in self.data.keys():
            val = self.data[key]
            typestr = str(type(val))
            typestr = typestr[typestr.find("'")+1:]
            typestr = typestr[:typestr.find("'")]
            fid.write("{}\t{}\t{}\n".format(key, val, typestr))
        fid.close()
        return filepath

    def fromfile(self,filepath=None):
        data = dict()
        if filepath is None: return data
        fid = open(filepath,'r')
        for line in fid:
            key = line[:line.find("\t")]
            line = line[line.find("\t")+1:]
            val = line[:line.find("\t")]
            line = line[line.find("\t")+1:]
            typestr = line[:line.find("\n")]
            if typestr == 'str': val = str(val)
            elif typestr == 'int': val = int(val)
            elif typestr == 'float': val = float(val)
            data[key] = val
        return data

    def get(self,key,default=None):
        if key in self.data.keys():
            return self.data[key]
        else:
            return default