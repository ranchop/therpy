import numpy as np
import os

def getFileList(folder = 'Not Provided', fmt = '.', outp = False):
    # Confirm that given folder path is correct
    if not os.path.exists(folder): raise ValueError("Folder '{}' doesn't exist".format(folder))
    # Verify format
    if type(fmt) is not str: raise ValueError("Format (fmt) must be a string")
    if fmt[0] != '.': fmt = '.' + fmt
    def useFile(filename):
        if fmt == '.': return os.path.splitext(filename)[1] != ''
        return os.path.splitext(filename)[1] == fmt
    # Folder contents
    filenames = [filename for filename in os.listdir(folder) if useFile(filename)]
    # Output
    if outp:
        names = [os.path.splitext(f)[0] for f in filenames]
        paths = [os.path.join(folder,f) for f in filenames]
        return (names, paths)
    return filenames

def getpath(*args):
    basepath = os.path.expanduser('~')
    rootpath = os.path.join(basepath, 'Documents', 'My Programs')
    # Process input
    if len(args) == 0: return rootpath
    elif len(args) == 1 and type(args[0]) is str: relative = args       # Single string
    elif len(args) == 1 and type(args[0]) is list: relative = args[0]   # A single list
    elif len(args) >= 2 and type(args[1]) is str: relative = args       # Bunch of strings
    else:
        print("Illegal type of relative path received at therpy.getpath")
        return os.path.join(rootpath,'Junk')
    # Extend and return
    return os.path.join(rootpath,*relative)



if __name__ == '__main__':
    folder = '/Users/ranchopc/Documents/My Programs/Python Library/therpy'
    print(getFileList(folder,'.py'))
