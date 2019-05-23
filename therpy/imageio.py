# image reading and writing

## Housekeeping
import numpy as np
import os
import datetime
from datetime import timedelta
import re
import astropy.io.fits as pyfits
import matplotlib.pyplot as pp
from shutil import copyfile
import warnings
import paramiko
import json
from sys import platform as _platform


## High Level Functions
def imagename2od(imagename):
    imagepath = imagename2imagepath(imagename)
    imagedata_raw = imagepath2imagedataraw(imagepath)
    imagedata_all = imagedataraw2imagedataall(imagedata_raw)
    imagedata_od = imagedataall2od(imagedata_all)
    return imagedata_od


def imagename2rawdata(imagename):
    imagepath = imagename2imagepath(imagename)
    imagedata_raw = imagepath2imagedataraw(imagepath)
    return imagedata_raw


def imagename2alldata(imagename):
    imagepath = imagename2imagepath(imagename)
    imagedata_raw = imagepath2imagedataraw(imagepath)
    imagedata_all = imagedataraw2imagedataall(imagedata_raw)
    return imagedata_all


## Low level functions
# get path to store downloaded images
def backuploc(lab='bec1'):
    # Get user home directory
    basepath = os.path.expanduser('~')
    # Find out the os
    from sys import platform as _platform
    # Platform dependent storage
    if _platform == 'darwin':
        # Mac OS X
        backuppath = os.path.join(basepath, 'Documents', 'My Programs', 'Raw Imagedata Temporary',lab)
    elif _platform == 'win32' or _platform == 'cygwin':
        # Windows
        backuppath = os.path.join(basepath, 'Documents', 'My Programs', 'Raw Imagedata Temporary',lab)
    else:
        # Unknown platform
        return None
    # If the folder doesn't exist, create it
    if not (os.path.exists(backuppath)): os.makedirs(backuppath)
    return backuppath


# If the extension is not provided, add the default of .fits
def fixextension(filename):
    # Add the .fits extension if not present
    imageformat = os.path.splitext(filename)[1]
    if imageformat == '': filename += '.fits'
    return filename


# Copy image to local backup location
def backupimage(imagepath_original, imagepath_backup):
    backuppath = os.path.split(imagepath_backup)[0]
    if not (os.path.exists(backuppath)): os.makedirs(backuppath)
    copyfile(imagepath_original, imagepath_backup)


# string for subfolder = imagename2subfolder( string for image name )
def imagename2subfolder(imagename=None, sftp=False):
    # Special case if imagename is not provided
    if imagename is None: return 'None'
    # Default values for pattern (future version: include as an optional input)
    # Version 1 (the default)
    re_pattern = '\d\d-\d\d-\d\d\d\d_\d\d_\d\d_\d\d'
    datetime_format = '%m-%d-%Y_%H_%M_%S'
    # Version 2 (sometimes '01' -> ' 1')
    re_pattern_2 = '\d\d-\d\d-\d\d\d\d_ \d_\d\d_\d\d'
    datetime_format_2 = '%m-%d-%Y_ %H_%M_%S'
    # Version 3 (Fermi3 Guppy format)
    re_pattern_3 = '\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d'
    datetime_format_3 = '%Y-%m-%d_%H-%M-%S'
    # Find '/' in the string and remove it -- to be done
    # Extract datetime
    imagetimestr = re.findall(re_pattern, imagename)
    imagetimestr2 = re.findall(re_pattern_2, imagename)
    imagetimestr3 = re.findall(re_pattern_3, imagename)
    if len(imagetimestr) == 1:
        imagetimestr = imagetimestr[0]
    elif len(imagetimestr2) == 1:
        imagetimestr = imagetimestr2[0]
        datetime_format =  datetime_format_2
    elif len(imagetimestr3) == 1:
        imagetimestr = imagetimestr3[0]
        datetime_format =  datetime_format_3
    else:
        return 'None'
    try:
        imagetime = datetime.datetime.strptime(imagetimestr, datetime_format)
    except ValueError as err:
        return err
    # Create subfolder
    imageyear = imagetime.strftime('%Y')
    imagemonth = imagetime.strftime('%Y-%m')
    imagedate = imagetime.strftime('%Y-%m-%d')
    if sftp:
        subfolder = imageyear+'/'+imagemonth+'/'+imagedate
    else:
        subfolder = os.path.join(imageyear, imagemonth, imagedate)

    return subfolder


# string for subfolder = imagename2subfolder( string for image name )
def imagename2subfolder_yesterday(imagename=None, sftp=False):
    # Special case if imagename is not provided
    if imagename is None: return 'None'
    # Default values for pattern (future version: include as an optional input)
    # Version 1 (the default)
    re_pattern = '\d\d-\d\d-\d\d\d\d_\d\d_\d\d_\d\d'
    datetime_format = '%m-%d-%Y_%H_%M_%S'
    # Version 2 (sometimes '01' -> ' 1')
    re_pattern_2 = '\d\d-\d\d-\d\d\d\d_ \d_\d\d_\d\d'
    datetime_format_2 = '%m-%d-%Y_ %H_%M_%S'
    # Version 3 (Fermi3 Guppy format)
    re_pattern_3 = '\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d'
    datetime_format_3 = '%Y-%m-%d_%H-%M-%S'
    # Find '/' in the string and remove it -- to be done
    # Extract datetime
    imagetimestr = re.findall(re_pattern, imagename)
    imagetimestr2 = re.findall(re_pattern_2, imagename)
    imagetimestr3 = re.findall(re_pattern_3, imagename)
    if len(imagetimestr) == 1:
        imagetimestr = imagetimestr[0]
    elif len(imagetimestr2) == 1:
        imagetimestr = imagetimestr2[0]
        datetime_format =  datetime_format_2
    elif len(imagetimestr3) == 1:
        imagetimestr = imagetimestr3[0]
        datetime_format =  datetime_format_3
    else:
        return 'None'
    try:
        imagetime = datetime.datetime.strptime(imagetimestr, datetime_format)
    except ValueError as err:
        return err
    # Create subfolder for yesterday
    imagetime = imagetime - timedelta(days=1)
    imageyear = imagetime.strftime('%Y')
    imagemonth = imagetime.strftime('%Y-%m')
    imagedate = imagetime.strftime('%Y-%m-%d')
    if sftp:
        subfolder = imageyear+'/'+imagemonth+'/'+imagedate
    else:
        subfolder = os.path.join(imageyear, imagemonth, imagedate)
    return subfolder


# imagedata = imagename2imagepath(imagename)
def imagename2imagepath(imagename, lab='bec1', redownload=False, use_sftp=False):
    # Extract the subfolder path
    subpath = imagename2subfolder(imagename)
    subpath_yesterday = imagename2subfolder_yesterday(imagename)
    # Fix the extension
    imagename = fixextension(imagename)
    # Check if it exists on temporary location
    imagepath_backup = os.path.join(backuploc(lab), subpath, imagename)
    imagepath_backup_yesterday = os.path.join(backuploc(lab), subpath_yesterday, imagename)

    if os.path.exists(imagepath_backup) and not redownload:
        return imagepath_backup

    if os.path.exists(imagepath_backup_yesterday ) and not redownload:
        return imagepath_backup_yesterday 

    # Find the base path depending on the platform
    if _platform == 'darwin':
        # Mac OS X
        if lab=='bec1':
            basepath = '/Volumes/Raw Data/Images'
        elif lab=='fermi3':
            basepath = '/Volumes/Raw Data/Fermi3/Images'
    elif _platform == 'win32' or _platform == 'cygwin':
        # Windows
        if lab=='bec1':
            basepath = '\\\\18.62.1.253\\Raw Data\\Images'
        elif lab=='fermi3':
            basepath = '\\\\18.62.1.253\\Raw Data\\Fermi3\\Images'
    else:
        # Unknown platform
        basepath = None
    # Check if server is connected
    if os.path.exists(basepath):
        # Find the fullpath to the image
        imagepath = os.path.join(basepath, subpath, imagename)
        # Check if file exists
        if os.path.exists(imagepath) is False:
            imagepath_today = imagepath
            subpath = imagename2subfolder_yesterday(imagename)
            imagepath = os.path.join(basepath, subpath, imagename)
            if os.path.exists(imagepath) is False:
                raise FileNotFoundError(
                    'Image NOT present on the server: Possibly invalid filename or folder location? Not found at : {} and {}'.format(
                        imagepath_today, imagepath))
        # Copy file to backup location
        backupimage(imagepath, imagepath_backup)

    if use_sftp:# use SFTP to transfer the file
        print('Server NOT connected!')
        # try to get creds
        config_path = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'config.json')

        if os.path.exists(config_path) is False:
            raise FileNotFoundError('Please place a config.json file in your Documents/My Programs directory to use SFTP transfer. ')
        else:
            try:
                with open(config_path) as file:
                    config = json.load(file)
            except:
                raise IOError('Problem reading config file')
            
        try:
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('18.62.1.253', username=config.get('user'), password=config.get('pass'))
            sftp = ssh.open_sftp()
        except:
            raise IOError('Connection refused')

        if lab=='bec1':
            basepath = 'Raw Data/Images'
        elif lab=='fermi3':
            basepath = 'Raw Data/Fermi3/Images'

        try: # check if image taken today
            imagepath_today = basepath+'/'+ subpath+'/'+ imagename
            sftp.stat(imagepath_today)
            imagepath = imagepath_today
        except:
            try: # check if image taken yesterday
                imagepath_yesterday = basepath+'/'+ subpath+'/'+ imagename
                sftp.stat(imagepath_yesterday)
                imagepath = imagepath_yesterday
            except: #raise error if no image
                raise FileNotFoundError(
                    'Image NOT present on the server: Possibly invalid filename or folder location? Not found at : {} and {}'.format(
                        imagepath_today, imagepath))

        if not os.path.exists(os.path.join(backuploc(lab), subpath)):
            os.makedirs(os.path.join(backuploc(lab), subpath))
        sftp.get(imagepath,imagepath_backup)
        sftp.close()

    # Return the backup path
    return imagepath_backup


# imagedata_raw = imagepath2imagedataraw(imagepath)
def imagepath2imagedataraw_fits(imagepath):
    # Check that imagepath is a valid path
    if os.path.exists(imagepath) is False:
        print('Invalid Filepath')
        return []
    # Load fits file
    try:
        imagedata_raw = pyfits.open(imagepath)
        imagedata_raw = imagedata_raw[0].data
    except TypeError as e:
        imagename = os.path.split(imagepath)[1]
        print("Error -- File may not be downloaded property -- {}".format(e))
        print("Trying to re-download file {}".format(imagename))
        imagename2imagepath(imagename, redownload=True)
        imagedata_raw = pyfits.open(imagepath)
        imagedata_raw = imagedata_raw[0].data

    imagedata_raw = np.float64(imagedata_raw)
    return imagedata_raw


# imagedata_raw = imagepath2imagedataraw(imagepath)
def imagepath2imagedataraw(imagepath):
    # Find the image format
    imageformat = os.path.splitext(imagepath)[1]
    if imageformat == '.fits':
        return imagepath2imagedataraw_fits(imagepath)
    else:
        print('Unknown file type')
        return None


def imagedataraw2imagedataall(imagedata_raw):
    # Make a copy of the input
    imagedata_all = imagedata_raw.copy()
    # Figure out image type and subtract background
    if imagedata_all.shape[0] == 3:
        imagedata_all[0] -= imagedata_all[2]
        imagedata_all[1] -= imagedata_all[2]
    elif imagedata_all.shape[0] == 4 and np.sum(imagedata_all[3]) == 0:
        imagedata_all[0] -= imagedata_all[2]
        imagedata_all[1] -= imagedata_all[2]
    elif imagedata_all.shape[0] == 4 and np.sum(imagedata_all[3]) != 0:
        imagedata_all[0] -= imagedata_all[2]
        imagedata_all[1] -= imagedata_all[3]
    else:
        raise NotImplementedError('Not a valid absorbance image: doesn\'t have 3 or 4 images')
    imagedata_all = imagedata_all[0:2]
    return imagedata_all


# imagedata_od = imagedataraw2od(imagedata_raw)
def imagedataall2od(imagedata_all):
    # Calculate OD
    with np.errstate(divide='ignore', invalid='ignore'):
        imagedata_od = np.log(imagedata_all[1] / imagedata_all[0])
    # Remove nan and inf with average of the surrounding -- to be done
    return imagedata_od


# TESTS
def tests():
    name = '05-17-2017_20_19_59_TopA'
    path = imagename2imagepath(name)
    data = imagepath2imagedataraw(path)
    print('success')

if __name__ == '__main__':
    tests()
