# image reading and writing

## Housekeeping
import numpy as np
import os
import datetime
from datetime import timedelta
import re
# import pyfits
import astropy.io.fits as pyfits
import matplotlib.pyplot as pp
from shutil import copyfile
import warnings


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
def backuploc():
    # Get user home directory
    basepath = os.path.expanduser('~')
    # override the basepath if you are using external drive with your laptop
    # basepath = '/Volumes/Brain'
    # Find out the os
    from sys import platform as _platform
    # Platform dependent storage
    if _platform == 'darwin':
        # Mac OS X
        backuppath = os.path.join(basepath, 'Documents', 'My Programs', 'Raw Imagedata Temporary')
    elif _platform == 'win32' or _platform == 'cygwin':
        # Windows
        backuppath = os.path.join(basepath, 'Documents', 'My Programs', 'Raw Imagedata Temporary')
    else:
        # Unknown platform
        return None
    # If the folder doesn't exist, create it
    if not (os.path.exists(backuppath)): os.makedirs(backuppath)
    return backuppath


# If the extension is not provided, add the default of .fits
def fixentension(filename):
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
def imagename2subfolder(imagename=None):
    # Special case if imagename is not provided
    if imagename is None: return 'None'
    # Default values for pattern (future version: include as an optional input)
    # Version 1 (the default)
    re_pattern = '\d\d-\d\d-\d\d\d\d_\d\d_\d\d_\d\d'
    datetime_format = '%m-%d-%Y_%H_%M_%S'
    # Version 2 (sometimes '01' -> ' 1')
    re_pattern_2 = '\d\d-\d\d-\d\d\d\d_ \d_\d\d_\d\d'
    datetime_format_2 = '%m-%d-%Y_ %H_%M_%S'
    # Find '/' in the string and remove it -- to be done
    # Extract datetime
    imagetimestr = re.findall(re_pattern, imagename)
    imagetimestr2 = re.findall(re_pattern_2, imagename)
    if len(imagetimestr) == 1:
        imagetimestr = imagetimestr[0]
    elif len(imagetimestr2) == 1:
        imagetimestr = imagetimestr2[0]
        datetime_format =  datetime_format_2
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
    subfolder = os.path.join(imageyear, imagemonth, imagedate)
    return subfolder


# string for subfolder = imagename2subfolder( string for image name )
def imagename2subfolder_yesterday(imagename=None):
    # Special case if imagename is not provided
    if imagename is None: return 'None'
    # Default values for pattern (future version: include as an optional input)
    # Version 1 (the default)
    re_pattern = '\d\d-\d\d-\d\d\d\d_\d\d_\d\d_\d\d'
    datetime_format = '%m-%d-%Y_%H_%M_%S'
    # Version 2 (sometimes '01' -> ' 1')
    re_pattern_2 = '\d\d-\d\d-\d\d\d\d_ \d_\d\d_\d\d'
    datetime_format_2 = '%m-%d-%Y_ %H_%M_%S'
    # Find '/' in the string and remove it -- to be done
    # Extract datetime
    imagetimestr = re.findall(re_pattern, imagename)
    imagetimestr2 = re.findall(re_pattern_2, imagename)
    if len(imagetimestr) == 1:
        imagetimestr = imagetimestr[0]
    elif len(imagetimestr2) == 1:
        imagetimestr = imagetimestr2[0]
        datetime_format =  datetime_format_2
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
    subfolder = os.path.join(imageyear, imagemonth, imagedate)
    return subfolder


# imagedata = imagename2imagepath(imagename)
def imagename2imagepath(imagename, redownload=False):
    # Extract the subfolder path
    subpath = imagename2subfolder(imagename)
    # Fix the extension
    imagename = fixentension(imagename)
    # Check if it exists on temporary location
    imagepath_backup = os.path.join(backuploc(), subpath, imagename)
    if os.path.exists(imagepath_backup) and not redownload:
        return imagepath_backup
    # Find the base path depending on the platform
    from sys import platform as _platform
    if _platform == 'darwin':
        # Mac OS X
        basepath = '/Volumes/Raw Data/Images'
    elif _platform == 'win32' or _platform == 'cygwin':
        # Windows
        basepath = '\\\\18.25.16.34\\Raw Data\\Images'
    else:
        # Unknown platform
        basepath = None
    # Check if server is connected
    if os.path.exists(basepath) is False:
        raise FileNotFoundError('Server NOT connected at {}! and file was not found at {}'.format(basepath, imagepath_backup))
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
