# Biswaroop Mukherjee's bec1db v1.2 

import os
import re
import time
import shutil
import datetime
import pandas as pd
import datetime
from sys import platform as _platform
import sqlite3
import urllib
import numpy as np
import warnings

# Internet checker
def internet_on():
    try:
        urllib.request.urlopen('http://17.172.224.47', timeout=1)
        return True
    except:
        print('Connect to the internet to get the latest database!')
        return False


###-------------------
###  SQLITE functions
###-------------------
class Zeus:
    def __init__(self, databasepath):
        self.databasepath = databasepath


    ## Put a dictionary array into the database
    def insert_dictionary(self, dict_array):

        # Connect to the DB
        conn = sqlite3.connect(self.databasepath)
        cur = conn.cursor()
        images_added = 0

        # Iterate over the dictionary array
        for dict in dict_array:

            # Add the necessary columns. Never delete columns!
            cur.execute('SELECT * FROM data')
            db_columns = [description[0].lower() for description in cur.description]
            dict_columns = [column.lower() for column in list(dict.keys())]
            columns_needed = set(dict_columns)-set(db_columns) # Find the missing keys (columns)
            for column in list(columns_needed):
                sql = 'ALTER TABLE data ADD "{col_name}" FLOAT'
                cur.execute(sql.format(col_name=column)) # Add all the missing keys


            # Save the data
            sql = 'INSERT INTO data ({fields}) VALUES ({values})'
            fields = ", ".join(dict.keys())
            values = ', '.join(['"{0}"'.format(value) for value in dict.values()])
            composed_sql = sql.format(fields=fields, values=values)
            try:
                cur.execute(composed_sql)
            except sqlite3.IntegrityError:
                print('Attemped to add a duplicate')
                images_added -= 1
                pass

            # Count images added
            images_added += 1

        print(str(images_added) +' images added to database')
        conn.commit()
        conn.close()

    ## Query the database
    def data_query(self, sql_query):
        conn = sqlite3.connect(self.databasepath)
        c = conn.cursor()
        c.execute(sql_query)
        result = c.fetchall()
        conn.close()
        return result

    ## Get all the columns
    def columns_query(self):
        conn = sqlite3.connect(self.databasepath)
        cur = conn.cursor()
        cur.execute('SELECT * FROM data LIMIT 1')
        db_columns = [description[0].lower() for description in cur.description]
        conn.close()
        params = [re.sub('exp_','',param) for param in db_columns]
        return params




    ## Obliterate and re-make the data table
    def data_obliterate(self):
        conn = sqlite3.connect(self.databasepath)
        c = conn.cursor()
        c.execute('''DROP TABLE IF EXISTS data''')
        c.execute('''CREATE TABLE data
                      (snippet_time text UNIQUE, Date text, Time text,
                      year INT, month INT, day INT,
                      hour INT, minute INT, second INT, unixtime INT)''')
        conn.commit()
        conn.close()


## Make paths for the databases and other stuff
def pathmake(main_folder, sub_folder):

    if _platform == 'darwin':
        # Mac OS X
        madepath = '/Volumes/'+main_folder+'/'+sub_folder
    elif _platform == 'win32' or _platform == 'cygwin':
        # Windows
        madepath = '\\\\18.62.1.253\\'+main_folder+'\\'+sub_folder
    else:
        # Unknown platform
        madepath = None

    ## Check if server is connected
    if os.path.exists(madepath) is False:
        raise FileNotFoundError("Couldn't connect to {}. Check if server is connected!".format(madepath))

    return madepath


### Useful for datetime
def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds()

def timestr_to_datetime(time_string):
    time_string = re.sub(' ','0',time_string)
    return datetime.datetime.strptime(time_string,'%m-%d-%Y_%H_%M_%S')

###-----------------------------------
### Functions to read snippet files
###-----------------------------------

## Make a dictionary from an arbitrary line in the file
def generate_dictionary(line):

    # Split line into name and params, and clean
    current_line = re.split('\t',line)  # Split line into image name and params
    current_line = [re.sub('\n','',line) for line in current_line] # Remove newlines

    # Get snippet_time, date and time
    snippet_time = current_line[0]
    [img_mm, img_dd, img_yy, img_hh, img_mi, img_ss] = re.split('-|_',snippet_time)
    img_date = img_yy+'-'+img_mm+'-'+img_dd
    img_time = img_hh+':'+img_mi+':'+img_ss
    extra_params = ['snippet_time',snippet_time] + ['Date', img_date] + ['Time', img_time] +  ['year', int(img_yy)] + ['month', int(img_mm)] + ['day', int(img_dd)] + ['hour', int(img_hh)] + ['minute', int(img_mi)] + ['second', int(img_ss)] + ['unixtime',unix_time(timestr_to_datetime(snippet_time))]

    # Set up params as a list
    paramline = re.sub(' |=|\+','',current_line[1])
    paramline = re.split(';|,',paramline)[0:-1] # Split by ; or ,
    paramline[::2] = ['exp_'+re.sub('\.','',param) for param in paramline[::2]] # remove . from parameter name only
    paramline[1::2] = [float(param) for param in paramline[1::2]] # turn experimental parameters into floats
    params =  extra_params + paramline
    params[::2] = [str.strip(param).lower() for param in params[::2]] # Clean params

    # Turn it into a dictionary and return
    return dict(zip(params[0::2], params[1::2]))

# Read the clipboard
def clipboard():
    # A helper method to copy image names from the clipboard
    df = pd.read_clipboard(names=['name'])
    images = df['name'].tolist()
    return images


## Convert files to dictionaries
def read_snippet_file(filename):

    if type(filename)!=str:
        print('Please enter a string for the filename!')
        return

    # Open file
    try:
        fo = open(filename,'r')
    except FileNotFoundError:
        print("Can't find that file!")

    # Read all lines
    lines = fo.readlines()
    database_dict = []

    # Parse into dictionary
    for k in range(len(lines)):
        current_dict = generate_dictionary(lines[k])
        database_dict = database_dict+[current_dict]

    # Close opened file
    fo.close()

    return database_dict

## Read a single snippet line
def read_snippet_line(filename, line_to_read):

    if type(filename)!=str:
        print('Please enter a string for the filename!')
        return

    # Open file
    try:
        fo = open(filename,'r')
    except FileNotFoundError:
        print("Can't find that file!")

    # Read the line
    for i, line in enumerate(fo):
        if i==line_to_read:
            line_dict = generate_dictionary(line)

    # Close opened file
    fo.close()

    return line_dict

## Define a local location to store the database
def localloc(lab='bec1'):
    # Get user home directory
    basepath = os.path.expanduser('~')
    # Find out the os
    from sys import platform as _platform
    # Platform dependent storage
    if _platform == 'darwin':
        # Mac OS X
        localpath = os.path.join(basepath, 'Documents', 'My Programs', 'Database', lab)
    elif _platform == 'win32' or _platform == 'cygwin':
        # Windows
        localpath = os.path.join(basepath, 'Documents', 'My Programs', 'Database', lab)
    else:
        # Unknown platform
        return None
    # If the folder doesn't exist, create it
    if not (os.path.exists(localpath)): os.makedirs(localpath)
    return localpath


# Clean the parameters
def clean_params(params):
    params = [re.sub(' |=|\+|\.','',param) for param in params]
    extra_params = ['snippet_time', 'Date', 'Time', 'year','month', 'day', 'hour', 'minute','second', 'unixtime']
    paramsout = ['exp_'+ param for param in params if param not in extra_params]+[param for param in params if param in extra_params]
    paramsout = [str.strip(param).lower() for param in paramsout]
    return paramsout

# copy the database using sftp if needed
def copy_db(dbpath, localpath, lab='bec1', password=''):
    if os.path.exists(dbpath):
        shutil.copy(dbpath, localpath)
    else:
        if password=='':
            raise ValueError('Please enter the server password. Hint Martin.')
        else:
            try:
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.connect('18.62.1.253', username='bec1admin', password=password)
                sftp = ssh.open_sftp()
            except:
                raise IOError('Connection refused')
                if lab=='bec1':
                    basepath = 'Processed Data/Database'
                elif lab=='fermi3':
                    basepath = 'Processed Data/Fermi 3/Database'
            sftp.get(basepath,localpath)
            sftp.close()



# The database api class
class Tullia:
    def __init__(self, delta=10, connect=True, lab='bec1', password=''):
        self.localdbpath = os.path.join(localloc(),'Zeus.db')
        self.delta=delta
        self.connect=connect
        self.lab=lab
        self.password=password
        try:
            self.refresh()
        except:
            print('The servers are not connected!')
            if os.path.exists(self.localdbpath):
                print('Using the local database copy. Could be outdated!')
            else:
                print()
                raise FileNotFoundError("Couldn't find the local database! Connect to the internet and servers to get a copy.")


    def refresh(self):
        ## Get the original database directory and version
        databasepath = pathmake(main_folder='Processed Data', sub_folder='Database')
        self.databasepath = os.path.join(databasepath,'Zeus.db')
        remote_version = os.stat(self.databasepath).st_mtime
        self.localdbpath = os.path.join(localloc(),'Zeus.db')
        ## Download the database
        internet = internet_on()
        if not internet and not os.path.exists(self.localdbpath):
            print('You need to connect to the internet to download a db')
        elif internet and not os.path.exists(self.localdbpath):
            # Copy if no version exists locally
            print('Downloading the database...')
            copy_db(self.databasepath, localloc(self.lab), lab=self.lab, password=self.password)
            os.utime(self.localdbpath,(remote_version, remote_version))
            print('Done')
        elif internet and os.path.exists(self.localdbpath):
            # Update db
            local_version = os.stat(self.localdbpath).st_mtime
            if local_version == remote_version:
                return
            elif self.connect==False:
                print('You have chosen to not update your database.')
            else:
                print('Updating the local database...')
                copy_db(self.databasepath, localloc(self.lab), lab=self.lab, password=self.password)
                os.utime(self.localdbpath,(remote_version, remote_version))
                print('Done')
                return


    def image_query(self,imagesin,paramsoriginals='*'):

        ## For string imagename and string parameter
        if type(imagesin)==str and type(paramsoriginals)==str and not paramsoriginals=='*':
            imagename=imagesin
            unixtime_0 = unix_time(timestr_to_datetime(imagename[0:19]))
            unixtimes = str(int(unixtime_0 -self.delta)) + ' AND ' + str(int(unixtime_0 +self.delta))
            sql = 'SELECT * from data where unixtime between {unixtime_range}'
            sql_query = sql.format(unixtime_range=unixtimes)
            zeus = Zeus(self.localdbpath)
            dbcolumns = zeus.columns_query()
            results = zeus.data_query(sql_query)
            results_df = pd.DataFrame(results, columns=dbcolumns)
            param = re.sub(' |=|\+|\.','',paramsoriginals)
            param = str.strip(param).lower()
            return results_df[param].tolist()[0]

        ## for string imagename but all columns are wanted
        elif type(imagesin)==str and paramsoriginals=='*':
            imagename=imagesin
            unixtime_0 = unix_time(timestr_to_datetime(imagename[0:19]))
            unixtimes = str(int(unixtime_0 -self.delta)) + ' AND ' + str(int(unixtime_0 +self.delta))
            sql = 'SELECT * from data where unixtime between {unixtime_range}'
            sql_query = sql.format(unixtime_range=unixtimes)
            zeus = Zeus(self.localdbpath)
            dbcolumns = zeus.columns_query()
            results = zeus.data_query(sql_query)
            results_df = pd.DataFrame(results, columns=dbcolumns)
            return results_df



        ## return all columns for multiple images
        elif paramsoriginals=='*':
            # Clean the image names and parameter names
            images = [image[0:19] for image in imagesin]
            paramsin =  ['unixtime' ] # need the unixtimes to find a match
            params = clean_params(paramsin)

            zeus = Zeus(self.localdbpath)

            # Convert the time strings to datetimes
            try:
                image_times = [timestr_to_datetime(image) for image in images]
            except ValueError:
                print('Some of the image names are not valid')
                raise
                return None


            # Make a new dataframe
            df = pd.DataFrame(columns=['imagename']+paramsin)

            # Generate the SQL query
            sql = '''SELECT * FROM data WHERE unixtime between {unixtime_range}'''

            timeranges = []
            for image_time in image_times:
                unixtime_0 = unix_time(image_time)
                unixtimes = str(int(unixtime_0 -self.delta)) + ' AND ' + str(int(unixtime_0 +self.delta))
                timeranges = timeranges + [unixtimes]

            times = ' OR unixtime between '.join(timeranges)
            sql_query = sql.format(unixtime_range=times)

            # Query the DB
            cols = zeus.columns_query()
            results = zeus.data_query(sql_query)
            results_df = pd.DataFrame(results,columns=cols)

            # Locate the requested images in the query results
            results_times = np.array(results_df['unixtime'].tolist())
            for image_time in image_times:
                unixtime_0 = unix_time(image_time)
                matchloc = np.argmin(np.abs(results_times-unixtime_0))
                df_to_append = results_df[matchloc:matchloc+1]
                df = df.append(pd.DataFrame(df_to_append,columns=cols), ignore_index=True)

            df['imagename'] = imagesin


            return df


        ## Otherwise, return all
        else:
            # Clean the image names and parameter names
            images = [image[0:19] for image in imagesin]
            if 'unixtime' in paramsoriginals:
                paramsin = paramsoriginals
            else:
                paramsin = paramsoriginals + ['unixtime' ] # need the unixtimes to find a match

            params = clean_params(paramsin)

            zeus = Zeus(self.localdbpath)

            # Convert the time strings to datetimes
            try:
                image_times = [timestr_to_datetime(image) for image in images]
            except ValueError:
                print('Some of the image names are not valid')
                raise
                return None


            # Make a new dataframe
            df = pd.DataFrame(columns=['imagename']+paramsin)

            # Generate the SQL query
            sql = '''SELECT {columns} FROM data WHERE unixtime between {unixtime_range}'''

            timeranges = []
            for image_time in image_times:
                unixtime_0 = unix_time(image_time)
                unixtimes = str(int(unixtime_0 -self.delta)) + ' AND ' + str(int(unixtime_0 +self.delta))
                timeranges = timeranges + [unixtimes]

            times = ' OR unixtime between '.join(timeranges)
            cols = ', '.join(params)
            sql_query = sql.format(columns=cols, unixtime_range=times)

            # Query the DB
            results = zeus.data_query(sql_query)
            results_df = pd.DataFrame(results,columns=paramsin)

            # Locate the requested images in the query results
            results_times = np.array(results_df['unixtime'].tolist())
            for image_time in image_times:
                unixtime_0 = unix_time(image_time)
                matchloc = np.argmin(np.abs(results_times-unixtime_0))
                df_to_append = results_df[matchloc:matchloc+1]
                df = df.append(pd.DataFrame(df_to_append,columns=paramsin), ignore_index=True)

            df['imagename'] = imagesin

            if not 'unixtime' in paramsoriginals:
                df = df.drop('unixtime', 1)

            return df
