# test git methods


## began in-depth review and documentation update, 07.17.26

## initial imports
# some key sources include:
#    Adam Gordon-Fennell, tidy lab tools
#    Photometry data preprocessing.ipynb, Simpson et al., 2024
#    Yizeng Liang and Zhang Zhimin, airPLS method (2010)


from re import search
from tdt import read_block
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import h5py
from pathlib import Path
from scipy.signal import medfilt, butter, iirnotch, filtfilt
from scipy.stats import linregress
from scipy.stats import sem
from scipy.optimize import curve_fit, minimize
from scipy.signal import resample_poly
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import median_filter
from scipy.ndimage import label as nd_label
from fractions import Fraction
from math import gcd
from scipy import stats
from glob import glob
import glob
import math
import re
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from itertools import groupby
from operator import itemgetter
from scipy.signal import decimate
from matplotlib.colors import LinearSegmentedColormap, to_rgb


plt.rcParams['figure.figsize'] = [10, 8] # Make default figure size larger.
plt.rcParams['axes.xmargin'] = 0          # Make default margin on x axis zero.
plt.rcParams['axes.labelsize'] = 12     #Set default axes label size 
plt.rcParams['axes.titlesize']=15
plt.rcParams['axes.titleweight']='heavy'
plt.rcParams['ytick.labelsize']= 10
plt.rcParams['xtick.labelsize']= 10
plt.rcParams['legend.fontsize']=12
plt.rcParams['legend.markerscale']=2




#updated doric extraction functions ---------------------------------------------------------------------------------------------------------

# new functions written 09.12.25 for the wireless photometry system
# updated documentation 07.17.26

'''
doric_fipho_data_extraction()
        |
        |
        +---- doric_extract_stream_data()
        |          |
        |          +-- extracts fluorescence traces
        |          +-- creates stream DataFrame
        |
        |
        +---- doric_extract_epoch_data()
                   |
                   +-- extracts TTL events
                   +-- creates epoch DataFrame

'''



import os
import h5py    # Doric data extraction requires the h5py library, to deal with the Hierarchica Data Format version 5 format of Doric files
import pandas as pd
import numpy as np
import re


def doric_extract_epoch_data(h5file, transition_threshold=0.5):
    """
    Extract onset times of digital IO events defined as falling edges (1->0) from Doric HDF5 file.

    Parameters:
        h5file : h5py.File object
            Opened .doric HDF5 file.
        transition_threshold : float, optional
            Threshold to binarize signal (default 0.5).

    Returns:
        pd.DataFrame with columns:
            - blockname: name of the signal series (e.g., 'Series001')
            - source_path: full HDF5 path of the dataset
            - input code: Username attribute from dataset or channel name fallback
            - onset: event onset time in milliseconds from DigitalIO/Time dataset
    """
    # begin by creating the empty list that will store epoch data. This will ultimately be a list of dictionaries
    # the list of dictionaries will become a pandas dataframe, with keys as column headers and dictionary content as row data
    all_onsets = []   
                     
        
    # use the .get method to extract signals in the h5 file, with None as the default if nothing is found
    # this doesn't directly load the data yet, but essentially creates a reference to the location of the signal data for later extraction
    # conceptually the h5 file is similar to nested dictionaries
    signals_root = h5file.get('DataAcquisition/NC500/Signals', None)  
    
    
    # if no signals are found, return empty dataframe. 
    #This is important for output consistency - the code always produces a dataframe with the same columns, even if empty. 
    if signals_root is None:
        print("❌ DataAcquisition/NC500/Signals group not found!")
        return pd.DataFrame(columns=['blockname', 'source_path', 'input code', 'onset'])

    # locate the digital IO data within each signal group 
    # this is essentially a subgroup of series data, the .get creates a reference to the various DIO series and the time vector
    for series_key, series_group in signals_root.items():
        digitalio_group = series_group.get('DigitalIO', None)
        if digitalio_group is None:
            continue  # continue with the larger loop if a DigitalIO group is not found within a series_group

        time_ds = digitalio_group.get('Time', None) # if a DigitalIO group is found, one series within is the time vector
        if time_ds is None:
            print(f"⚠️ Missing 'Time' dataset in {series_key}/DigitalIO")
            continue
            
        # time_ds is just the h5 data file object, a reference to the time vector but not the vector itself
        # using time_ds[()] loads the actual numerical array that is the content of the object
        io_time = time_ds[()]  # Time vector in milliseconds

        for dio_key, dio_ds in digitalio_group.items():
            if dio_key == 'Time':
                continue  # skip time dataset itself

            try:
                io_state = dio_ds[()]  
                # this is creating a numerical array consisting of a series of either 0 or 1, 
                # as the dio_ds refers to the content related to the dio_key
                
            except Exception as e:
                print(f"⚠️ Could not read dataset {dio_key}: {e}")
                continue

            if io_state.shape != io_time.shape:
                print(f"⚠️ Shape mismatch between time and signal data in {series_key}/DigitalIO/{dio_key}")
                continue

            # Convert signal to binary using threshold
            io_state_bin = (io_state > transition_threshold).astype(int)

            # Find falling edges: transitions from 1 to 0
            # uses numpy .diff to find changes between consecutive samples
            # uses .where to identify the indices where falling edges occur, within the parenthesis is a boolean array
            # add one since the diff array is one element shorter than the original array, corrects the index of the edge
            idx_falling = np.where(np.diff(io_state_bin) == -1)[0] + 1 

            # Extract falling-edge times
            falling_times = io_time[idx_falling]

            # Extract metadata
            attrs = dio_ds.attrs
            username = attrs.get('Username', '') # extract the label of the DIO event, e.g. program start
            if isinstance(username, bytes):
                username = username.decode('utf-8')

            source_path = f"{series_key}/DigitalIO/{dio_key}"

            # now the all_onsets list is populated with 1 dictionary per event
            # this loop creates an event for every detected falling edge, within the large loop for each DIO
            for t in falling_times:
                all_onsets.append({
                    'blockname': series_key,
                    'source_path': source_path,
                    'input code': username or dio_key,
                    'onset': t
                })

    # now the list of dictionaries between formated as a pandas dataframe
    if all_onsets:
        return pd.DataFrame(all_onsets)[['blockname', 'source_path', 'input code', 'onset']]
    else:
        return pd.DataFrame(columns=['blockname', 'source_path', 'input code', 'onset'])

    
# use the dictionary method .get to convert names of channels

def doric_convert_channel_id(channel_id):
    mapping = {
        'Headstage01LockIn01': 'Headstage01_470nm',
        'Headstage01LockIn02': 'Headstage01_415nm',
        'Headstage02LockIn01': 'Headstage02_470nm',
        'Headstage02LockIn02': 'Headstage02_415nm',
        'Headstage03LockIn01': 'Headstage03_470nm',
        'Headstage03LockIn02': 'Headstage03_415nm',
        'Headstage04LockIn01': 'Headstage04_470nm',
        'Headstage04LockIn02': 'Headstage04_415nm'
    }
    return mapping.get(channel_id, channel_id) # .get method uses the original channel_id as a default if the mapping fails



def doric_extract_stream_data(h5file, blockname):
  
    stream_df_combined = pd.DataFrame()
    info_records = []

    # Read start/stop times from root attributes
    try:
        root_attrs = h5file['DataAcquisition']['NC500'].attrs
        device_start_time = root_attrs.get('DeviceStartTime', '')
        device_stop_time = root_attrs.get('DeviceStopTime', '')
    except Exception as e:
        print(f"⚠️ Could not read device start/stop times: {e}")
        device_start_time = ''
        device_stop_time = ''

    # Read headstage nicknames and sampling rates
    # begin by creating the empty dictionary
    headstage_meta = {}
    try:
        configs = h5file['Configurations']['NC500']['Antenna01']['Settings']
        # loop through the contents of configs (headstage01, headstage02)
        # this will create a nested dictionary with nickname and fs for each headstage
        for hs_key in configs:
            if hs_key.startswith('Headstage'):
                hs_group = configs[hs_key]
                nickname = hs_group.attrs.get('HeadstageNickname', '')
                if isinstance(nickname, bytes):
                    nickname = nickname.decode('utf-8')
                sampling_rate = hs_group.attrs.get('ADCDemodulatedSampling', np.nan)
                headstage_meta[hs_key] = {'nickname': nickname, 'sampling_rate': sampling_rate}
    except Exception as e:
        print(f"⚠️ Warning: Could not read headstage metadata: {e}")

    # Process streams
    # as in the epoch extraction function, .get method to access signals, series001 consisting of headstage01, 02 and DIOs
    signals_root = h5file.get('DataAcquisition/NC500/Signals', None) 
    if signals_root is None:
        print("❌ DataAcquisition/NC500/Signals group not found!")
        return pd.DataFrame(), pd.DataFrame()

    for series_key in signals_root:
        series_group = signals_root[series_key]
        
        # loop through Headstage01LockIn01, etc. 
        for stream_key in series_group:
            if 'LockIn' not in stream_key:
                continue

            stream_group = series_group[stream_key]

            # Extract headstage ID (e.g., Headstage01)
            # the re regular expressions (regex) module is used to extract the headstage identifier, pattern identification in text
            m = re.match(r'(Headstage\d+)', stream_key) # extract e.g. "Headstage01" from the stream key
            headstage_id = m.group(1) if m else None
            
            # series of .get to access the nested info within the metadata dictionary
            # first narrow to appropriate headstage, then get nickname and fs
            nickname = headstage_meta.get(headstage_id, {}).get('nickname', '') 
            sampling_rate = headstage_meta.get(headstage_id, {}).get('sampling_rate', np.nan)

            # Read datasets
            time_ds = stream_group.get('Time', None)
            signal_ds = stream_group.get(stream_key, None)

            if time_ds is None or signal_ds is None:
                print(f"⚠️ Missing data in {stream_key}, skipping.")
                continue

            # as in the epoch extraction, this step retrieves the actual array of data from the referenced locations within the group
            time = time_ds[()]
            raw_au = signal_ds[()]

            if len(time) != len(raw_au):
                print(f"⚠️ Mismatch in time and signal length for {stream_key}, skipping.")
                continue

            channel_raw = stream_key
            channel_name = doric_convert_channel_id(stream_key)

            # Build data rows
            # this step uses pandas to build a dataframe from a dictionary
            # in pandas, scalar values / single items are automatically broadcast to match the length of array-like columns
            # array columns must have the same length however
            df = pd.DataFrame({
                'blockname': blockname,
                'channel_raw': channel_raw,
                'channel_name': channel_name,
                'nickname': nickname,
                'time': time,
                'raw_au': raw_au
            })

            stream_df_combined = pd.concat([stream_df_combined, df], ignore_index=True)

            # Build info metadata row
            info_records.append({
                'blockname': blockname,
                'nickname': nickname,
                'channel_raw': channel_raw,
                'channel_name': channel_name,
                'fs': sampling_rate,
                'device_start_time': device_start_time,
                'device_stop_time': device_stop_time
            })

    streams_info_df = pd.DataFrame(info_records)
    return stream_df_combined, streams_info_df


# batch processing of a folder containing Doric files
# this pipeline manager function uses the above extraction functions

def doric_fipho_data_extraction_old(dir_raw, dir_extracted):

    # list comprehension to find files
    raw_files = [f for f in os.listdir(dir_raw) if f.endswith('.doric')]
    processed = set(os.path.splitext(f)[0].replace('_streams_data', '').replace('_epocs_data', '')
                    for f in os.listdir(dir_extracted) if f.endswith('.feather'))

    files_to_process = [f for f in raw_files if os.path.splitext(f)[0] not in processed]

    if not files_to_process:
        print(f"✅ All files in '{dir_raw}' have already been processed.")
        return

    print(f"\n📁 Extracting from: {dir_raw}")
    print(f"📤 Saving to: {dir_extracted}")

    for filename in files_to_process:
        blockname = os.path.splitext(filename)[0] # split to extract the name of the session / recording 
        full_path = os.path.join(dir_raw, filename)

        print(f"\n🔍 Processing {filename} ...")

        try:
            # use read mode to access the HDf5 file contents
            with h5py.File(full_path, 'r') as f:
                # Extract streams data and metadata
                try:
                    stream_df, streams_info_df = doric_extract_stream_data(f, blockname)
                except Exception as e:
                    print(f"⚠️ Failed to extract stream data from {filename}: {e}")
                    stream_df = pd.DataFrame()
                    streams_info_df = pd.DataFrame()

                # Extract epochs
                try:
                    epoc_df = doric_extract_epoch_data(f)
                    epoc_df["blockname"] = blockname
                except Exception as e:
                    print(f"⚠️ Failed to extract epoch data from {filename}: {e}")
                    epoc_df = pd.DataFrame()

                # Save data as a feather file, fast binary pandas format
                if not stream_df.empty:
                    stream_outfile = os.path.join(dir_extracted, f"{blockname}_streams_data.feather")
                    stream_df.reset_index(drop=True).to_feather(stream_outfile)
                    print(f"✅ Saved: {stream_outfile}")
                else:
                    print("⚠️ No stream data to save.")

                # Save metadata
                if not streams_info_df.empty:
                    info_outfile = os.path.join(dir_extracted, f"{blockname}_streams_info.feather")
                    streams_info_df.reset_index(drop=True).to_feather(info_outfile)
                    print(f"✅ Saved: {info_outfile}")
                else:
                    print("⚠️ No streams info to save.")

                # Save epoch data
                if not epoc_df.empty:
                    epoc_outfile = os.path.join(dir_extracted, f"{blockname}_epocs_data.feather")
                    epoc_df.reset_index(drop=True).to_feather(epoc_outfile)
                    print(f"✅ Saved: {epoc_outfile}")
                else:
                    print("⚠️ No epoch data to save.")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")



# batch processing of a folder containing Doric files
# this pipeline manager function uses the above extraction functions
# updated to create empty epoch files when no events occurred

def doric_fipho_data_extraction(dir_raw, dir_extracted):

    # list comprehension to find files
    raw_files = [f for f in os.listdir(dir_raw) if f.endswith('.doric')]
    processed = set(os.path.splitext(f)[0].replace('_streams_data', '').replace('_epocs_data', '')
                    for f in os.listdir(dir_extracted) if f.endswith('.feather'))

    files_to_process = [f for f in raw_files if os.path.splitext(f)[0] not in processed]

    if not files_to_process:
        print(f"✅ All files in '{dir_raw}' have already been processed.")
        return

    print(f"\n📁 Extracting from: {dir_raw}")
    print(f"📤 Saving to: {dir_extracted}")

    for filename in files_to_process:
        blockname = os.path.splitext(filename)[0] # split to extract the name of the session / recording 
        full_path = os.path.join(dir_raw, filename)

        print(f"\n🔍 Processing {filename} ...")

        try:
            # use read mode to access the HDf5 file contents
            with h5py.File(full_path, 'r') as f:
                # Extract streams data and metadata
                try:
                    stream_df, streams_info_df = doric_extract_stream_data(f, blockname)
                except Exception as e:
                    print(f"⚠️ Failed to extract stream data from {filename}: {e}")
                    stream_df = pd.DataFrame()
                    streams_info_df = pd.DataFrame()

                # Extract epochs
                epoc_failed = False
                
                try:
                    epoc_df = doric_extract_epoch_data(f)
                    epoc_df["blockname"] = blockname
                except Exception as e:
                    print(f"⚠️ Failed to extract epoch data from {filename}: {e}")
                    epoc_df = pd.DataFrame()
                    epoc_failed = True
                    
                    
                # Save data as a feather file, fast binary pandas format
                if not stream_df.empty:
                    stream_outfile = os.path.join(dir_extracted, f"{blockname}_streams_data.feather")
                    stream_df.reset_index(drop=True).to_feather(stream_outfile)
                    print(f"✅ Saved: {stream_outfile}")
                else:
                    print("⚠️ No stream data to save.")

                # Save metadata
                if not streams_info_df.empty:
                    info_outfile = os.path.join(dir_extracted, f"{blockname}_streams_info.feather")
                    streams_info_df.reset_index(drop=True).to_feather(info_outfile)
                    print(f"✅ Saved: {info_outfile}")
                else:
                    print("⚠️ No streams info to save.")

                # Save epoch data
                if not epoc_failed:
                    if epoc_df.empty:
                        epoc_df = pd.DataFrame({
                            'blockname': pd.Series(dtype=str),
                            'source_path': pd.Series(dtype=str),
                            'input code': pd.Series(dtype=str),
                            'onset': pd.Series(dtype=float),
                        })
                        print("ℹ️ No epoch events found; saving empty epoch file.")
 
                    epoc_outfile = os.path.join(dir_extracted, f"{blockname}_epocs_data.feather")
                    epoc_df.reset_index(drop=True).to_feather(epoc_outfile)
                    print(f"✅ Saved: {epoc_outfile}")
            
                else:
                    print("⚠️ Epoch extraction failed; no epoch file saved.")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")







# info -----------------------------------------------------------------------------------------------------------------

def tidy_tdt_info(data_tdt):
    # extracts info from tdt structure

    tidy_info = pd.DataFrame(
        {
            'blockname': [data_tdt.info.blockname],
            'tankpath': [data_tdt.info.tankpath],
            'start_date': [data_tdt.info.start_date],
            'utc_start_time': [data_tdt.info.utc_start_time],
            'stop_date': [data_tdt.info.stop_date],
            'utc_stop_time': [data_tdt.info.utc_stop_time],
            'duration': [data_tdt.info.duration],
            'stream_channel': [data_tdt.info.stream_channel],
            'snip_channel': [data_tdt.info.snip_channel]

        })
    
    return tidy_info


# streams --------------------------------------------------------------------------------------------------------------
def extract_stream_info(data_tdt_streams):
    # return info for an individual tdt stream

    tidy_streams_info = pd.DataFrame(
        {
            'name': data_tdt_streams.name,
            'code': data_tdt_streams.code,
            'size': [data_tdt_streams.size],
            'type': data_tdt_streams.type,
            'type_str': data_tdt_streams.type_str,
            'ucf': data_tdt_streams.ucf,
            'fs': data_tdt_streams.fs,
            'dform': data_tdt_streams.dform,
            'start_time': data_tdt_streams.start_time,
            'channel': data_tdt_streams.channel
        })

    return (tidy_streams_info)


def extract_stream_data(data_tdt_streams):
    # return data for an individual tdt stream

    tidy_streams_data = pd.DataFrame({
        'channel': data_tdt_streams.name,
        'raw_au': data_tdt_streams.data
    })

    return (tidy_streams_data)


def tidy_tdt_streams(data):
    # returns data and info from all streams within tdt structure

    first_concat = 1

    for stream in data.streams.keys():
        if (search('_\d\d\d[A-Za-z]', stream)):  # if stream contains _###C (id for fp channels; filters tdt preprocessed streams)
            if (first_concat):
                streams_info = extract_stream_info(data.streams[stream])
                streams_data = extract_stream_data(data.streams[stream])

                streams_info.rename(index={0: 'isosbestic'}, inplace=True)  # see if I can include names for each channel
                
                first_concat = 0;
            else:
                streams_info = pd.concat([streams_info, extract_stream_info(data.streams[stream])])
                streams_data = pd.concat([streams_data, extract_stream_data(data.streams[stream])])

                streams_info.rename(index={0: 'GCaMP'}, inplace=True)  # see if I can include names for each channel

                
    streams_info.insert(0, 'start_date', data.info.start_date)
    streams_info.insert(0, 'blockname', data.info.blockname)

    streams_data.insert(0, 'start_date', data.info.start_date)
    streams_data.insert(0, 'blockname', data.info.blockname)

    return (streams_info, streams_data)



# epochs ---------------------------------------------------------------------------------------------------------------
def extract_epoch_info(data_tdt_streams):
    # return info for an individual tdt epoc

    tidy_epoch_info = pd.DataFrame(
        {
            'name': data_tdt_streams.name,
            'type': data_tdt_streams.type,
            'type_str': data_tdt_streams.type_str,
            'dform': data_tdt_streams.dform,
            'size': [data_tdt_streams.size]
        })

    return (tidy_epoch_info)


def extract_epoch_data(data_tdt_streams):
    # return data for an individual tdt epoc

    tidy_epoch_data = pd.DataFrame(
        {'name': data_tdt_streams.name,
         'onset': data_tdt_streams.onset,
         'offset': data_tdt_streams.offset,
         'data': data_tdt_streams.data
         })

        # may need to modify this to include the "data" array, when I'm using Port A / port B for byte communication
    
    return (tidy_epoch_data)


def tidy_tdt_epocs(data):
    flag_epoch = 0

    for epoch in data.epocs.keys():
        flag_epoch = 1

        if (epoch == list(data.epocs.keys())[0]):
            epocs_info = extract_epoch_info(data.epocs[epoch])
            epocs_data = extract_epoch_data(data.epocs[epoch])

        else:
            epocs_info = pd.concat([epocs_info, extract_epoch_info(data.epocs[epoch])])
            epocs_data = pd.concat([epocs_data, extract_epoch_data(data.epocs[epoch])])

    if flag_epoch:
        epocs_info.insert(0, 'start_date', data.info.start_date)
        epocs_info.insert(0, 'blockname', data.info.blockname)

        epocs_data.insert(0, 'start_date', data.info.start_date)
        epocs_data.insert(0, 'blockname', data.info.blockname)

        return (epocs_info, epocs_data, flag_epoch)
    else:
        return(0,0, flag_epoch)

    
    
    


# extract and tidy all files in raw directory that are not located in extracted directory ------------------------------------------

# this is the big function, which will process files into a tidy organization and utilize the above functions to extract relevant info and data


def tidy_tdt_extract_and_tidy(dir_raw, dir_extracted):
    # return lists of files in raw and processed directories___
    raw_block_paths = os.listdir(dir_raw)
    processed_files = os.listdir(dir_extracted)

    # filter out hidden files from raw_block_paths
    raw_block_paths = list(filter(lambda raw_block_paths: raw_block_paths.find('.') == -1, raw_block_paths))

    # trim file names to blocknames
    processed_files = {x.replace('_streams_info.feather','') for x in processed_files}
    processed_files = {x.replace('_streams_data.feather','') for x in processed_files}
    processed_files = {x.replace('_epocs_info.feather','')   for x in processed_files}
    processed_files = {x.replace('_epocs_data.feather','')   for x in processed_files}
    processed_files = {x.replace('_info.feather','')         for x in processed_files}

    processed_files = list(processed_files)

    # remove files that have already been processed____________
    process_block_paths = raw_block_paths

    for processed_file in processed_files:
        try:
            while True:
                process_block_paths.remove(processed_file)
        except ValueError:
            pass


    if len(process_block_paths) >= 1:
        print('')
        print('extracting data from dir: ' + dir_raw + ' to dir: ' + dir_extracted)

        for process_block_path in process_block_paths:
            block_path = os.path.join(dir_raw, process_block_path)

            print("extracting blockpath: " + block_path)

            data = read_block(block_path, evtype = ['all'])

            data_info = tidy_tdt_info(data)
            streams_info, streams_data = tidy_tdt_streams(data)
            epocs_info, epocs_data, flag_epoch = tidy_tdt_epocs(data)

            session_id = data_info['blockname'][0]

            data_info.to_feather(os.path.join(dir_extracted, session_id +'_info.feather'))

            streams_info.reset_index(drop = True).to_feather(os.path.join(dir_extracted, session_id + '_streams_info.feather'))
            streams_data.reset_index(drop = True).to_feather(os.path.join(dir_extracted, session_id + '_streams_data.feather'))

            if(flag_epoch):
                epocs_info.reset_index(drop = True).to_feather(os.path.join(dir_extracted, session_id + '_epocs_info.feather'))
                epocs_data.reset_index(drop = True).to_feather(os.path.join(dir_extracted, session_id + '_epocs_data.feather'))
    else:
        print('no files to extract... all fp in dir :'+ dir_raw + ' has already been extracted to dir: ' + dir_extracted)

        
        
        
# plot raw traces, create the time vector and fs variable ----------------------------------------------------------------------------------
# this has been updated to handle files from parallel recordings in four rats

    ## updated 05.12.25 to be more flexible, should work regardless of how many boxes where recorded from.
    
        ### updated to fix time vector issues, may not work for TDT yet
    
# updating documentation, 07.17.26


def streams_peek_ratpack_tdt_or_doric(streams_info, streams_data, box_to_rat, zoomtemp):
    import numpy as np
    import matplotlib.pyplot as plt

    # create some empty dictionaries
    box_data = {}
    rat_data = {}
    
    ts_dict = None
    fs = streams_info['fs'][0]

    if 'channel_raw' in streams_data.columns and streams_data['channel_raw'].str.contains('Headstage').any():
        print("🔍 Detected format: Doric")
        data_format = 'Doric'

        # Build box_data from Doric data (extract channels per box using nicknames)
        # this block uses pandas filtering to parse down the larger streams_data dataframe to only contain the photometry data of interest,
        # skipping boxes with no animal in them
        for box, rat in box_to_rat.items():
            if rat == 'null':
                continue
            ch_415 = streams_data[(streams_data['nickname'] == rat) & (streams_data['channel_name'].str.contains('415nm'))]
            ch_470 = streams_data[(streams_data['nickname'] == rat) & (streams_data['channel_name'].str.contains('470nm'))]

            if ch_415.empty or ch_470.empty:
                print(f"⚠️ Missing expected channels for {box} ({rat}). Skipping.")
                continue

            # the .values method converts the pandas series (the 'raw_au' column) into a numpy array
            isos_405 = ch_415['raw_au'].values
            GCaMP_465 = ch_470['raw_au'].values

            box_data[box] = {'405': isos_405, '465': GCaMP_465}

        if not box_data:
            raise ValueError("No matching photometry channels found in the Doric data.")

        # Create time vector dictionary, one per box
        ts_dict = {}
        for box, rat in box_to_rat.items():
            if rat == 'null':
                continue
            ch_415 = streams_data[(streams_data['nickname'] == rat) & (streams_data['channel_name'].str.contains('415nm'))]
            ch_470 = streams_data[(streams_data['nickname'] == rat) & (streams_data['channel_name'].str.contains('470nm'))]
            
            # this line checks that the channels have equivalent time vectors 
            if not np.allclose(ch_415['time'].values, ch_470['time'].values):
                raise ValueError(f"Mismatch in time vectors for box {box}.")
            ts_dict[box] = ch_415['time'].values

        print(f"\n✅ Found photometry data for boxes: {', '.join(box_data.keys())}\n")

        # Map box data to rat_data by rat nickname
        for box, rat in box_to_rat.items():
            if rat == 'null' or box not in box_data:
                continue
            rat_data[rat] = {
                'isos_405': box_data[box]['405'],
                'GCaMP_465': box_data[box]['465']
            }

        # Plotting Doric data (per box time vectors)
        fig_0, ax1 = plt.subplots(figsize=(8, 4))
        
        # use .twinx from matplotlib to plot both GCaMP and isos signals on one plot
        ax2 = ax1.twinx()
        lines = []
        colors_465 = [[0.8, 0, 0], [1, 0.7, 0], [0, 0.5, 0.8], [0.4, 0, 0.8]]
        colors_405 = [[0.7, 0.7, 0.7], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]

        # use the built-in function enumerate() to have i as a counter that retrieves a color from the lists above
        for i, (box, data) in enumerate(box_data.items()):
            ts = ts_dict[box]
            plot_465 = ax1.plot(ts, data['465'], color=colors_465[i], label=f'465 - GCaMP_{box}')
            plot_405 = ax2.plot(ts, data['405'], color=colors_405[i], label=f'405_{box}')
            lines.extend(plot_465 + plot_405) # not append, because it's adding multiple objects in each iteration

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('465 fluor.', color=[0.5, 0, 0])
        ax2.set_ylabel('405 fluor.', color=[0, 0, 0.5])
        ax1.set_title('Raw signals')
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right', bbox_to_anchor=(1.50, 0.98))

        # Zoomed plot
        fig_1, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        lines = []

        for i, (box, data) in enumerate(box_data.items()):
            ts = ts_dict[box]
            plot_465 = ax1.plot(ts, data['465'], color=colors_465[i], label=f'465 - GCaMP_{box}')
            plot_405 = ax2.plot(ts, data['405'], color=colors_405[i], label=f'405_{box}')
            lines.extend(plot_465 + plot_405)

        plt.xlim(zoomtemp)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('465 fluor.', color=[0.5, 0, 0])
        ax2.set_ylabel('405 fluor.', color=[0, 0, 0.5])
        ax1.set_title('Raw signals (Zoomed)')
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right', bbox_to_anchor=(1.50, 0.98))

    else:
        print("🔍 Detected format: TDT")
        data_format = 'TDT'

        # box - channel mapping is hardcoded, since the TDT (unlike wireless Doric) always has channels corresponding to specific boxes
        box_labels = {
            'BOX2': ('_405A', '_465A'),
            'BOX3': ('_405C', '_465C'),
            'BOX4': ('_405E', '_465E'),
            'BOX5': ('_405G', '_465G'),
        }

        for box, (ch405, ch465) in box_labels.items():
            if ch405 in streams_data['channel'].values and ch465 in streams_data['channel'].values:
                isos_405 = streams_data[streams_data['channel'] == ch405]['raw_au'].values
                GCaMP_465 = streams_data[streams_data['channel'] == ch465]['raw_au'].values

                assert len(isos_405) == len(GCaMP_465), f"Data length mismatch in box {box}"

                box_data[box] = {'405': isos_405, '465': GCaMP_465}

                if box_to_rat and box in box_to_rat:
                    rat = box_to_rat[box]
                    if rat.lower() != 'null':
                        rat_data[rat] = {'isos_405': isos_405, 'GCaMP_465': GCaMP_465}

        if not box_data:
            raise ValueError("No matching photometry channels found in the TDT data.")

        # Use actual time values from streams_data for each box
        ts_dict = {}
        for box in box_data.keys():
            ch465 = box_labels[box][1]
            time_vals = streams_data[streams_data['channel'] == ch465]['time'].values
            if len(time_vals) != len(box_data[box]['465']):
                raise ValueError(f"Time and data length mismatch for box {box}: time={len(time_vals)} vs data={len(box_data[box]['465'])}")
            ts_dict[box] = time_vals

        print(f"\n✅ Found photometry data for boxes: {', '.join(box_data.keys())}\n")

        # Plotting TDT data (per box time vectors)
        fig_0, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        lines = []

        colors_465 = [[0.8, 0, 0], [1, 0.7, 0], [0, 0.5, 0.8], [0.4, 0, 0.8]]
        colors_405 = [[0.7, 0.7, 0.7], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]

        for i, (box, data) in enumerate(box_data.items()):
            ts = ts_dict[box]
            plot_465 = ax1.plot(ts, data['465'], color=colors_465[i], label=f'465 - GCaMP_{box}')
            plot_405 = ax2.plot(ts, data['405'], color=colors_405[i], label=f'405_{box}')
            lines.extend(plot_465 + plot_405)

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('465 fluor.', color=[0.5, 0, 0])
        ax2.set_ylabel('405 fluor.', color=[0, 0, 0.5])
        ax1.set_title('Raw signals')
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right', bbox_to_anchor=(1.50, 0.98))

        # Zoomed plot
        fig_1, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        lines = []

        for i, (box, data) in enumerate(box_data.items()):
            ts = ts_dict[box]
            plot_465 = ax1.plot(ts, data['465'], color=colors_465[i], label=f'465 - GCaMP_{box}')
            plot_405 = ax2.plot(ts, data['405'], color=colors_405[i], label=f'405_{box}')
            lines.extend(plot_465 + plot_405)

        plt.xlim(zoomtemp)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('465 fluor.', color=[0.5, 0, 0])
        ax2.set_ylabel('405 fluor.', color=[0, 0, 0.5])
        ax1.set_title('Raw signals (Zoomed)')
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right', bbox_to_anchor=(1.50, 0.98))

    return rat_data, fs, ts_dict, fig_0, fig_1, data_format



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# updated documentation 07.20.26

def remove_artifacts_dual_signal(
    time_vector,
    signal1,
    signal2,
    threshold_drop_signal1=0.15,
    threshold_drop_signal2=0.05,
    sudden_drop_threshold_signal1=0.1,
    sudden_drop_threshold_signal2=0.03,
    min_artifact_duration=1,
    max_artifact_duration=500,
    sampling_rate=100,
    median_filter_size=11,
    baseline_exclusion_seconds=0,
    manual_windows=None,  # list of (start, end) in seconds
    plot=True,
    signal1_label='Signal 1',
    signal2_label='Signal 2',
    zoom_start=60,
    zoom_window=120
):
    # define a sub-function for the artifact detection and rejection
    # the artifact detection is based on a drop in amplitude below a certain threshold, 
    # because the Doric wireless system has periodic "drops" in signal where the amplitude plummets essentially in negative square pulses
    
    def clean_signal(signal, threshold_drop, sudden_drop_threshold, signal_name):
        signal = np.array(signal)
        cleaned = signal.copy() # preserve the original signal

        # the beginning may be excluded, e.g. because that recording period includes LEDs turning on, headstage placement, etc. 
        exclusion_samples = int(baseline_exclusion_seconds * sampling_rate) 
        
        # the purpose of the median filter is to provide a comparison "signal" to detect sharp deviations in local amplitude
        baseline = median_filter(signal, size=median_filter_size)

        # Automatic artifact detection
        # firstly, create a boolean array of false 
        drop_mask_amp = np.zeros_like(signal, dtype=bool)
        # boolean mask is updated to detect deviations larger than the threshold
        drop_mask_amp[exclusion_samples:] = (baseline[exclusion_samples:] - signal[exclusion_samples:]) > (threshold_drop * baseline[exclusion_samples:])
        # repeat with another boolean mask, this one using the differential to detect sudden drops across samples (common with the Doric signal "drops")
        diff_signal = np.diff(signal, prepend=signal[0])
        drop_mask_sudden = np.zeros_like(signal, dtype=bool)
        drop_mask_sudden[exclusion_samples:] = diff_signal[exclusion_samples:] < -sudden_drop_threshold
        # combine the two boolen arrays
        drop_mask = drop_mask_amp | drop_mask_sudden

        # this nd_label comes from scipy, it labels discrete artifacts (e.g. consecutive True points in the drop_mask are a single artifact)
        labeled, num_features = nd_label(drop_mask)
        # a buffer is used to ensure the entire artifact is removed along with onset / offset effects
        buffer = int(0.1 * sampling_rate)
        
        # this loop addes the buffer samples to flank each artifact, using the output of nd_label
        # artifacts that don't satisfy criteria (e.g. for duration) are ignored
        # numpy interpolation is performed between the identified indices for each artifact
        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            start_idx = max(indices[0] - buffer, 0)
            end_idx = min(indices[-1] + buffer, len(signal) - 1)
            indices_expanded = np.arange(start_idx, end_idx + 1)
            duration = len(indices_expanded)

            if duration < min_artifact_duration or duration > max_artifact_duration:
                continue
            if start_idx == 0 or end_idx >= len(signal) - 1:
                continue
            if start_idx < exclusion_samples:
                continue

            # Interpolate: x are the indices to keep, flanking the artifact, and y are the values at those indices
            x = [start_idx - 1, end_idx + 1]
            y = [cleaned[start_idx - 1], cleaned[end_idx + 1]]
            cleaned[indices_expanded] = np.interp(indices_expanded, x, y)

        # Manual artifact interpolation
        if manual_windows is not None:
            for start_sec, end_sec in manual_windows:
                start_idx = np.searchsorted(time_vector, start_sec) # find the indices corresponding to specific time points
                end_idx = np.searchsorted(time_vector, end_sec)
                if start_idx == 0 or end_idx >= len(signal) - 1:
                    continue
                indices_manual = np.arange(start_idx, end_idx + 1)
                x = [start_idx - 1, end_idx + 1]
                y = [cleaned[start_idx - 1], cleaned[end_idx + 1]]
                cleaned[indices_manual] = np.interp(indices_manual, x, y)

        return cleaned, baseline

    # Clean both signals
    signal1_clean, signal1_baseline = clean_signal(signal1, threshold_drop_signal1, sudden_drop_threshold_signal1, signal1_label)
    signal2_clean, signal2_baseline = clean_signal(signal2, threshold_drop_signal2, sudden_drop_threshold_signal2, signal2_label)

    if plot:
        plt.style.use('dark_background')
        ts = np.array(time_vector)

        signal1 = np.array(signal1)
        signal2 = np.array(signal2)
        signal1_clean = np.array(signal1_clean)
        signal2_clean = np.array(signal2_clean)

        signal1_ylim = [np.min(signal1) * 0.9, np.max(signal1) * 1.1]
        signal2_ylim = [np.min(signal2) * 0.9, np.max(signal2) * 1.1]

        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Plot raw and cleaned Signal 1
        ax1.plot(ts, signal1, color='yellow', linewidth=4, alpha=0.3, label=f'{signal1_label} (raw)', zorder=1)
        ax1.plot(ts, signal1_clean, color=[0.1, 0.7, 0.2], linewidth=2, label=f'{signal1_label} (cleaned)', zorder=3)
        ax1.set_ylabel(f'{signal1_label}', color='lightgreen')
        ax1.set_ylim(signal1_ylim)

        # Plot raw and cleaned Signal 2
        ax2 = ax1.twinx()
        ax2.plot(ts, signal2, color='magenta', linewidth=4, alpha=0.3, label=f'{signal2_label} (raw)', zorder=1)
        ax2.plot(ts, signal2_clean, color=[0.7, 0.7, 0.7], linewidth=2, label=f'{signal2_label} (cleaned)', zorder=3)
        ax2.set_ylabel(f'{signal2_label}', color='gray')
        ax2.set_ylim(signal2_ylim)

        ax1.set_xlabel('Time (seconds)')
        plt.title('Artifact Removal: Raw vs Cleaned Signals')

        # Zoom overlay
        ax1.axvspan(zoom_start, zoom_start + zoom_window, color='blue', alpha=0.1, label='Zoom window')

        # Manual interpolation overlays
        if manual_windows is not None:
            for i, (start_sec, end_sec) in enumerate(manual_windows):
                lbl = 'Manual interpolation' if i == 0 else None  # avoid duplicate legend entries
                ax1.axvspan(start_sec, end_sec, color='red', alpha=0.2, label=lbl)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()

        # Zoomed-in plot
        fig_zoom, ax1_zoom = plt.subplots(figsize=(12, 4))
        zoom_mask = (ts >= zoom_start) & (ts <= zoom_start + zoom_window)
        ax1_zoom.plot(ts[zoom_mask], signal1[zoom_mask], color='yellow', linewidth=2, label=f'{signal1_label} (raw)', zorder=1)
        ax1_zoom.plot(ts[zoom_mask], signal1_baseline[zoom_mask], color='red', linewidth=2, label=f'{signal1_label} (median filter)', zorder=5)
        ax1_zoom.plot(ts[zoom_mask], signal1_clean[zoom_mask], color=[0.1, 0.7, 0.2], linewidth=2, label=f'{signal1_label} (cleaned)', zorder=3)
        ax2_zoom = ax1_zoom.twinx()
        ax2_zoom.plot(ts[zoom_mask], signal2[zoom_mask], color='magenta', linewidth=2, label=f'{signal2_label} (raw)', zorder=1)
        ax2_zoom.plot(ts[zoom_mask], signal2_baseline[zoom_mask], color='red', linewidth=2, label=f'{signal2_label} (median filter)', zorder=5)
        ax2_zoom.plot(ts[zoom_mask], signal2_clean[zoom_mask], color=[0.7, 0.7, 0.7], linewidth=2, label=f'{signal2_label} (cleaned)', zorder=3)

        ax1_zoom.set_ylabel(f'{signal1_label}', color='lightgreen')
        ax2_zoom.set_ylabel(f'{signal2_label}', color='gray')
        ax1_zoom.set_xlabel('Time (seconds)')
        plt.title(f'Zoomed-In View: {zoom_start:.1f} to {zoom_start + zoom_window:.1f} sec')

        # Manual interpolation overlays in zoom
        if manual_windows is not None:
            for i, (start_sec, end_sec) in enumerate(manual_windows):
                lbl = 'Manual interpolation' if i == 0 else None
                ax1_zoom.axvspan(start_sec, end_sec, color='cyan', alpha=0.2, label=lbl)

        # Combine zoom legends
        lines1, labels1 = ax1_zoom.get_legend_handles_labels()
        lines2, labels2 = ax2_zoom.get_legend_handles_labels()
        ax1_zoom.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        plt.show()

    return signal1_clean, signal2_clean



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# updated documentation 07.20.26

def filt_downsamp_streams(isos_405_all, GCaMP_465_all, ts_all, fs, new_fs, filt, zoom_start=0, zoom_window=10):
    """
    Filters and downsamples fiber photometry signals, preserving the true time vector via interpolation.
    Plots raw and filtered/downsampled signals for comparison.
    
    Parameters:
    - isos_405_all: np.ndarray, raw 405 signal
    - GCaMP_465_all: np.ndarray, raw 465 signal
    - ts_all: np.ndarray, original time vector (may be non-uniform)
    - fs: float, original sampling frequency (Hz)
    - new_fs: float, target downsampled frequency (Hz)
    - filt: float, low-pass filter cutoff (Hz)
    - zoom_start: float, optional zoom start time (sec)
    - zoom_window: float, optional zoom window size (sec)
    
    Returns:
    - isos_405_downsamp: downsampled 405 signal
    - GCaMP_465_downsamp: downsampled 465 signal
    - ts_downsamp: downsampled time vector
    - fig_8: matplotlib Figure of full signals
    - fig_zoom: matplotlib Figure of zoomed signals
    """
    from scipy.signal import butter, filtfilt
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('dark_background')

    # --- Filter signals ---
    # create and apply the filter, uses scipy
    # this uses a butterworth filter applied forward and backward to prevent phase shift
    b, a = butter(2, filt, btype='low', fs=fs)
    isos_405_filt = filtfilt(b, a, isos_405_all)
    GCaMP_465_filt = filtfilt(b, a, GCaMP_465_all)

    # --- Downsample via interpolation ---
    # the numpy interpolation here estimates the signal values corresponding to the new downsampled time vector points
    # interpolation preserves exact temporal correspondance between the time vector and the signal
    step = 1 / new_fs
    ts_downsamp = np.arange(ts_all[0], ts_all[-1], step)
    isos_405_downsamp = np.interp(ts_downsamp, ts_all, isos_405_filt)
    GCaMP_465_downsamp = np.interp(ts_downsamp, ts_all, GCaMP_465_filt)

    # --- Full Signal Plot ---
    fig_8, ax1 = plt.subplots()
    ax1.plot(ts_all, GCaMP_465_all, color='red', alpha=0.8, linewidth=2, label='GCaMP_465 (raw)')
    ax1.plot(ts_downsamp, GCaMP_465_downsamp, color=[0, 1, 0], linewidth=1.5, label='GCaMP_465 (filtered+downsamp)')

    ax2 = ax1.twinx()
    ax2.plot(ts_all, isos_405_all, color='cyan', alpha=0.4, linewidth=2, label='isos_405 (raw)')
    ax2.plot(ts_downsamp, isos_405_downsamp, color=[1,0,1], linewidth=1.5, label='isos_405 (filtered+downsamp)')

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GCaMP_465', color='green')
    ax2.set_ylabel('isos_405', color='gray')
    ax1.set_title('Full Signals: Raw vs Filtered+Downsampled')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # --- Zoomed-In Plot ---
    zoom_end = zoom_start + zoom_window
    raw_mask = (ts_all >= zoom_start) & (ts_all <= zoom_end)
    down_mask = (ts_downsamp >= zoom_start) & (ts_downsamp <= zoom_end)

    fig_zoom, ax1_zoom = plt.subplots()
    # Raw traces
    ax1_zoom.plot(ts_all[raw_mask], GCaMP_465_all[raw_mask], color='red', alpha=0.8, linewidth=2, label='GCaMP_465 (raw)')
    ax2_zoom = ax1_zoom.twinx()
    ax2_zoom.plot(ts_all[raw_mask], isos_405_all[raw_mask], color='cyan', alpha=0.4, linewidth=2, label='isos_405 (raw)')

    # Filtered + downsampled
    ax1_zoom.plot(ts_downsamp[down_mask], GCaMP_465_downsamp[down_mask], color=[0, 1, 0], linewidth=3, label='GCaMP_465 (filtered+downsamp)')
    ax2_zoom.plot(ts_downsamp[down_mask], isos_405_downsamp[down_mask], color=[1,0,1], linewidth=3, label='isos_405 (filtered+downsamp)')

    # Set plot limits
    ax1_zoom.set_xlim(zoom_start, zoom_end)
    #ax1_zoom.set_ylim([np.min(GCaMP_465_downsamp[down_mask]) * 0.9, np.max(GCaMP_465_downsamp[down_mask]) * 1.1])
    #ax2_zoom.set_ylim([np.min(isos_405_downsamp[down_mask]) * 0.9, np.max(isos_405_downsamp[down_mask]) * 1.1])

    ax1_zoom.set_xlabel('Time (seconds)')
    ax1_zoom.set_ylabel('GCaMP_465', color='green')
    ax2_zoom.set_ylabel('isos_405', color='gray')
    ax1_zoom.set_title(f'Zoomed-In Signals: {zoom_start:.1f}-{zoom_end:.1f} sec')
    ax1_zoom.legend(loc='upper left')
    ax2_zoom.legend(loc='upper right')

    return isos_405_downsamp, GCaMP_465_downsamp, ts_downsamp, fig_8, fig_zoom


    
# extract a designated segment of the stream data, plot ----------------------------------------------------------------------------------

    ## new version for Doric analysis
    ## updated along with many other functions 10.16.25 to fix alignment issues
    ### need to apply similar fixes to TDT functions. 
    
# updated documentation 07.22.26

    # this code block is used to load the event timestamps and identifying information
    # additionally, the photometry data is trimmed to a specified duration
    # the timespan to be further analyzed is defined based on behavioral or task events represented by the event timestamps
    # for example, during intermittent-access self-admin sessions, the onset of drug availability is signaled by a TTL 
      ## if the objective is to examine the photometry signal during the 5 min available period, one uses segment mode "during" and defines "keepmin_streams" as 5. 

    # photometry data preceding and following the window of interest is discarded, 
    ## and all photometry and event timestamp data is aligned such that ts = 0 at the beginning of the specified interval
    # segment modes can be selected to keep photometry data preceding the availability onset, following the available period, or capturing both transitions


def streams_trim_events_doric(isos_405_filt_downsamp, GCaMP_465_filt_downsamp,
                      dropsec_streams, keepmin, ts_downsamp,
                      new_fs, y1, y2, x, startcode,
                      epoc_data=None, box=None, noncont=False, tzoom=None, segment_mode=None, pre_buffer_sec=15, post_buffer_sec=15):
    """
    Modified for new Doric epoc_data format with string-based 'input code'.
    Returns trimmed, filtered streams and figures.
    
    segment_mode options:
        'before' → keep the X minutes *before* the TTL
        'during' → keep the X minutes *starting at* the TTL
        'after'  → keep the X minutes *after* the available period (offset by keepmin)

    
    """
    
    # Set default buffers to 0 if None
    if pre_buffer_sec is None:
        pre_buffer_sec = 0
    if post_buffer_sec is None:
        post_buffer_sec = 0
    
    print(f'the downsampled sampling rate is {new_fs} Hz\n')
    
    # this is how much of the recording to discard, expressed in samples
    disc = int(new_fs * dropsec_streams)
    print(f'discard the first {dropsec_streams} seconds, data points from 0 to {disc}')
    
    # this is how much of the recording to keep, expressed in samples
    intvl = int(keepmin * 60 * new_fs)
    
    # Use searchsorted to find the index where time vector crosses dropsec_streams
    ttl_idx = np.searchsorted(ts_downsamp, dropsec_streams)  # index of dropsec_code event
    

    # --- Choose segment based on mode ---
    # this is mainly just relevant for IntA SA recordings, others would just be None / during
    # the purpose of the buffers is to capture events that occur very close to the beginning or end of the analyzed interval
    
    if segment_mode is None:
        start = max(ttl_idx - int(pre_buffer_sec * new_fs), 0)         # <-- add pre-buffer
        end = ttl_idx + intvl + int(post_buffer_sec * new_fs)          # <-- add post-buffer
        span = (dropsec_streams - pre_buffer_sec, dropsec_streams + keepmin * 60 + post_buffer_sec)
        segment_label = f"segment ({keepmin} min starting at TTL, ±{pre_buffer_sec}/{post_buffer_sec} s buffer)"
        
    elif segment_mode == 'before':
        start = max(ttl_idx - intvl, 0)
        end = ttl_idx
        span = (dropsec_streams - keepmin * 60, dropsec_streams)
        segment_label = f"PRECEDING unavailable segment ({keepmin} min before TTL)"   
    elif segment_mode == 'during':
        start = max(ttl_idx - int(pre_buffer_sec * new_fs), 0)         # <-- add pre-buffer
        end = ttl_idx + intvl + int(post_buffer_sec * new_fs)          # <-- add post-buffer
        span = (dropsec_streams - pre_buffer_sec, dropsec_streams + keepmin * 60 + post_buffer_sec)
        segment_label = f"AVAILABLE segment ({keepmin} min starting at TTL, ±{pre_buffer_sec}/{post_buffer_sec} s buffer)"
    elif segment_mode == 'after':
        start = ttl_idx + intvl
        end = ttl_idx + 2 * intvl
        span = (dropsec_streams + 300, dropsec_streams + 300 + (keepmin * 60))
        segment_label = f"FOLLOWING unavailable segment ({keepmin} min after available period)"
    elif segment_mode == 'transitions':
        # 1 min before TTL to 1 min after available period ends
        pre_buffer = int(60 * new_fs)  # 1 min in samples
        post_buffer = int(60 * new_fs)
        start = max(ttl_idx - pre_buffer, 0)
        end = ttl_idx + intvl + post_buffer
        span = (dropsec_streams - 60, dropsec_streams + (keepmin * 60) + 60)
        segment_label = f"TRANSITION segment (1 min before → 1 min after available period, total {keepmin + 2:.1f} min)"

    else:
        raise ValueError("segment_mode must be one of ['before', 'during', 'after', 'transitions']")
        
        
    print(f'\nkeep a duration of {keepmin} minutes')
    
    # --- safety check, ensure that all vectors have the proper length ---
    if start < 0:
        start = 0
    if end > len(isos_405_filt_downsamp):
        end = len(isos_405_filt_downsamp)
        print(f'including to end of stream: data points {start} to {end}')
    else:
        print(f"Selected indices, included interval: start={start}, end={end} ({(end-start)/new_fs/60:.2f} min)")
   
    if not (
        len(isos_405_filt_downsamp)
        == len(GCaMP_465_filt_downsamp)
        == len(ts_downsamp)
    ):
        raise ValueError(
            "405, 465, and time vector must all have identical lengths."
        )
        
    # use the start and end indices defined above to slice the isos and GCaMP vectors
    isos_405 = isos_405_filt_downsamp[start:end]
    GCaMP_465 = GCaMP_465_filt_downsamp[start:end]
    
    # Preserve true time alignment: trim and shift the time vector to start at zero for alignment

    print('\n- - - - - initial vector lengths - - - - -\n')
    print(f"len(isos_405_filt_downsamp): {len(isos_405_filt_downsamp)}")
    print(f"len(ts_downsamp): {len(ts_downsamp)}")
    
    
    print(f"\nstart index: {start}")
    print(f"end index: {end}")

    if start >= len(ts_downsamp) or end <= start:
        raise ValueError(f"Invalid slice indices for trimming: start={start}, end={end}, array length={len(ts_downsamp)}")
    
    
    # Trim ts to match the signals, this includes the buffers if defined above
    ts = ts_downsamp[start:end]

    # Apply pre-buffer shift
    # Trim and shift ts to include pre-buffer
    # this step ensures that ts = 0 corresponds to the TTL for session start, drug availability, etc. 
    ts = ts - ts[0]                   # start at 0
    ts = ts - pre_buffer_sec           # now ts[0] = -pre_buffer_sec

  
    # for display purposes, confirmation, etc. measure the kept interval
    trimmed_sec = ts[-1] - ts[0]
    trimmed_min = trimmed_sec / 60

    print(f"\nTrimmed duration: {trimmed_sec:.2f} sec (~{trimmed_min:.2f} min) "
          f"(requested {keepmin} min, pre_buffer {pre_buffer_sec}s, post_buffer {post_buffer_sec}s)")


    # plot span adjustment
    # Trimmed span in the same coordinate system as ts
    # Highlight the main segment (excluding pre/post buffer)
    
    span_trimmed = (0, keepmin * 60)        # relative to ts


    # If there is a pre-buffer, ts[0] = -pre_buffer_sec, so main segment starts at 0 s
    # If no pre-buffer, ts[0] = 0, so main segment also starts at 0 s

    
    # Event mappings
    if noncont:
        event_labels = {
            'D09': 'program start',
            'D13': 'program start',
            'D25': 'program start',
            'D29': 'program start',
            'D10': 'cue light',
            'D14': 'cue light',
            'D26': 'cue light',
            'D30': 'cue light',
            'D11': 'house light',
            'D15': 'house light',
            'D27': 'house light',
            'D31': 'house light',
        }
        event_colors = {
            'D09': 'magenta',
            'D13': 'magenta',
            'D25': 'magenta',
            'D29': 'magenta',
            'D10': 'green',
            'D14': 'green',
            'D26': 'green',
            'D30': 'green',
            'D11': 'red',
            'D15': 'red',
            'D27': 'red',
            'D31': 'red',
        }
    else:
        event_labels = {
            'D09': 'program start / drug available',
            'D13': 'program start / drug available',
            'D25': 'program start / drug available',
            'D29': 'program start / drug available',
            'D10': 'active lever',
            'D14': 'active lever',
            'D26': 'active lever',
            'D30': 'active lever',
            'D11': 'inactive lever',
            'D15': 'inactive lever',
            'D27': 'inactive lever',
            'D31': 'inactive lever',
            'D12': 'drug infusion',
            'D16': 'drug infusion',
            'D28': 'drug infusion',
            'D32': 'drug infusion',
        }

        event_colors = {
            'D09': 'magenta', 'D13': 'magenta', 'D25': 'magenta', 'D29': 'magenta',
            'D10': 'yellow', 'D14': 'yellow', 'D26': 'yellow', 'D30': 'yellow',
            'D11': 'cyan', 'D15': 'cyan', 'D27': 'cyan', 'D31': 'cyan',
            'D12': 'red', 'D16': 'red', 'D28': 'red', 'D32': 'red',
        }

    
    # Filter epoc_data for current box and D codes
    epoc_filtered = None

    if epoc_data is not None and box is not None:
        epoc_data = epoc_data.copy()
        epoc_data['D_code'] = epoc_data['input code'].str.extract(r'^(D\d{2})')[0]
        epoc_data['box_match'] = epoc_data['input code'].str.contains(box, case=False, na=False)

        epoc_filtered = epoc_data[
            epoc_data['D_code'].isin(event_labels.keys()) & epoc_data['box_match']
        ].copy()

        # For the trimmed plot (second plot)
        # this ensures that the event times are aligned to the time vector above
        # Transform absolute onset times into the trimmed time axis.
        # as above:
        #     ts = ts_downsamp[start:end]
        #     ts = ts - ts[0]
        #     ts = ts - pre_buffer_sec
        epoc_filtered['ts_aligned'] = epoc_filtered['onset'].values - ts_downsamp[start] - pre_buffer_sec

        # For the full plot (first plot)
        epoc_filtered['ts_untrimmed'] = epoc_filtered['onset'].values
  
    # plotting function
    # span is the interval for further analysis
    # epoc filtered is defined above, the epoc data file trimmed to box of interest
    
    
    def plot_block(ts, sig1, sig2, title, span=None, xlim_range=None,
               ylim1=None, ylim2=None, epoc_filtered=None, trimmed=True):

        fig, ax1 = plt.subplots()
        ax1.plot(ts, sig1, color=[0.7, 0.7, 0.7], label='405')

        ax2 = ax1.twinx()
        ax2.plot(ts, sig2, color=[0, 1, 0], label='465')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('405 (mV)', color=[0.7, 0.7, 0.7])
        ax2.set_ylabel('465 (mV)', color=[0, 1, 0])
        ax1.set_title(title)
        ax1.set_ylim(0.3, 0.8)
        ax2.set_ylim(0.5, 1.0)

        if ylim1 is not None:
            ax1.set_ylim(ylim1)
        if ylim2 is not None:
            ax2.set_ylim(ylim2)     

        if xlim_range is not None:
            ax1.set_xlim(xlim_range)

        if span is not None:
            # use .axvspan method to shade the window corresponding to the interval of interest
            ax1.axvspan(*span, color='magenta', alpha=0.1, label='Kept Window')

        # -----------------------------------------
        # EVENT TICKS WITH LABEL DEDUPLICATION
        # -----------------------------------------
        event_handles = [] # empty list
        event_labels_seen = set()   # <--- prevents duplicate legend entries

        if epoc_filtered is not None:
            col = 'ts_aligned' if trimmed else 'ts_untrimmed'

            for code, behav_label in event_labels.items():
                # in each iteraction, extract the timestamps corresponding to an event code from the above dictionary
                # times is a numpy array of the timestamps for that event type
                times = epoc_filtered.loc[
                    epoc_filtered['D_code'] == code, col
                ].values

                if len(times) == 0:
                    continue

                # draw the ticks
                for t in times:
                    if xlim_range is None or (xlim_range[0] <= t <= xlim_range[1]):
                        ax1.axvline(t, color=event_colors[code],
                                    linestyle='--', linewidth=1)

                # create **one** handle per behavior label
                if behav_label not in event_labels_seen:
                    h, = ax1.plot([], [], color=event_colors[code],
                                  linestyle='--', label=behav_label)
                    
                    # create a growing list of line2D objects for the event types
                    event_handles.append(h)
                    event_labels_seen.add(behav_label)

            # -----------------------------
            # MERGE LEGENDS & REMOVE DUPLICATES
            # -----------------------------
            # collect the handles and labels from the plots above, i.e. 405 and 465 lines
            
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            # Start combined lists
            handles_all = h1 + h2 + event_handles
            labels_all = l1 + l2 + [h.get_label() for h in event_handles]  # line2D objects have the label attribute

            # Deduplicate based on label while preserving order
            seen = set()
            handles_final = []
            labels_final = []

            # create a zip object that can reference pairs of handles and labels from the above lists
            # create the final trimmed lists for the legend
            for h, lab in zip(handles_all, labels_all):
                if lab not in seen:
                    seen.add(lab)
                    handles_final.append(h)
                    labels_final.append(lab)

            ax1.legend(handles_final, labels_final,
                       loc='upper left', bbox_to_anchor=(1.05, 1),
                       fontsize=8)
        return fig


    # --- Plots ---
    
    print(f'\nPlotting {segment_label} ...')
    
 
   # Full plot
    fig2a = plot_block(
        ts_downsamp, isos_405_filt_downsamp, GCaMP_465_filt_downsamp,
        f'Untrimmed streams, {segment_label}',
        xlim_range=None, span=span, epoc_filtered=epoc_filtered, trimmed=False
    )

    # Trimmed plot
    fig2b = plot_block(
        ts, isos_405, GCaMP_465,
        f'Trimmed {segment_label} (time shifted to 0 s)',
        xlim_range=tzoom, span=span_trimmed,
        #ylim1=(np.min(isos_405) * 0.8, np.max(isos_405) * 1.5),
        #ylim2=(np.min(GCaMP_465) * 0.8, np.max(GCaMP_465) * 1.5),
        epoc_filtered=epoc_filtered, trimmed=True
    )


    return (fig2a, fig2b, isos_405, GCaMP_465, ts)







# extract time points / indices for epoc event onsets ----------------------------------------------------------------------------------

# updated 11.07.25 to accommodate doric or tdt data
# documentation updated 07.22.26

def epoc_onsets_FR(epoc_data, keepmin, ts, box, new_fs, dropsec, data_format):
   
    """
    Aligns epoc onset timestamps with downsampled and trimmed stream data.

    Parameters:
        epoc_data (DataFrame): The original epoc dataset.
        keepmin (float): Minutes of data to keep.
        trange (tuple): Time window of interest around each event.
        ts (np.array): Time vector corresponding to trimmed stream data.
        box (str): Identifier for the box/subject.
        new_fs (float): New (downsampled) sampling rate.
        dropsec (float): Time in seconds where trimmed stream begins.

    Returns:
        epoc_ts_and_indices (DataFrame): Epoc timestamps, stream indices, and labels.
        ts_epocs (np.array): Indices in trimmed time vector for each epoc.
    """
    
    # Ensure dropsec is scalar
    if isinstance(dropsec, (np.ndarray, list)):
        dropsec = float(dropsec[0])
    
    # Filter epoc data for relevant box
    if data_format == 'TDT':
        epoc_data_trimmed = epoc_data[epoc_data['name'] == box]
    elif data_format == 'Doric':
        epoc_data_trimmed = epoc_data[epoc_data['input code'].str.contains(box, case=False, na=False)]
    
    print(f'dropsec is equal to {dropsec}')

    # Trim epoc events to the kept stream interval
    start_time = dropsec
    end_time = dropsec + keepmin * 60
    epoc_data_trimmed = epoc_data_trimmed[
        (epoc_data_trimmed['onset'] >= start_time) &
        (epoc_data_trimmed['onset'] <= end_time)
    ]


    # Translate event codes to labels - MedPC programs after 10.01.25, coh5
    
    if data_format == 'TDT':
        
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['data', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec

        # Find closest index in trimmed stream time vector
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        # Translate event codes to labels - MedPC programs prior to 10.01.25
        '''
        event_map = {
            0: 'null',
            1: 'trial begin',
            2: 'active lever',
            4: 'inactive lever',
            8: 'drug infusion',
            10: 'drug infusion + active lever',
            16: 'trial end',
        }
        '''

        # Translate event codes to labels - MedPC programs after 10.01.25, coh5
        event_map = {
            0: 'null',
            17: 'trial begin',
            2: 'active lever',
            4: 'inactive lever',
            8: 'drug infusion',
            10: 'drug infusion + active lever',
        }

        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['data']]
        
        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['data'].values,
            'event': events
        }).reset_index(drop=True)
        
        
    elif data_format == 'Doric':
        
        epoc_data_trimmed['D_code'] = epoc_data_trimmed['input code'].str.extract(r'(D\d{2})')
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['D_code', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        
        # convert event timestamps (in seconds, relative to TTL) to indices that can be aligned with photometry data
        # np.searchsorted returns the indices of ts that correspond to the values of epoc_data_sorted_shift
        # .clip restricts to valid indices, with a minimum index of 0 and a maximum of the length of ts minus 1
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        event_map = {
            'D09': 'program start',
            'D13': 'program start',
            'D25': 'program start',
            'D29': 'program start',
            'D10': 'active lever',
            'D14': 'active lever',
            'D26': 'active lever',
            'D30': 'active lever',
            'D11': 'inactive lever',
            'D15': 'inactive lever',
            'D27': 'inactive lever',
            'D31': 'inactive lever',
            'D12': 'drug infusion',
            'D16': 'drug infusion',
            'D28': 'drug infusion',
            'D32': 'drug infusion',
        }
        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['D_code']]

        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['D_code'].values,
            'event': events
        }).reset_index(drop=True)
    
    return epoc_ts_and_indices, ts_epocs


   
    
# extract time points / indices for epoc event onsets ----------------------------------------------------------------------------------

# this version for the intermittent access self-admin session codes

    # updated 11.07.25 to accommodate doric data
    # updated 11.10.25 to allow for extraction from specified interval, given by 'segment_mode'
    
# documentation updated 07.22.26


def epoc_onsets_IntA(epoc_data, keepmin, ts, box, new_fs, dropsec, segment_mode, data_format, avail_period_sec = 300):
   
    """
    Aligns epoc onset timestamps with downsampled and trimmed stream data.

    Parameters:
        epoc_data (DataFrame): The original epoc dataset.
        segment_mode (str): One of ['before', 'during', 'after', 'transitions'].
        keepmin (float): Minutes of data to keep.
        ts (np.array): Time vector corresponding to trimmed stream data.
        box (str): Identifier for the box/subject.
        new_fs (float): New (downsampled) sampling rate.
        dropsec (float): Time in seconds of TTL signal.
        data_format (str): 'TDT' or 'Doric'.

    Returns:
        epoc_ts_and_indices (DataFrame): Epoc timestamps, stream indices, and labels.
        ts_epocs (np.array): Indices in trimmed time vector for each epoc.
    """
    
    # Determine start and end time of segment based on segment_mode
    if segment_mode is None:
        start_time = dropsec
        end_time = dropsec + keepmin * 60
    if segment_mode == 'before':
        start_time = dropsec - keepmin * 60
        end_time = dropsec
    elif segment_mode == 'during':
        start_time = dropsec
        end_time = dropsec + keepmin * 60
    elif segment_mode == 'after':
        start_time = dropsec + avail_period_sec  # Available period is always 5 min
        end_time = start_time + keepmin * 60  # Can be shorter than keepmin
    elif segment_mode == 'transitions':
        start_time = dropsec - 60  # 1 min preceding TTL
        end_time = dropsec + avail_period_sec + 60  # 1 min after available period
        transition_ts = dropsec + avail_period_sec  # timestamp of available → unavailable
    else:
        raise ValueError("segment_mode must be one of ['before', 'during', 'after', 'transitions']")

    # Ensure dropsec is scalar
    if isinstance(dropsec, (np.ndarray, list)):
        dropsec = float(dropsec[0])
        
    
    # Filter epoc data for relevant box
    if data_format == 'TDT':
        epoc_data_trimmed = epoc_data[epoc_data['name'] == box]
    elif data_format == 'Doric':
        epoc_data_trimmed = epoc_data[epoc_data['input code'].str.contains(box, case=False, na=False)] 
        # | epoc_data['input code'].str.contains(r'^D12\b', case=False, na=False)]   # only if issue with DIO codes  
    
    print(f'dropsec is equal to {dropsec}')

    # Trim epoc events to the kept stream interval
    epoc_data_trimmed = epoc_data_trimmed[
        (epoc_data_trimmed['onset'] >= start_time) &
        (epoc_data_trimmed['onset'] <= end_time)
    ]

    
    # Translate event codes to labels - MedPC programs after 10.01.25, coh5
    
    if data_format == 'TDT':
        
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['data', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - start_time

        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        # Translate event codes to labels
        '''
        event_map = {
            0: 'null',
            2: 'drug available onset',
            3: 'trial begin + drug available onset',
            4: 'unavailable onset',
            8: 'active lever',
            9: 'trial begin + active lever',
            10: 'available onset + active lever',
            12: 'unavailable onset + active lever',
            16: 'inactive lever',
            32: 'infusion',
            40: 'active lever + infusion',
            41: 'trial begin + active lever + infusion',
            42: 'available onset + active lever + infusion',
            64: 'trial end',
            72: 'trial end + active lever',
            104: 'trial end + active lever + infusion',
            128: 'active lever drug unavailable'
        }
        '''

        ## new map for updated MedPC programs, 10.10.25
        event_map = {
            0: 'null',
            1: 'trial begin',
            17: 'trial begin + drug available onset',
            16: 'drug available onset',
            2: 'active lever',
            10: 'active lever + infusion',
            8: 'infusion',
            4: 'inactive lever',
        }

        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['data']]
        
        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['data'].values,
            'event': events
        }).reset_index(drop=True)
        
        # Add transition timestamp for 'transitions' mode
        if segment_mode == 'transitions':
            transition_row = pd.DataFrame({
                'epoc_ts': [transition_ts - start_time],  # align to trimmed segment
                'epoc_indices': [np.clip(np.searchsorted(ts, transition_ts - start_time, side='right') - 1, 0, len(ts) - 1)],
                'data': [np.nan],  # no actual epoc code
                'event': ['available → unavailable']
            })

            epoc_ts_and_indices = pd.concat([epoc_ts_and_indices, transition_row], ignore_index=True)

        # Sort by event label first, then chronologically within each group
        epoc_ts_and_indices = epoc_ts_and_indices.sort_values(by=['event', 'epoc_ts']).reset_index(drop=True)
        
        
    elif data_format == 'Doric':
        
        epoc_data_trimmed['D_code'] = epoc_data_trimmed['input code'].str.extract(r'(D\d{2})')
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['D_code', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - start_time
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        event_map = {
            'D09': 'drug available',
            'D13': 'drug available',
            'D25': 'drug available',
            'D29': 'drug available',
            'D10': 'active lever',
            'D14': 'active lever',
            'D26': 'active lever',
            'D30': 'active lever',
            'D11': 'inactive lever',
            'D15': 'inactive lever',
            'D27': 'inactive lever',
            'D31': 'inactive lever',
            'D12': 'drug infusion',
            'D16': 'drug infusion',
            'D28': 'drug infusion',
            'D32': 'drug infusion',
        }
        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['D_code']]

        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['D_code'].values,
            'event': events
        }).reset_index(drop=True)
    
    
        # Add transition timestamp for 'transitions' mode
        if segment_mode == 'transitions':
            transition_row = pd.DataFrame({
                'epoc_ts': [transition_ts - start_time],  # align to trimmed segment
                'epoc_indices': [np.clip(np.searchsorted(ts, transition_ts - start_time, side='right') - 1, 0, len(ts) - 1)],
                'data': [np.nan],  # no actual epoc code
                'event': ['available_unavailable']
            })
            epoc_ts_and_indices = pd.concat([epoc_ts_and_indices, transition_row], ignore_index=True)

        # Sort by event label first, then chronologically within each group
        # Create a priority column: 0 for 'program start / drug available', 1 for everything else
        epoc_ts_and_indices['priority'] = epoc_ts_and_indices['event'].apply(
            lambda x: 0 if x == 'drug available' else 1
        )

        # Sort by priority, then event alphabetically, then timestamp
        epoc_ts_and_indices = epoc_ts_and_indices.sort_values(
            by=['priority', 'event', 'epoc_ts']
        ).drop(columns=['priority']).reset_index(drop=True)    

    return epoc_ts_and_indices, ts_epocs


   

# extract time points / indices for epoc event onsets ----------------------------------------------------------------------------------

# this version for the extinction training session codes

    ## updated 11.07.25 for Doric data and missed infusions
    
# updated documentation 07.22.26
    
def epoc_onsets_ext(epoc_data, keepmin, ts, box, new_fs, dropsec, data_format, refractory=4.0):
    """
    Aligns epoc onset timestamps with downsampled and trimmed stream data.

    Adds 'missed infusion' events for active lever presses that WOULD have resulted in a reward,
    based on a refractory period.

    Parameters:
        epoc_data (DataFrame): The original epoc dataset.
        keepmin (float): Minutes of data to keep.
        ts (np.array): Time vector corresponding to trimmed stream data.
        box (str): Identifier for the box/subject.
        new_fs (float): New (downsampled) sampling rate.
        dropsec (float): Time in seconds where trimmed stream begins.
        data_format (str): 'TDT' or 'Doric'.
        refractory (float): Refractory period for active lever -> reward in seconds.

    Returns:
        epoc_ts_and_indices (DataFrame): Epoc timestamps, stream indices, labels, including 'missed infusion'.
        ts_epocs (np.array): Indices in trimmed time vector for each epoc.
    """
    
    import numpy as np
    import pandas as pd
    
    if isinstance(dropsec, (np.ndarray, list)):
        dropsec = float(dropsec[0])
    
    # Filter epoc data for relevant box
    if data_format == 'TDT':
        epoc_data_trimmed = epoc_data[epoc_data['name'] == box]
    elif data_format == 'Doric':
        epoc_data_trimmed = epoc_data[epoc_data['input code'].str.contains(box, case=False, na=False)]
    
    start_time = dropsec
    end_time = dropsec + keepmin * 60
    epoc_data_trimmed = epoc_data_trimmed[
        (epoc_data_trimmed['onset'] >= start_time) & 
        (epoc_data_trimmed['onset'] <= end_time)
    ]
    
    # ---------------- TDT ---------------- #
    if data_format == 'TDT':
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['data','onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts)-1)
        
        
        # Translate event codes to labels
        '''
        event_map = {
            0: 'null',
            1: 'trial begin',
            2: 'active lever',
            4: 'inactive lever',
            8: 'missed infusion',
            16: 'trial end',
            10: 'active lever + missed infusion',
            11: 'trial begin + active lever + missed infusion',
            12: 'inactive lever + missed infusion',
            13: 'trial begin + inactive lever + missed infusion',
            26: 'trial end + active lever + missed infusion',
            28: 'trial end + inactive lever + missed infusion',
        }
        '''

        ## new map for updated MedPC programs, 10.10.25
        
        event_map = {
            0: 'null',
            1: 'trial begin',
            17: 'trial begin + drug available onset',
            16: 'drug available onset',
            2: 'active lever',
            10: 'active lever + infusion',
            8: 'infusion',
            4: 'inactive lever',
        }
        events = [event_map.get(val,'unknown') for val in epoc_data_sorted['data']]
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['data'].values,
            'event': events
        }).reset_index(drop=True)
        
        # ---------------- Missed Infusion ---------------- #
        # this is added to collect the timestamps associated with "missed infusions", which is a subset of active lever presses
        # active lever presses that occur in the absent 4-second cue period are excluded
        
        # Get all active lever timestamps
        active_times = epoc_ts_and_indices.loc[epoc_ts_and_indices['event'] == 'active lever', 'epoc_ts'].values
        
        missed = []
        last_reward_time = -np.inf
        for t in active_times:
            if t - last_reward_time >= refractory:
                missed.append(t)
                last_reward_time = t  # refractory starts here
        
        # Append missed infusion events
        if missed:
            missed_df = pd.DataFrame({
                'epoc_ts': missed,
                'epoc_indices': np.searchsorted(ts, np.array(missed), side='right') - 1,
                'data': [np.nan]*len(missed),
                'event': ['missed infusion']*len(missed)
            })
            epoc_ts_and_indices = pd.concat([epoc_ts_and_indices, missed_df], ignore_index=True)
            epoc_ts_and_indices = epoc_ts_and_indices.sort_values(by='epoc_ts').reset_index(drop=True)
        
    # ---------------- Doric ---------------- #
    elif data_format == 'Doric':
        epoc_data_trimmed['D_code'] = epoc_data_trimmed['input code'].str.extract(r'(D\d{2})')
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['D_code','onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts)-1)
        
        event_map = {
            'D09':'program start','D13':'program start',
            'D25':'program start','D29':'program start',
            'D10':'active lever','D14':'active lever','D26':'active lever','D30':'active lever',
            'D11':'inactive lever','D15':'inactive lever','D27':'inactive lever','D31':'inactive lever',
            'D12':'drug infusion','D16':'drug infusion','D28':'drug infusion','D32':'drug infusion'
        }
        events = [event_map.get(val,'unknown') for val in epoc_data_sorted['D_code']]
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['D_code'].values,
            'event': events
        }).reset_index(drop=True)
        
        # Missed infusion logic
        # this is added to collect the timestamps associated with "missed infusions", which is a subset of active lever presses
        # active lever presses that occur in the absent 4-second cue period are excluded
        # use .loc, a pandas method to slice dataframe --> df.loc[row_selector, column_selector]
        active_times = epoc_ts_and_indices.loc[epoc_ts_and_indices['event'] == 'active lever', 'epoc_ts'].values
        
        # create an empty list 
        missed = []
        
        # this needs to be negative infinity, not 0, otherwise presses occurring within 4 seconds of start would be rejected
        # this can be read as "no reward time has yet occurred" 
        last_reward_time = -np.inf 
        for t in active_times:
            if t - last_reward_time >= refractory:
                missed.append(t)
                last_reward_time = t
        
        if missed:
            missed_df = pd.DataFrame({
                'epoc_ts': missed,
                'epoc_indices': np.searchsorted(ts, np.array(missed), side='right') - 1,
                'data': [np.nan]*len(missed),
                'event': ['missed infusion']*len(missed)
            })
            epoc_ts_and_indices = pd.concat([epoc_ts_and_indices, missed_df], ignore_index=True)
            # Define event order: "program start" first, then others alphabetically
            event_order = ['program start'] + sorted(
                [e for e in epoc_ts_and_indices['event'].unique() if e != 'program start']
            )

            # Convert 'event' to categorical with this order
            epoc_ts_and_indices['event'] = pd.Categorical(
                epoc_ts_and_indices['event'],
                categories=event_order,
                ordered=True
            )

            # Sort by event first, then timestamp
            epoc_ts_and_indices = epoc_ts_and_indices.sort_values(
                by=['event', 'epoc_ts']
            ).reset_index(drop=True)
            
    return epoc_ts_and_indices, ts_epocs
    
    


# extract time points / indices for epoc event onsets ----------------------------------------------------------------------------------

# this version for the noncontingent cue presentations
## updated 11.07.25 for Doric data files

# reviewed 07.22.26

def epoc_onsets_noncont(epoc_data, keepmin, ts, box, new_fs, dropsec, data_format):
   
    """
    Aligns epoc onset timestamps with downsampled and trimmed stream data.

    Parameters:
        epoc_data (DataFrame): The original epoc dataset.
        keepmin (float): Minutes of data to keep.
        trange (tuple): Time window of interest around each event.
        ts (np.array): Time vector corresponding to trimmed stream data.
        box (str): Identifier for the box/subject.
        new_fs (float): New (downsampled) sampling rate.
        dropsec (float): Time in seconds where trimmed stream begins.

    Returns:
        epoc_ts_and_indices (DataFrame): Epoc timestamps, stream indices, and labels.
        ts_epocs (np.array): Indices in trimmed time vector for each epoc.
    """
    
    # Ensure dropsec is scalar
    if isinstance(dropsec, (np.ndarray, list)):
        dropsec = float(dropsec[0])
    
    # Filter epoc data for relevant box
    if data_format == 'TDT':
        epoc_data_trimmed = epoc_data[epoc_data['name'] == box]
    elif data_format == 'Doric':
        epoc_data_trimmed = epoc_data[epoc_data['input code'].str.contains(box, case=False, na=False)]
    
    print(f'dropsec is equal to {dropsec}')

    # Trim epoc events to the kept stream interval
    start_time = dropsec
    end_time = dropsec + keepmin * 60
    epoc_data_trimmed = epoc_data_trimmed[
        (epoc_data_trimmed['onset'] >= start_time) &
        (epoc_data_trimmed['onset'] <= end_time)
    ]

    
    # Translate event codes to labels - MedPC programs after 10.01.25, coh5
    
    if data_format == 'TDT':
        
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['data', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        
        # Find closest index in trimmed stream time vector
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        event_map = {
            0: 'null',
            1: 'trial begin',
            8: 'cue light',
            16: 'houselight',
        }

        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['data']]
        
        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['data'].values,
            'event': events
        }).reset_index(drop=True)
        
        
    elif data_format == 'Doric':
        
        epoc_data_trimmed = epoc_data_trimmed.copy()
        epoc_data_trimmed['D_code'] = epoc_data_trimmed['input code'].str.extract(r'(D\d{2})')
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['D_code', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        event_map = {
            'D09': 'program start',
            'D13': 'program start',
            'D25': 'program start',
            'D29': 'program start',
            'D10': 'cue light',
            'D14': 'cue light',
            'D26': 'cue light',
            'D30': 'cue light',
            'D11': 'houselight',
            'D15': 'houselight',
            'D27': 'houselight',
            'D31': 'houselight',
            'D12': '',
            'D16': '',
            'D28': '',
            'D32': '',
        }
        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['D_code']]

        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['D_code'].values,
            'event': events
        }).reset_index(drop=True)
    

    return epoc_ts_and_indices, ts_epocs



# extract time points / indices for epoc event onsets ----------------------------------------------------------------------------------

# this version for the progressive ratio test session codes

    ## updated 11.07.25 for Doric data

# reviewed 07.22.26
    
def epoc_onsets_PR(epoc_data, keepmin, ts, box, new_fs, dropsec, data_format):
   
    """
    Aligns epoc onset timestamps with downsampled and trimmed stream data.

    Parameters:
        epoc_data (DataFrame): The original epoc dataset.
        keepmin (float): Minutes of data to keep.
        trange (tuple): Time window of interest around each event.
        ts (np.array): Time vector corresponding to trimmed stream data.
        box (str): Identifier for the box/subject.
        new_fs (float): New (downsampled) sampling rate.
        dropsec (float): Time in seconds where trimmed stream begins.

    Returns:
        epoc_ts_and_indices (DataFrame): Epoc timestamps, stream indices, and labels.
        ts_epocs (np.array): Indices in trimmed time vector for each epoc.
    """
    
    # Ensure dropsec is scalar
    if isinstance(dropsec, (np.ndarray, list)):
        dropsec = float(dropsec[0])
    
    # Filter epoc data for relevant box
    if data_format == 'TDT':
        epoc_data_trimmed = epoc_data[epoc_data['name'] == box]
    elif data_format == 'Doric':
        epoc_data_trimmed = epoc_data[epoc_data['input code'].str.contains(box, case=False, na=False)]
    
    print(f'dropsec is equal to {dropsec}')

    # Trim epoc events to the kept stream interval
    start_time = dropsec
    end_time = dropsec + keepmin * 60
    epoc_data_trimmed = epoc_data_trimmed[
        (epoc_data_trimmed['onset'] >= start_time) &
        (epoc_data_trimmed['onset'] <= end_time)
    ]


    
    # Translate event codes to labels - MedPC programs after 10.01.25, coh5
    
    if data_format == 'TDT':
        
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['data', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        # Find closest index in trimmed stream time vector
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        # Translate event codes to labels
        '''
        event_map = {
            0: 'null',
            1: 'trial begin',
            8: 'active lever',
            16: 'inactive lever',
            32: 'infusion',
            64: 'trial end',
            40: 'active lever + infusion',
            41: 'trial begin + active lever + infusion',
            48: 'inactive lever + infusion',
            49: 'trial begin + inactive lever + infusion',
            104: 'trial end + active lever + infusion',
            112: 'trial end + inactive lever + infusion',
        }
        '''

        ## new map for updated MedPC programs, 10.10.25
        event_map = {
            0: 'null',
            1: 'trial begin',
            17: 'trial begin + drug available onset',
            16: 'drug available onset',
            2: 'active lever',
            10: 'active lever + infusion',
            8: 'infusion',
            4: 'inactive lever',
        }

        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['data']]
        
        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['data'].values,
            'event': events
        }).reset_index(drop=True)
        
        
    elif data_format == 'Doric':
        
        epoc_data_trimmed = epoc_data_trimmed.copy()
        epoc_data_trimmed['D_code'] = epoc_data_trimmed['input code'].str.extract(r'(D\d{2})')
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['D_code', 'onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts) - 1)

        event_map = {
            'D09': 'program start',
            'D13': 'program start',
            'D25': 'program start',
            'D29': 'program start',
            'D10': 'active lever',
            'D14': 'active lever',
            'D26': 'active lever',
            'D30': 'active lever',
            'D11': 'inactive lever',
            'D15': 'inactive lever',
            'D27': 'inactive lever',
            'D31': 'inactive lever',
            'D12': 'drug infusion',
            'D16': 'drug infusion',
            'D28': 'drug infusion',
            'D32': 'drug infusion',
        }
        
        events = [event_map.get(val, 'unknown') for val in epoc_data_sorted['D_code']]

        # Combine into a dataframe
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['D_code'].values,
            'event': events
        }).reset_index(drop=True)
    

    return epoc_ts_and_indices, ts_epocs


   


# extract time points / indices for epoc event onsets ----------------------------------------------------------------------------------
    # this version for cued reinstatement
 ## updated 11.07.25 for Doric data and missed infusions

# reviewed 07.22.26
    
    
def epoc_onsets_CR(epoc_data, keepmin, ts, box, new_fs, dropsec, data_format, refractory=4.0):
    """
    Aligns epoc onset timestamps with downsampled and trimmed stream data.

    Adds 'missed infusion' events for active lever presses that WOULD have resulted in a reward,
    based on a refractory period.

    Parameters:
        epoc_data (DataFrame): The original epoc dataset.
        keepmin (float): Minutes of data to keep.
        ts (np.array): Time vector corresponding to trimmed stream data.
        box (str): Identifier for the box/subject.
        new_fs (float): New (downsampled) sampling rate.
        dropsec (float): Time in seconds where trimmed stream begins.
        data_format (str): 'TDT' or 'Doric'.
        refractory (float): Refractory period for active lever -> reward in seconds.

    Returns:
        epoc_ts_and_indices (DataFrame): Epoc timestamps, stream indices, labels, including 'missed infusion'.
        ts_epocs (np.array): Indices in trimmed time vector for each epoc.
    """
    
    import numpy as np
    import pandas as pd
    
    if isinstance(dropsec, (np.ndarray, list)):
        dropsec = float(dropsec[0])
    
    # Filter epoc data for relevant box
    if data_format == 'TDT':
        epoc_data_trimmed = epoc_data[epoc_data['name'] == box]
    elif data_format == 'Doric':
        epoc_data_trimmed = epoc_data[epoc_data['input code'].str.contains(box, case=False, na=False)]
    
    start_time = dropsec
    end_time = dropsec + keepmin * 60
    epoc_data_trimmed = epoc_data_trimmed[
        (epoc_data_trimmed['onset'] >= start_time) & 
        (epoc_data_trimmed['onset'] <= end_time)
    ]
    
    # ---------------- TDT ---------------- #
    if data_format == 'TDT':
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['data','onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts)-1)
        
        # Translate event codes to labels
        '''
        event_map = {
            0: 'null',
            1: 'trial begin',
            8: 'active lever',
            16: 'inactive lever',
            32: 'missed infusion',
            64: 'trial end',
            40: 'active lever + missed infusion',
            41: 'trial begin + active lever + missed infusion',
            48: 'inactive lever + missed infusion',
            49: 'trial begin + inactive lever + missed infusion',
            104: 'trial end + active lever + missed infusion',
            112: 'trial end + inactive lever + missed infusion',
        }
        '''

        ## new map for updated MedPC programs, 10.10.25
        
        event_map = {
            0: 'null',
            1: 'trial begin',
            17: 'trial begin + drug available onset',
            16: 'drug available onset',
            2: 'active lever',
            10: 'active lever + infusion',
            8: 'infusion',
            4: 'inactive lever',
        }
        events = [event_map.get(val,'unknown') for val in epoc_data_sorted['data']]
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['data'].values,
            'event': events
        }).reset_index(drop=True)
        
        # ---------------- Missed Infusion ---------------- #
        # Get all active lever timestamps
        active_times = epoc_ts_and_indices.loc[epoc_ts_and_indices['event'] == 'active lever', 'epoc_ts'].values
        
        missed = []
        last_reward_time = -np.inf
        for t in active_times:
            if t - last_reward_time >= refractory:
                missed.append(t)
                last_reward_time = t  # refractory starts here
        
        # Append missed infusion events
        if missed:
            missed_df = pd.DataFrame({
                'epoc_ts': missed,
                'epoc_indices': np.searchsorted(ts, np.array(missed), side='right') - 1,
                'data': [np.nan]*len(missed),
                'event': ['missed infusion']*len(missed)
            })
            epoc_ts_and_indices = pd.concat([epoc_ts_and_indices, missed_df], ignore_index=True)
            epoc_ts_and_indices = epoc_ts_and_indices.sort_values(by='epoc_ts').reset_index(drop=True)
        
    # ---------------- Doric ---------------- #
    elif data_format == 'Doric':
        
        epoc_data_trimmed = epoc_data_trimmed.copy()
        epoc_data_trimmed['D_code'] = epoc_data_trimmed['input code'].str.extract(r'(D\d{2})')
        epoc_data_sorted = epoc_data_trimmed.sort_values(by=['D_code','onset'])
        epoc_data_sorted_shift = epoc_data_sorted['onset'] - dropsec
        ts_epocs = np.clip(np.searchsorted(ts, epoc_data_sorted_shift, side='right') - 1, 0, len(ts)-1)
        
        event_map = {
            'D09':'program start','D13':'program start',
            'D25':'program start','D29':'program start',
            'D10':'active lever','D14':'active lever','D26':'active lever','D30':'active lever',
            'D11':'inactive lever','D15':'inactive lever','D27':'inactive lever','D31':'inactive lever',
            'D12':'drug infusion','D16':'drug infusion','D28':'drug infusion','D32':'drug infusion'
        }
        events = [event_map.get(val,'unknown') for val in epoc_data_sorted['D_code']]
        
        epoc_ts_and_indices = pd.DataFrame({
            'epoc_ts': epoc_data_sorted_shift,
            'epoc_indices': ts_epocs,
            'data': epoc_data_sorted['D_code'].values,
            'event': events
        }).reset_index(drop=True)
        
        # Missed infusion logic
        active_times = epoc_ts_and_indices.loc[epoc_ts_and_indices['event'] == 'active lever', 'epoc_ts'].values
        
        missed = []
        last_reward_time = -np.inf
        for t in active_times:
            if t - last_reward_time >= refractory:
                missed.append(t)
                last_reward_time = t
        
        if missed:
            missed_df = pd.DataFrame({
                'epoc_ts': missed,
                'epoc_indices': np.searchsorted(ts, np.array(missed), side='right') - 1,
                'data': [np.nan]*len(missed),
                'event': ['missed infusion']*len(missed)
            })
            epoc_ts_and_indices = pd.concat([epoc_ts_and_indices, missed_df], ignore_index=True)
            # Define event order: "program start" first, then others alphabetically
            event_order = ['program start'] + sorted(
                [e for e in epoc_ts_and_indices['event'].unique() if e != 'program start']
            )

            # Convert 'event' to categorical with this order
            epoc_ts_and_indices['event'] = pd.Categorical(
                epoc_ts_and_indices['event'],
                categories=event_order,
                ordered=True
            )

            # Sort by event first, then timestamp
            epoc_ts_and_indices = epoc_ts_and_indices.sort_values(
                by=['event', 'epoc_ts']
            ).reset_index(drop=True)
            
    return epoc_ts_and_indices, ts_epocs
    
    



# drop a specified duration from the beginning of the recording, filter, re-plot ----------------------------------------------------------------------------------
    ## updated 06.23.25 to include event tickmarks
    ## updated 10.16.25 to fix time vector issues
    
# older version of the trimming, doesn't currently account for segment_mode options


def streams_trim_events_tdt(isos_405_filt_downsamp, GCaMP_465_filt_downsamp,
                            dropsec_streams, keepmin, ts_downsamp,
                            new_fs, y1, y2, x, startcode,
                            epoc_data=None, box=None, noncont=False):
    """
    Trims and aligns filtered 405 and 465 signals to a specified time window after trial start.
    Plots both untrimmed and trimmed versions, with optional event tick marks when epoc_data & box are provided.
    
    Returns:
        fig2a (Figure): Untrimmed streams figure.
        fig2b (Figure): Trimmed streams figure.
        isos_405 (np.array): Trimmed 405 signal.
        GCaMP_465 (np.array): Trimmed 465 signal.
        ts (np.array): Trimmed time vector (shifted to start at 0).
        fig2a_zoomed (Figure): Zoomed view of untrimmed streams.
    """

    print(f'The downsampled sampling rate is {new_fs} Hz\n')

    disc = int(new_fs * dropsec_streams)
    intvl = int(keepmin * 60 * new_fs)

    # Use searchsorted to find the index in ts_downsamp where time crosses dropsec_streams
    start = np.searchsorted(ts_downsamp, dropsec_streams)
    end = start + intvl

    print(f'Discard the first {dropsec_streams} seconds (start index: {start})')

    print(f'\nKeep a duration of {keepmin} minutes')

    if end > len(isos_405_filt_downsamp):
        end = len(isos_405_filt_downsamp)
        print(f'Including to end of stream: data points {start} to {end}')
    else:
        print(f'Included interval: data points {start} to {end}')

    # Defensive indexing
    if start >= len(ts_downsamp) or end <= start:
        raise ValueError(f"Invalid slice indices for trimming: start={start}, end={end}, array length={len(ts_downsamp)}")

    # Trim data
    isos_405 = isos_405_filt_downsamp[start:end]
    GCaMP_465 = GCaMP_465_filt_downsamp[start:end]
    ts = ts_downsamp[start:end]  # Slice
    ts = ts - dropsec_streams    # Shift to align with trimmed events (trial start = t=0)

    print(f'\nTrimmed duration: {ts[-1]:.2f} sec (~{ts[-1]/60:.2f} min)')
     
    if start >= len(ts_downsamp) or end <= start:
        raise ValueError(f"Invalid slice indices for trimming: start={start}, end={end}, array length={len(ts_downsamp)}")


    # --- Event Setup ---
    if noncont:
        event_labels = {
            16: 'houselight',
            17: 'trial begin + drug available onset',
            8: 'cue light',
        }
        event_colors = {
            16: 'red',
            8: 'green',
        }
    else:
        event_labels = {
            16: 'drug available onset',
            17: 'trial begin + drug available onset',
            2: 'active lever',
            4: 'inactive lever',
            8: 'infusion',
            10: 'active lever + infusion',
        }
        event_colors = {
            16: 'magenta', 17: 'magenta',
            2: 'yellow', 4: 'cyan',
            8: 'red', 10: 'orange'
        }

    if epoc_data is not None and box is not None:
        epoc_filtered = epoc_data[
            (epoc_data['name'] == box) &
            (epoc_data['data'].isin(event_labels.keys()))
        ]
    else:
        epoc_filtered = None

    def plot_event_ticks(ax, ts_range=None, trim_offset=0):
        handles, labels = [], []
        used = set()
        if epoc_filtered is None:
            return handles, labels
        for code, label in event_labels.items():
            times = epoc_filtered[epoc_filtered['data'] == code]['onset'].values
            times = times - trim_offset  # Align with trimmed ts
            for t in times:
                if ts_range is None or (ts_range[0] <= t <= ts_range[1]):
                    ax.axvline(t, color=event_colors.get(code, 'gray'), linestyle='--', linewidth=1)
            if len(times) and code not in used:
                h, = ax.plot([], [], color=event_colors.get(code, 'gray'), label=label)
                handles.append(h)
                labels.append(label)
                used.add(code)
        return handles, labels

    def plot_block(ts, sig1, sig2, title, xlim_range=None, span=None, ylim1=None, ylim2=None, trim_offset=0):
        fig, ax1 = plt.subplots()
        p1 = ax1.plot(ts, sig1, color=[0.7]*3, label='405')
        ax2 = ax1.twinx()
        p2 = ax2.plot(ts, sig2, color=[0.1, 0.7, 0.2], label='465')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('405 (mV)', color=[0.7]*3)
        ax2.set_ylabel('465 (mV)', color=[0.1, 0.7, 0.2])
        ax1.set_title(title)

        if xlim_range:
            ax1.set_xlim(xlim_range)
        if ylim1:
            ax1.set_ylim(ylim1)
        if ylim2:
            ax2.set_ylim(ylim2)

        if span:
            ax1.axvspan(*span, color='magenta', alpha=0.1, label='Kept Window')

        ev_h, ev_l = plot_event_ticks(ax1, ts_range=xlim_range, trim_offset=trim_offset)
        lines = p1 + p2 + ev_h
        labels = ['405', '465'] + ev_l
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        return fig

    # --- Plotting ---
    span = (dropsec_streams, dropsec_streams + keepmin * 60)
    fig2a = plot_block(ts_downsamp, isos_405_filt_downsamp, GCaMP_465_filt_downsamp,
                       'Untrimmed streams (trial begin)', xlim_range=None, span=span)

    zoom = (dropsec_streams - 120, dropsec_streams + keepmin * 60 + 120)
    fig2a_zoomed = plot_block(ts_downsamp, isos_405_filt_downsamp, GCaMP_465_filt_downsamp,
                              'Zoomed ±2 min around kept window', xlim_range=zoom, span=span)

    fig2b = plot_block(
        ts, isos_405, GCaMP_465,
        'Trimmed streams',
        xlim_range=(0, ts[-1]),
        ylim1=(np.min(isos_405) * 0.8, np.max(isos_405) * 1.5),
        ylim2=(np.min(GCaMP_465) * 0.8, np.max(GCaMP_465) * 1.5),
        trim_offset=dropsec_streams
    )

    return (fig2a, fig2b, isos_405, GCaMP_465, ts, fig2a_zoomed)





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# updated documentation 07.31.26

def visualize_and_correct_artifacts(
    epoc_ts_and_indices,
    new_fs,
    ts,
    GCaMP_465,
    isos_405,
    correct_stream='both',              # Options: 'isos', 'gcamp', 'both', or 'none'
    segments_to_correct=None,           # list of segment indices to correct (1-based)
    baseline_window_sec=2.0,
    artifact_duration_sec=4.0,          # customizable artifact duration
    buffer_sec=0.15,
    interpolation_ranges_sec=None,      # list of tuples, e.g., [(-0.2, 0.2), (3.8, 4.2)]
    interpolation_position='both',     # 'onset', 'offset', or 'both'
    correction_mode='both'       # options: 'offset', 'interpolate', 'both', 'none'
 
):
    """
    Correct artifacts using median-based, clamped subtraction to prevent post-correction overshoot.
    Supports optional interpolation at onset, offset, or both.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import sem

    # Early exit if epoc_ts_and_indices is empty
    if epoc_ts_and_indices is None or epoc_ts_and_indices.empty:
        print("epoc_ts_and_indices is empty → skipping artifact correction.")
        return GCaMP_465.copy(), isos_405.copy()

    # use assert statements to check inputs
    assert correct_stream in ['isos', 'gcamp', 'both', 'none'], "Invalid option for correct_stream"
    assert interpolation_position in ['onset', 'offset', 'both'], "Invalid option for interpolation_position"
    assert correction_mode in ['offset', 'interpolate', 'both', 'none'], "correction_mode must be 'offset', 'interpolate', 'both', or 'none'"

    # create the dataframe of events that include cue light illumination (potential light artifacts)
    drug_infusions = epoc_ts_and_indices[
        epoc_ts_and_indices['event'].str.contains('infusion|cue light', case=False)
    ]
    
    if drug_infusions.empty:
        print("No 'drug infusion' events found.")
        return GCaMP_465.copy(), isos_405.copy()

    # convert windows to number of samples from seconds
    baseline_pts = int(new_fs * baseline_window_sec)
    artifact_pts = int(new_fs * artifact_duration_sec)
    buffer_pts = int(new_fs * buffer_sec)

    overlay_pre = int(new_fs * 4)
    overlay_post = int(new_fs * 8)
    overlay_total = overlay_pre + overlay_post
    overlay_time = np.linspace(-4, 8, overlay_total)

    gcamp_segments = []
    isos_segments = []
    GCaMP_corrected = GCaMP_465.copy()
    isos_corrected = isos_405.copy()

    # map(int, iterable) applies a function to each element of the iterable, in this case converting each epoc_indices element to an integer
    # then wrap in a list 
    onset_indices = list(map(int, drug_infusions['epoc_indices']))
    print(f"Detected {len(onset_indices)} artifact segments.")

    # if a subset of segments isn't specified, correct all of them
    if segments_to_correct is None:
        segments_to_correct = list(range(1, len(onset_indices) + 1))
    else:
        print(f"Correcting only segments: {segments_to_correct}")

    # start looping through each event
    for seg_idx, onset in enumerate(onset_indices, start=1):
        seg_start = onset - overlay_pre
        seg_end = onset + overlay_post

        if seg_start < 0 or seg_end >= len(ts):
            print(f"Skipping overlay segment #{seg_idx}: out of bounds.")
            continue

        # create lists of numpy arrays, each one corresponding to infusion / cue light events
        gcamp_segments.append(GCaMP_465[seg_start:seg_end])
        isos_segments.append(isos_405[seg_start:seg_end])

        if correct_stream == 'none' or seg_idx not in segments_to_correct:
            continue

        # --- Baseline and artifact windows ---
        # use the various windows to get the index start and stop for baseline, artifact
        baseline_start = max(onset - baseline_pts - buffer_pts, 0)
        baseline_end = onset - buffer_pts

        artifact_core_start = onset
        artifact_core_end = min(onset + artifact_pts, len(ts))

        # ======================================================
        # 1. MEDIAN-BASED OFFSET SUBTRACTION  (if enabled)
        # ======================================================
        if correction_mode in ['offset', 'both']:

            # offset is the difference between the pre-event baseline and the event window
            # median, rather than mean, less sensitive to noise / spikes in amplitude
            # Median-based offset
            baseline_median_gcamp = np.median(GCaMP_465[baseline_start:baseline_end])
            artifact_median_gcamp = np.median(GCaMP_465[artifact_core_start:artifact_core_end])
            # offset should not be negative - artifacts are appearing as almost square-pulse increases in amplitude
            offset_gcamp = max(0, artifact_median_gcamp - baseline_median_gcamp)

            baseline_median_isos = np.median(isos_405[baseline_start:baseline_end])
            artifact_median_isos = np.median(isos_405[artifact_core_start:artifact_core_end])
            offset_isos = max(0, artifact_median_isos - baseline_median_isos)

            # Apply offset across artifact region
            if correct_stream in ['gcamp', 'both']:
                GCaMP_corrected[artifact_core_start:artifact_core_end] -= offset_gcamp
            if correct_stream in ['isos', 'both']:
                isos_corrected[artifact_core_start:artifact_core_end] -= offset_isos


        # ======================================================
        # 2. OPTIONAL INTERPOLATION  (if enabled)
        # ======================================================
        if correction_mode in ['interpolate', 'both'] and interpolation_ranges_sec:

            # loop through the list of tuples
            for start_sec, end_sec in interpolation_ranges_sec:
                interp_start = int(onset + start_sec * new_fs)
                interp_end   = int(onset + end_sec * new_fs)

                # ---- Onset interpolation (negative range) ----
                # use np.linspace to create the interpolations
                if interpolation_position in ['onset', 'both'] and start_sec < 0:
                    if correct_stream in ['gcamp', 'both']:
                        y0 = GCaMP_465[interp_start]
                        y1 = GCaMP_corrected[interp_end] if correction_mode != 'interpolate' else GCaMP_465[interp_end]
                        GCaMP_corrected[interp_start:interp_end] = np.linspace(y0, y1, interp_end - interp_start)

                    if correct_stream in ['isos', 'both']:
                        y0 = isos_405[interp_start]
                        y1 = isos_corrected[interp_end] if correction_mode != 'interpolate' else isos_405[interp_end]
                        isos_corrected[interp_start:interp_end] = np.linspace(y0, y1, interp_end - interp_start)

                # ---- Offset interpolation (positive range) ----
                if interpolation_position in ['offset', 'both'] and start_sec > 0:
                    if correct_stream in ['gcamp', 'both']:
                        y0 = GCaMP_corrected[interp_start]
                        y1 = GCaMP_465[interp_end]
                        GCaMP_corrected[interp_start:interp_end] = np.linspace(y0, y1, interp_end - interp_start)

                    if correct_stream in ['isos', 'both']:
                        y0 = isos_corrected[interp_start]
                        y1 = isos_405[interp_end]
                        isos_corrected[interp_start:interp_end] = np.linspace(y0, y1, interp_end - interp_start)

    
    # ---------------------------------------------------------------------
    # UPDATED legend-smart plotting function (multi-column legend)
    # ---------------------------------------------------------------------
    def plot_segments_with_legend(segments, overlay_time, colors, title, artifact_duration_sec):

        if not segments:
            return

        # use np.vstack to collect the event segment arrays into a matrix, where each row is one event
        stack = np.vstack(segments)
        # mean and sem across rows
        mean_seg = np.mean(stack, axis=0)
        sem_seg = sem(stack, axis=0)

        plt.figure(figsize=(12, 6))
        lines = []
        labels = []

        for i, seg in enumerate(segments):
            # 'line,' with the comma is a shorthand that returns item [0] of the list produced by plt.plot
            # [i % len(colors)] involves the modulo operator - it is a "remainder after division", meaning colors can repeat if there were fewer colors in the list than segments. 
            line, = plt.plot(overlay_time, seg, color=colors[i % len(colors)], alpha=0.4)
            lines.append(line)
            labels.append(f"Segment {i+1}")

        # Mean line
        mean_line, = plt.plot(overlay_time, mean_seg, color=[1, 1, 0], linewidth=3)
        lines.append(mean_line)
        labels.append("Mean")

        plt.fill_between(overlay_time, mean_seg - sem_seg, mean_seg + sem_seg,
                         color=[0.7, 0.7, 0.7], alpha=0.3)

        plt.axvspan(0, artifact_duration_sec, color=[1, 0, 1], alpha=0.1)

        plt.title(title)
        plt.xlabel("Time (s) rel. to artifact onset")
        plt.ylabel("Signal amplitude")

        # ------- MULTI-COLUMN LEGEND BELOW PLOT -------
        # given number of events in some recordings, this is necessary to prevent a massive vertical legend. 
        max_labels_per_col = 10
        n_labels = len(labels)
        ncols = math.ceil(n_labels / max_labels_per_col)

        # Create figure and axis first
        # assign names to current figure and axes (plt.get current fig, axes)
        fig = plt.gcf()
        ax = plt.gca()

        # Make space for the legend without shrinking the plot too much
        fig.subplots_adjust(bottom=0.22)  # increase if legend is large

        legend = fig.legend(
            lines,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),   # centered horizontally
            ncol=ncols,
            columnspacing=1.0,
            handletextpad=0.5,
        )

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------
    # this essentially interpolates between the starting and ending colors with evenly spaced steps
    '''
    def gradient_colors(start_rgb, end_rgb, n):
        colors = []
        for i in range(n):
            factor = i / max(1, n - 1)
            color = []
            for j in range(3):
                value = start_rgb[j] + (end_rgb[j] - start_rgb[j]) * factor
                color.append(value)
            colors.append(color)
        return colors
    '''
    # a more compact version
    def gradient_colors(start_rgb, end_rgb, n):
        return [
            [start_rgb[j] + (end_rgb[j] - start_rgb[j]) * (i / max(1, n - 1)) for j in range(3)]
            for i in range(n)
        ]

    gcamp_colors = gradient_colors([0, 0, 1], [0, 1, 0], len(gcamp_segments))
    isos_colors = gradient_colors([1, 0, 0], [1, 0, 1], len(isos_segments))

    # --- Plot Original Segments ---
    plot_segments_with_legend(gcamp_segments, overlay_time, gcamp_colors, "GCaMP 465 - Original Segments", artifact_duration_sec)
    plot_segments_with_legend(isos_segments, overlay_time, isos_colors, "Isosbestic 405 - Original Segments", artifact_duration_sec)


    # --- Plot corrected segments (selected only) ---
    corr_gcamp_segments = []
    corr_isos_segments = []
    for seg_idx, onset in enumerate(onset_indices, start=1):
        if seg_idx not in segments_to_correct:
            continue
        seg_start = onset - overlay_pre
        seg_end = onset + overlay_post
        if seg_start >= 0 and seg_end < len(ts):
            corr_gcamp_segments.append(GCaMP_corrected[seg_start:seg_end])
            corr_isos_segments.append(isos_corrected[seg_start:seg_end])

    # Plot corrected segments if any
    n_corr = len(corr_gcamp_segments)
    if n_corr > 0:
        if correct_stream in ['gcamp', 'both']:
            plot_segments_with_legend(corr_gcamp_segments, overlay_time, gcamp_colors[:n_corr],
                                      "GCaMP 465 - Corrected Segments (Selected)", artifact_duration_sec)
        if correct_stream in ['isos', 'both']:
            plot_segments_with_legend(corr_isos_segments, overlay_time, isos_colors[:n_corr],
                                      "Isosbestic 405 - Corrected Segments (Selected)", artifact_duration_sec)

    return GCaMP_corrected, isos_corrected





# motion correction using a linear fit between isosbestic and GCaMP streams ------------------------------------------------------------------------------------------------------------
# documentation updated 07.31.26

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.stats import linregress

def motcorr_unified(
    ts,
    isos,
    gcamp,
    tzoom,
    method="linregress",
    clamp_negative=True,
    r2_threshold=0.05,
    window_sec=3,
    fs=None,
    plot=True
):
    """
    Unified motion correction framework supporting:
    - Linear regression
    - Polyfit (degree 1)
    """

    figs = []

    # Infer sampling rate if not supplied
    if fs is None:
        fs = 1 / np.median(np.diff(ts))

    # ================================
    # METHOD 1: linregress
    # ================================
    # this is a method based off Simpson et al., 2024, who also use linregress
    # clamp_negative was added because if the isos and GCaMP signals are anti-correlated (very weak motion artifact / shared dynamics),
    # the estiamted motion should not be subtracted because it will actually add artifact
    
    if method == "linregress":
        slope, intercept, r_value, _, _ = linregress(isos, gcamp)
        r2 = r_value**2

        if clamp_negative and slope < 0:
            slope = 0
            intercept = 0
            print('negative regression slope, correction not applied')

        if r2 < r2_threshold:
            est_motion = None
        else:
            est_motion = slope * isos + intercept

    # ================================
    # METHOD 2: polyfit
    # ================================
    # this is a method based off Charles Zhou, Adam Gordon-Fennell and others at UW
    
    elif method == "polyfit":
        p = np.polyfit(isos, gcamp, 1)
        slope = p[0]
        intercept = p[1]

        # compute R²
        gcamp_hat = slope * isos + intercept
        ss_res = np.sum((gcamp - gcamp_hat)**2)
        ss_tot = np.sum((gcamp - np.mean(gcamp))**2)
        r2 = 1 - ss_res/ss_tot

        if clamp_negative and slope < 0:
            slope = 0
            intercept = 0

        if r2 < r2_threshold:
            est_motion = np.zeros_like(isos)
        else:
            est_motion = slope * isos + intercept

    else:
        raise ValueError("method must be 'linregress' or 'polyfit'")

    # ================================
    # APPLY CORRECTION
    # ================================
    if est_motion is None:
        gcamp_corrected = gcamp.copy()
    else:
        gcamp_corrected = gcamp - est_motion
    

    # ================================
    # PLOT
    # ================================
    if plot:
        # scatter plot (for regression methods)
        if method in ["linregress", "polyfit"]:
            fig1 = plt.figure(figsize=(6,6))
            plt.scatter(isos[::5], gcamp[::5], alpha=0.3, marker='.')
            x = np.linspace(np.min(isos), np.max(isos), 200)
            plt.plot(x, slope * x + intercept, color='red')
            plt.title(f"{method}: slope={slope:.4f}, R²={r2:.4f}")
            figs.append(fig1)

        # time plot
        '''
        
        fig2, ax = plt.subplots(figsize=(12,5))
        ax.plot(ts, gcamp, color='magenta', label="GCaMP original")
        ax.plot(ts, isos, color=[0.7,0.7,0.7], label="isos (405)", linewidth=3)
        
        if est_motion is not None:
            ax.plot(ts, est_motion, color='yellow', label="Estimated motion")
        else:
            ax.text(
                0.01, 0.95,
                "Motion correction skipped (low R²)",
                transform=ax.transAxes,
                color='red',
                fontsize=10,
                verticalalignment='top'
            )
        #ax.plot(ts, gcamp_corrected, color=[0.1,0.7,0.2], label="GCaMP corrected")

        ax.set_xlim(tzoom)
        ax.legend()
        ax.set_title(f"Motion Correction ({method})")
        figs.append(fig2)
        
        '''
        
        # try update
        # time plot with separate y-axis for isos
        fig2, ax1 = plt.subplots(figsize=(12,5))

        # Primary axis: GCaMP and estimated motion
        ax1.plot(
            ts,
            gcamp,
            color='magenta',
            label="GCaMP original",
            zorder = 2
        )

        if est_motion is not None:
            ax1.plot(
                ts,
                est_motion,
                color='yellow',
                label="Estimated motion",
                zorder=3
            )
        else:
            ax1.text(
                0.01,
                0.95,
                "Motion correction skipped (low R²)",
                transform=ax1.transAxes,
                color='red',
                fontsize=10,
                verticalalignment='top'
            )

        ax1.set_ylabel("GCaMP signal", color='magenta')
        ax1.tick_params(axis='y', labelcolor='magenta')


        # Secondary axis: isos
        ax2 = ax1.twinx()
        ax2.set_zorder(0)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        ax2.plot(
            ts,
            isos,
            color=[0.7,0.7,0.7],
            label="isos (405)",
            zorder = 1
        )

        ax2.set_ylabel("isos (405)", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')


        # Shared formatting
        ax1.set_xlim(tzoom)
        ax1.set_xlabel("Time (s)")
        ax1.set_title(f"Motion Correction ({method})")


        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right"
        )

        figs.append(fig2)
        
        
        
        # time plot
        fig3, ax = plt.subplots(figsize=(12,5))
        ax.plot(ts, gcamp_corrected, color=[0.1,0.7,0.2], label="GCaMP corrected")
        ax.set_xlim(tzoom)
        ax.legend()
        ax.set_title(f"Motion Correction ({method})")
        figs.append(fig3)


    return gcamp_corrected, est_motion, slope, intercept, r2, figs



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# updated documentation 08.07.26


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import math

def plot_perievent_segments(
    epoc_ts_and_indices,
    new_fs,
    ts,
    GCaMP_465,
    isos_405_nocuearts,
    est_motion,
    GCaMP_465_motcorr,
    event_filter=None,
    pre_window_sec=4.0,
    post_window_sec=8.0,
    event_col="event",       
    index_col="epoc_indices"  
):
    """
    Plot peri-event segments for GCaMP, isosbestic, motion, and motion-corrected signals.
    This version mimics the old function exactly and avoids KeyErrors.
    """

    # -----------------------------------
    # check that inputs are available
    # -----------------------------------
    if epoc_ts_and_indices is None or epoc_ts_and_indices.empty:
        print("No events to plot (empty DataFrame).")
        return

    if event_col not in epoc_ts_and_indices.columns or index_col not in epoc_ts_and_indices.columns:
        print(f"DataFrame missing required columns: {event_col}, {index_col}")
        print("Available columns:", epoc_ts_and_indices.columns)
        return

    # -----------------------------------
    # Select events
    # -----------------------------------
    # the str.contains is boolean indexing, filtering the epoc_ts_and_indices dataframe to only the rows where the event matches the event_filter
    matched_events = epoc_ts_and_indices[
        epoc_ts_and_indices[event_col].str.contains(event_filter, case=False)
    ]

    if matched_events.empty:
        print(f"No events found matching filter: {event_filter}")
        return

    # map(function, iterable) converts the pandas series of indices to integers, then that becomes a list
    onset_indices = list(map(int, matched_events[index_col]))
    print(f"Detected {len(onset_indices)} perievent segments.")

    # -----------------------------------
    # Windowing
    # -----------------------------------
    pre_pts = int(pre_window_sec * new_fs)
    post_pts = int(post_window_sec * new_fs)
    seg_len = pre_pts + post_pts
    overlay_time = np.linspace(-pre_window_sec, post_window_sec, seg_len)

    # -----------------------------------
    # Segment extraction helper
    # -----------------------------------
    def extract_segments(signal):
        if signal is None:
            return None
        segments = []
        for onset in onset_indices:
            start = onset - pre_pts
            end = onset + post_pts
            if start < 0 or end >= len(signal):
                continue
            segments.append(signal[start:end])
        return segments if segments else None

    gcamp_segments = extract_segments(GCaMP_465)
    isos_segments = extract_segments(isos_405_nocuearts)
    motion_segments = extract_segments(est_motion) if est_motion is not None else None
    motcorr_segments = extract_segments(GCaMP_465_motcorr)

    # -----------------------------------
    # Gradient color helper
    # -----------------------------------
    def gradient_colors(start_rgb, end_rgb, n):
        if n <= 1:
            # If only one color, just return the start color
            return [start_rgb]
        return [
            [start_rgb[j] + (end_rgb[j] - start_rgb[j]) * (i / (n - 1)) for j in range(3)]
            for i in range(n)
        ]


    # -----------------------------------
    # Plotting helper
    # -----------------------------------
    def plot_segments_with_legend(segments, overlay_time, colors, title):
        if not segments:
            print(f"No segments to plot for: {title}")
            return

        stack = np.vstack(segments)
        mean_seg = np.mean(stack, axis=0)
        sem_seg = sem(stack, axis=0)

        plt.figure(figsize=(10, 5))

        # Individual segments
        for i, seg in enumerate(segments):
            plt.plot(overlay_time, seg, color=colors[i % len(colors)], alpha=0.4)

        # Mean line
        plt.plot(overlay_time, mean_seg, color=[1, 1, 0], linewidth=3, label="Mean")

        # SEM shading
        plt.fill_between(overlay_time, mean_seg - sem_seg, mean_seg + sem_seg, color=[0.7, 0.7, 0.7], alpha=0.3)

        # Highlight 0–4 sec interval
        plt.axvspan(0, 4, color=[1, 0, 1], alpha=0.1)

        plt.title(title)
        plt.xlabel("Time (s) rel. to event onset")
        plt.ylabel("Signal amplitude")
        plt.tight_layout()
        plt.show()

    # -----------------------------------
    # Colors
    # -----------------------------------
    gcamp_colors = gradient_colors([0, 0, 1], [0, 1, 0], len(gcamp_segments) if gcamp_segments else 1)
    isos_colors = gradient_colors([1, 0, 0], [1, 0, 1], len(isos_segments) if isos_segments else 1)
    motion_colors = gradient_colors([0.2, 0.2, 0.2], [0.9, 0.9, 0.9], len(motion_segments) if motion_segments else 1)
    motcorr_colors = gradient_colors([0, 0.5, 1], [0, 1, 1], len(motcorr_segments) if motcorr_segments else 1)

    # -----------------------------------
    # Generate plots
    # -----------------------------------
    plot_segments_with_legend(gcamp_segments, overlay_time, gcamp_colors, "GCaMP 465 - Perievent Segments")
    plot_segments_with_legend(isos_segments, overlay_time, isos_colors, "Isosbestic 405 - Perievent Segments")
    plot_segments_with_legend(motion_segments, overlay_time, motion_colors, "Estimated Motion - Perievent Segments")
    plot_segments_with_legend(motcorr_segments, overlay_time, motcorr_colors, "GCaMP 465 (Motion-Corrected) - Perievent Segments")

    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# correct for bleaching with alternate method: subtraction of double expontential fit ------------------------------------------------------------------------------------------------------------

# documentation updated 08.07.26
# this is essentially the double exponential correction method used by Simpson et al., 2024



def dubexp_correct(ts, isos_405, GCaMP_465, tzoom, downsample_factor=10):
    """
    Double exponential correction of GCaMP_465 and isos_405 signals.
    Curve fitting is done on downsampled data for speed; outputs are full-resolution.

    Parameters:
        ts                  : time vector
        isos_405            : isosbestic signal
        GCaMP_465           : GCaMP signal
        tzoom               : time range for plotting (tuple)
        downsample_factor   : integer, reduce number of points for fitting

    Returns:
        isos_405_expfit, isos_405_expcorr, GCaMP_465_expfit, GCaMP_465_expcorr, fig_9, fig_10
    """

    # --- Double exponential model ---
    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
        tau_fast = tau_slow * tau_multiplier
        return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)

    # --- downsample signals for faster fitting ---
    # downsample by a factor of 10 by default

    idx = np.arange(0, len(ts), downsample_factor)
    ts_ds = ts[idx] - ts[idx][0]
    GCaMP_465_ds = GCaMP_465[idx]
    isos_405_ds = isos_405[idx]
    ts_ds0 = ts_ds - ts_ds[0]
    ts0 = ts - ts[0]

    # --- Fit double exponential to GCaMP ---
    # here output the parameteres using the downsampled vectors
    # curve_fit is from scipy, it identifies the values of amp, tau, tau_multi, and const that best match the data
    max_sig = np.max(GCaMP_465_ds)
    initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])

    GCaMP_465_parms, _ = curve_fit(double_exponential, ts_ds0, GCaMP_465_ds,
                                   p0=initial_params, bounds=bounds, maxfev=10000)
    
    # reconstruct full-resolution fit
    # now apply to the non-downsampled data
    GCaMP_465_expfit = double_exponential(ts0, *GCaMP_465_parms)

    # repeat with isos
    # --- Fit double exponential to isos_405 ---
    max_sig = np.max(isos_405_ds)
    initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    isos_405_parms, _ = curve_fit(double_exponential, ts_ds0, isos_405_ds,
                                  p0=initial_params, bounds=bounds, maxfev=10000)
    isos_405_expfit = double_exponential(ts0, *isos_405_parms)

    # --- Plot original signals with fits ---
    fig_9, ax1 = plt.subplots()
    plot1 = ax1.plot(ts, GCaMP_465, color=[0.1,0.7,0.2], label='GCaMP_465')
    plot3 = ax1.plot(ts, GCaMP_465_expfit, color=[1,0,1], linewidth=1.5, label='Exponential fit')
    ax2 = ax1.twinx()
    plot2 = ax2.plot(ts, isos_405, color=[0.7,0.7,0.7], label='isos_405')
    plot4 = ax2.plot(ts, isos_405_expfit, color=[1,0,1], linewidth=1.5)

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GCaMP_465 Signal (V)', color='r')
    ax2.set_ylabel('isos_405 Signal (V)', color='b')
    ax1.set_title('Filtered signals with double exponential fits')

    GCaMP_ylim = [np.min(GCaMP_465)*0.75, 1.25*np.max(GCaMP_465)]
    isos_ylim = [np.min(isos_405)*0.75, 1.25*np.max(isos_405)]
    lines = plot1 + plot2 + plot3
    labels = [l.get_label() for l in lines] # list comprehension
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_ylim(GCaMP_ylim)
    ax2.set_ylim(isos_ylim)

    # --- Subtract exponential fits ---
    GCaMP_465_expcorr = GCaMP_465 - GCaMP_465_expfit
    isos_405_expcorr = isos_405 - isos_405_expfit

    # --- Plot corrected signals ---
    fig_10, ax1 = plt.subplots()
    plot1 = ax1.plot(ts, GCaMP_465_expcorr, color=[0.1,0.7,0.2], label='GCaMP_465_expcorr')
    ax2 = ax1.twinx()
    plot2 = ax2.plot(ts, isos_405_expcorr, color=[0.7,0.7,0.7], label='isos_405_expcorr')

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GCaMP_465 (mV)', color='r')
    ax2.set_ylabel('isos_405 (mV)', color='b')
    ax1.set_title('Bleaching Correction by Double Exponential Fit')

    GCaMP_ylim = [np.min(GCaMP_465_expcorr)*4, 4*np.max(GCaMP_465_expcorr)]
    isos_ylim = [np.min(isos_405_expcorr)*4, 4*np.max(isos_405_expcorr)]
    lines = plot1 + plot2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_ylim(GCaMP_ylim)
    ax2.set_ylim(isos_ylim)
    plt.xlim(tzoom)

    return isos_405_expfit, isos_405_expcorr, GCaMP_465_expfit, GCaMP_465_expcorr, fig_9, fig_10





    
# correct for bleaching with alternate method: subtraction of a baseline calculated using an adaptive iteratively reweighted Penalized Least Squares (airPLS) algorithm  ------------------------------------------------------------------------------------------------------------

# this is the code referenced in O'Neal et al., 2022, see his methods for reference 

# documentation updated 08.07.26

# WhittakerSmooth() = fits a smooth curve to data according to given weights
# airPLS() = iteratively adjusts those weights to distinguish baseline from peaks in the data
# baselinecorr() = applies airPLS correction to signals
    
'''
airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares

This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls

Reference:
Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively 
reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

Description from the original documentation:
Baseline drift always blurs or even swamps signals and deteriorates analytical 
results, particularly in multivariate analysis.  It is necessary to correct baseline 
drift to perform further data analysis. Simple or modified polynomial fitting has 
been found to be effective in some extent. However, this method requires user 
intervention and prone to variability especially in low signal-to-noise ratio 
environments. The proposed adaptive iteratively reweighted Penalized Least Squares
(airPLS) algorithm doesn't require any user intervention and prior information, 
such as detected peaks. It iteratively changes weights of sum squares errors (SSE) 
between the fitted baseline and original signals, and the weights of SSE are obtained 
adaptively using between previously fitted baseline and original signals. This 
baseline estimator is general, fast and flexible in fitting baseline.


LICENCE
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see 
'''



def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_= 100, porder= 1, itermax= 15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
        
    return z


def baselinecorr(ts, isos_405, isos_405_expcorr, GCaMP_465, GCaMP_465_motcorr_expcorr, tzoom, lambda_=100, porder=1, itermax=15):

    
    # --- adjust motion-corrected and exponential fit corrected data so airPLS weighting works correctly ---
    
    isos_405_scaled = isos_405_expcorr + np.mean(isos_405)
    GCaMP_465_scaled = GCaMP_465_motcorr_expcorr + np.mean(GCaMP_465)
    
   
    # --- Calculate airPLS baseline ---
    isos_405_baseline = airPLS(isos_405_scaled, lambda_=lambda_, porder=porder, itermax=itermax) 
    GCaMP_465_baseline = airPLS(GCaMP_465_scaled, lambda_=lambda_, porder=porder, itermax=itermax) 

    
    # --- Subtract baseline ---
    isos_405_airPLS = isos_405_scaled - isos_405_baseline - np.mean(isos_405)
    GCaMP_465_airPLS = GCaMP_465_scaled - GCaMP_465_baseline - np.mean(GCaMP_465)

    # --- Center signals at zero ---
    isos_405_airPLS_centered = isos_405_airPLS - np.mean(isos_405_airPLS)
    GCaMP_465_airPLS_centered = GCaMP_465_airPLS - np.mean(GCaMP_465_airPLS)
    

    # --- Plot raw streams and calculated baselines ---
    fig_a = plt.figure(figsize=(10,7))
    ax1 = fig_a.add_subplot(211)
    ax1.plot(ts, isos_405_scaled, color=[0.7,0.7,0.7], label='isos_405', linewidth=2)
    ax1.plot(ts, isos_405_baseline, color=[1,0,1], label='isos_405_baseline', linewidth=2)
    ax1.set_ylabel('405 fluor. (mV)', color=[0.7,0.7,0.7])
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98,0.98))

    ax2 = fig_a.add_subplot(212)
    ax2.plot(ts, GCaMP_465_scaled, color=[0.1,0.7,0.2], label='GCaMP_465', linewidth=2)
    ax2.plot(ts, GCaMP_465_baseline, color=[1,0,1], label='GCaMP_465_baseline', linewidth=2)
    ax2.set_ylabel('465 fluor. (mV)', color=[0.1,0.7,0.2])
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98,0.98))

    ax2.set_xlabel('Time (seconds)')
    ax1.set_title('airPLS baseline calculation')

    # --- Plot baseline-corrected, centered signals ---
    fig_c, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(ts, GCaMP_465_airPLS_centered, color=[0.1,0.7,0.2], label='GCaMP_465_airPLS_centered')
    ax1.set_ylabel('465 fluor. (mV)', color=[0.1,0.7,0.2])
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98,0.98))
    ax1.set_xlim(tzoom)

    ax2.plot(ts, isos_405_airPLS_centered, color=[0.7,0.7,0.7], label='isos_405_airPLS_centered')
    ax2.set_ylabel('405 fluor. (mV)', color=[0.7,0.7,0.7])
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98,0.98))
    ax2.set_xlim(tzoom)
    ax2.set_xlabel('Time (seconds)')

    ax1.set_title('Streams with baseline correction (airPLS) and zero-centering')

    return (isos_405_baseline, GCaMP_465_baseline, 
            isos_405_airPLS_centered, GCaMP_465_airPLS_centered, 
            fig_a, fig_c)




# normalization by delta F / F ------------------------------------------------------------------------------------------------------------
# updated 08.10.26

def deltaFF(signal, control, ts):
    
    # function to compute deltaF/F using fitted control channel and filtered signal channel

    res = np.subtract(signal, control) # numerator of F(t)-f0
    normData = np.divide(res, control) # (F(t)-f0)/F0
    normData = normData*100
    
    # =====================================================================
    #  FIGURE: dF/F plot
    # =====================================================================

    fig_dff, ax = plt.subplots(figsize=(10, 4))

    ax.plot(ts, normData, color=[0.1, 0.6, 0.7],
            label='GCaMP_465_dF/F', linewidth=2)

    ax.set_ylabel('465 (dF/F)', color=[0.1, 0.6, 0.7])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('dF/F normalization of GCaMP 465')

    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))

    #if yzoom is not None:
    #    ax.set_ylim(yzoom)

    #ax.set_xlim(tzoom)


    return normData



# normalization by z-scoring ------------------------------------------------------------------------------------------------------------
# updated 08.10.26

def zscore(ts, isos_405, GCaMP_465, tzoom, yzoom=None):
    
    """
    Z-score normalization and dF/F calculation with plotting format 
    matched to baselinecorr().
    """

    # --- Z-score normalization ---
    isos_405_zscore = (isos_405 - np.mean(isos_405)) / np.std(isos_405)
    GCaMP_465_zscore = (GCaMP_465 - np.mean(GCaMP_465)) / np.std(GCaMP_465)

    # =====================================================================
    #  FIGURE 1: Two stacked subplots (same style as baselinecorr)
    # =====================================================================

    fig_z, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # --- 405 Z-score ---
    ax1.plot(ts, isos_405_zscore, color=[0.7, 0.7, 0.7],
             label='isos_405_zscore', linewidth=2)
    ax1.set_ylabel('405 (z-score)', color=[0.7, 0.7, 0.7])
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    ax1.set_xlim(tzoom)

    # --- 465 Z-score ---
    ax2.plot(ts, GCaMP_465_zscore, color=[0.1, 0.7, 0.2],
             label='GCaMP_465_zscore', linewidth=2)
    ax2.set_ylabel('465 (z-score)', color=[0.1, 0.7, 0.2])
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    ax2.set_xlim(tzoom)
    ax2.set_xlabel('Time (seconds)')

    ax1.set_title('Z-score normalization of 405 and 465 signals')

  
    if yzoom is not None:
        ax1.set_ylim(yzoom)

    ax1.set_xlim(tzoom)

    return (
        isos_405_zscore,
        GCaMP_465_zscore,
        fig_z
    )


# visually compare normalizations ------------------------------------------------------------------------------------------------------------
# updated 08.10.26

def norm_compare(GCaMP_465_zscore, GCaMP_465_dFF, ts, tzoom):
    
    plt.figure(figsize=(10,7))
    fig_6,ax1=plt.subplots()  # create a plot to allow for dual y-axes plotting
    plot1=ax1.plot(ts, GCaMP_465_dFF, color=[0.1, 0.6, 0.7]) 
    ax2=plt.twinx()# create a right y-axis, sharing x-axis on the same plot
    plot2=ax2.plot(ts, GCaMP_465_zscore, color=[0.1, 0.7, 0.2]) 

    #plt.plot(ts, corrected_data,color='r')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('465 fluor. dFF')
    ax1.set_title('normalization comparison')
    ax2.set_ylabel('465 fluor. zscore')
    legend = fig_6.legend(['GCaMP_465_dFF','GCaMP_465_zscore'],loc='upper right', bbox_to_anchor=(0.9, 0.88))

  
    # Zoomed-in plot 
    fig_zoom, ax1_zoom = plt.subplots(figsize=(10, 7)) 
    ax1_zoom.plot( ts, GCaMP_465_dFF, color=[0.1, 0.6, 0.7] ) 
    ax2_zoom = ax1_zoom.twinx() 
    ax2_zoom.plot( ts, GCaMP_465_zscore, color=[0.1, 0.7, 0.2] ) 
    # Set the zoomed x-axis range 
    ax1_zoom.set_xlim(tzoom) 
    ax1_zoom.set_xlabel('Time (s)') 
    ax1_zoom.set_ylabel('465 fluor. dFF') 
    ax1_zoom.set_title('Normalization comparison - zoomed') 
    ax2_zoom.set_ylabel('465 fluor. zscore') 
    fig_zoom.legend( ['GCaMP_465_dFF', 'GCaMP_465_zscore'], loc='upper right', bbox_to_anchor=(0.9, 0.88) )
    
    
    # quantitative comparisons
    
    r = np.corrcoef(GCaMP_465_dFF, GCaMP_465_zscore)[0, 1]
    print(f"r = {r:.4f}")

    from scipy.stats import linregress

    slope, intercept, r_value, p_value, stderr = linregress(
        GCaMP_465_dFF,
        GCaMP_465_zscore
    )

    print(f"slope = {slope:.4f}")
    print(f"intercept = {intercept:.4f}")
    print(f"R² = {r_value**2:.4f}")
 
    dff_z = (
    GCaMP_465_dFF - np.mean(GCaMP_465_dFF)
    ) / np.std(GCaMP_465_dFF)

    r = np.corrcoef(dff_z, GCaMP_465_zscore)[0, 1]

    print(f"Correlation between z(dF/F) and z-score = {r:.4f}")
    
    
    return fig_6, fig_zoom




# peak detection using scipy --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 08.14.25: modified to save peri-peak segments in a dataframe, and add a baseline correction step similar to epoc_streams
# modified to account for buffer regions in IntA SA segments, which should be ignored for peak measures. 
# modified to include artifact rejection to improve peak data quality
# documentation updated 08.10.26


def detect_peaks_old(
    ts, 
    new_fs, 
    x,
    mph=None, 
    mpd=100, 
    threshold=0, 
    prominence=None, 
    edge='rising',
    kpsh=False, 
    valley=False, 
    show=False, 
    ax=None, 
    title=True, 
    tzoom=None,
    plot_histogram=False,
    plot_overlay=False, 
    overlay_window=5,
    auc_peak_window=2,
    peak_trange=None,           # e.g., [-5, 10]
    baseline_trange=None,       # e.g., [-5, 3]
    pre_buffer_sec=0,           # seconds to exclude at start
    post_buffer_sec=0,          # seconds to exclude at end
    max_slope=None,             # dF/F per second (e.g., 10)
    min_width_sec=None,         # e.g., 0.05
    max_width_sec=None,         # optional
    max_amp_z=None,              # robust z-score (e.g., 6)
    show_rejected = False,
    plot_rejected_overlay=False
):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, peak_widths
    from scipy.stats import median_abs_deviation

    # ---------------------
    # Input cleanup
    # ---------------------
    x = np.atleast_1d(x).astype('float64')
    nan_mask = np.isnan(x)
    if np.any(nan_mask):
        x[nan_mask] = np.nanmin(x) - 1

    x_proc = -x if valley else x

    kwargs = {
        'height': mph,
        'distance': mpd,
        'threshold': threshold,
        'prominence': prominence
    }

    if edge == 'both':
        kwargs['plateau_size'] = True
    else:
        kwargs['plateau_size'] = None

    # ---------------------
    # Sampling rate
    # ---------------------
    if len(ts) < 2:
        raise ValueError("Time array too short.")
    fs = new_fs

    # ---------------------
    # Buffering
    # ---------------------
    start_idx = int(pre_buffer_sec * fs)
    end_idx = len(x) - int(post_buffer_sec * fs)
    if start_idx >= end_idx:
        raise ValueError("Buffers exclude entire signal.")

    x_proc_sub = x_proc[start_idx:end_idx]

    # ---------------------
    # Peak detection
    # ---------------------
    peaks_sub, properties = find_peaks(x_proc_sub, **kwargs)
    peaks = peaks_sub + start_idx

    # ==========================================================
    # ARTIFACT REJECTION
    # ==========================================================
    if len(peaks) > 0:
        valid_mask = np.ones(len(peaks), dtype=bool)

        # ---------- 1. SLOPE CHECK ----------
        if max_slope is not None:
            dx = np.gradient(x, 1/fs)  # dF/F per second
            for i, p in enumerate(peaks):
                local = slice(max(p-2, 0), min(p+3, len(dx)))
                if np.nanmax(np.abs(dx[local])) > max_slope:
                    valid_mask[i] = False
                    
        # ---------- 2. WIDTH CHECK ----------
        if min_width_sec is not None or max_width_sec is not None:
            widths, _, _, _ = peak_widths(x_proc, peaks, rel_height=0.5)
            widths_sec = widths / fs

            if min_width_sec is not None:
                valid_mask &= widths_sec >= min_width_sec
            if max_width_sec is not None:
                valid_mask &= widths_sec <= max_width_sec

        # ---------- 3. AMPLITUDE OUTLIER CHECK ----------
        if max_amp_z is not None and len(peaks) > 5:
            peak_vals = x[peaks]
            med = np.nanmedian(peak_vals)
            mad = median_abs_deviation(peak_vals, nan_policy='omit')
            if mad > 0:
                z = (peak_vals - med) / (1.4826 * mad)
                valid_mask &= np.abs(z) <= max_amp_z

        n_rejected = np.sum(~valid_mask)
        if n_rejected > 0:
            print(f"🚫 Rejected {n_rejected} artifactual peaks")

        accepted_peaks = peaks[valid_mask]
        rejected_peaks = peaks[~valid_mask]
        peaks = accepted_peaks

    # ==========================================================
    # PERI-PEAK EXTRACTION
    # ==========================================================
    peak_segments_raw = pd.DataFrame()
    peak_segments_baselined = None
    data_for_stats = None

    if peak_trange is not None and len(peaks) > 0:
        tr_start, tr_duration = peak_trange
        tr_end = tr_start + tr_duration
        n_samples = int(tr_duration * fs)

        for peak_idx in peaks:
            peak_time = ts[peak_idx]

            seg_start_time = peak_time + tr_start
            seg_end_time = peak_time + tr_end

            seg_start_idx = np.searchsorted(ts, seg_start_time)
            seg_end_idx = np.searchsorted(ts, seg_end_time)

            if seg_start_idx < start_idx or seg_end_idx > end_idx:
                continue

            segment = x[seg_start_idx:seg_end_idx]
            if len(segment) != n_samples:
                continue

            series = pd.Series(segment, name=f"peak_{peak_time:.3f}s")
            peak_segments_raw = pd.concat(
                [peak_segments_raw, series.reset_index(drop=True)], axis=1
            )

        # ---------- Baseline correction ----------
        if not peak_segments_raw.empty:
            if baseline_trange is not None:
                baseline_start, baseline_dur = baseline_trange
                baseline_end = baseline_start + baseline_dur

                baseline_start_idx = int((baseline_start - tr_start) * fs)
                baseline_end_idx = int((baseline_end - tr_start) * fs)

                if baseline_start_idx < 0 or baseline_end_idx > peak_segments_raw.shape[0]:
                    raise ValueError("Baseline range out of bounds.")

                baselines = peak_segments_raw.iloc[
                    baseline_start_idx:baseline_end_idx
                ].mean()

                peak_segments_baselined = peak_segments_raw.subtract(baselines, axis=1)
                data_for_stats = peak_segments_baselined
            else:
                data_for_stats = peak_segments_raw

    # ==========================================================
    # PERI-PEAK EXTRACTION (REJECTED PEAKS)
    # ==========================================================
    rejected_segments = pd.DataFrame()

    if plot_rejected_overlay and peak_trange is not None and len(rejected_peaks) > 0:
        tr_start, tr_duration = peak_trange
        tr_end = tr_start + tr_duration
        n_samples = int(tr_duration * fs)

        for peak_idx in rejected_peaks:
            peak_time = ts[peak_idx]

            seg_start_time = peak_time + tr_start
            seg_end_time   = peak_time + tr_end

            seg_start_idx = np.searchsorted(ts, seg_start_time)
            seg_end_idx   = np.searchsorted(ts, seg_end_time)

            if seg_start_idx < start_idx or seg_end_idx > end_idx:
                continue

            segment = x[seg_start_idx:seg_end_idx]
            if len(segment) != n_samples:
                continue

            rejected_segments = pd.concat(
                [rejected_segments,
                 pd.Series(segment, name=f"rej_{peak_time:.3f}s").reset_index(drop=True)],
                axis=1
            )

    # ==========================================================
    # PEAK STATS
    # ==========================================================
    if data_for_stats is not None and not data_for_stats.empty:
        peak_amps = data_for_stats.max(axis=0).values

        aucs = []
        if auc_peak_window is not None:
            half = int((auc_peak_window / 2) * fs)
            center = data_for_stats.shape[0] // 2
            auc_start = center - half
            auc_end = center + half

            auc_time = np.linspace(
                -auc_peak_window/2, auc_peak_window/2, auc_end - auc_start
            )

            for col in data_for_stats.columns:
                trace = data_for_stats[col].iloc[auc_start:auc_end]
                aucs.append(np.trapz(trace, auc_time))
        else:
            aucs = np.array([])

        duration = ts[end_idx - 1] - ts[start_idx]
        peak_freq = len(peak_amps) / duration if duration > 0 else 0
    else:
        peak_amps = x[peaks] if len(peaks) > 0 else np.array([])
        peak_freq = len(peaks) / (ts[end_idx-1] - ts[start_idx]) if end_idx > start_idx else 0
        aucs = np.array([])

    # ==========================================================
    # OPTIONAL PLOTTING (unchanged)
    # ==========================================================
    
    def _plot(ts, x, mph, mpd, threshold, prominence, edge, valley, ax, ind, title, tzoom=None, properties=None):
        """Plot results with optional zoomed-in view and prominence lines."""
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 10))
            own_fig = True
        else:
            own_fig = False

        label = 'valleys' if valley else 'peaks'
        color_signal = [0.1, 0.7, 0.2]
        color_peaks = [1, 0, 1]

        ax[0].plot(ts, x, color=color_signal, lw=1)
        if ind.size:
            ax[0].plot(ts[ind], x[ind], '+', mfc=None, mec=color_peaks, mew=2, ms=8, label=f'{len(ind)} {label}')
            ax[0].legend(loc='best', framealpha=.5, numpoints=1)

        ax[0].set_xlim(ts[0], ts[-1])
        ax[0].set_ylim(np.nanmin(x) - 0.1 * np.nanstd(x), np.nanmax(x) + 0.1 * np.nanstd(x))
        ax[0].set_xlabel('Time (seconds)', fontsize=14)
        ax[0].set_ylabel('Amplitude', fontsize=14)

        if title:
            if not isinstance(title, str):
                mode = 'Valley detection' if valley else 'Peak detection'
                title = f"{mode} (mph={mph}, mpd={mpd}, threshold={threshold}, prominence={prominence}, edge='{edge}')"
            ax[0].set_title(title)

        if tzoom is not None:
            start_time, end_time = tzoom
            zoom_indices = np.where((ts >= start_time) & (ts <= end_time))[0]

            if zoom_indices.size > 0:
                ax[1].plot(ts[zoom_indices], x[zoom_indices], color=color_signal, lw=1)
                zoomed_in_peaks = ind[(ts[ind] >= start_time) & (ts[ind] <= end_time)]
                if zoomed_in_peaks.size:
                    ax[1].plot(ts[zoomed_in_peaks], x[zoomed_in_peaks], '+', mfc=None, mec=color_peaks, mew=2, ms=8,
                               label=f'{len(zoomed_in_peaks)} {label}')
                    ax[1].legend(loc='best', framealpha=.5, numpoints=1)

                    if properties and 'prominences' in properties:
                        # Get indices of zoomed-in peaks relative to original peak array
                        zoom_mask = (ts[ind] >= start_time) & (ts[ind] <= end_time)
                        zoomed_in_indices = np.where(zoom_mask)[0]  # these are relative to `ind`

                        for j, idx in enumerate(zoomed_in_indices):
                            peak_idx = ind[idx]
                            y_top = x[peak_idx]
                            y_bottom = y_top - properties['prominences'][idx]
                            ax[1].vlines(ts[peak_idx], y_bottom, y_top, color='orange', linestyle='dashed', label='Prominence' if j == 0 else "")

                ax[1].set_xlim(start_time, end_time)
                ax[1].set_ylim(np.nanmin(x[zoom_indices]) - 0.1 * np.nanstd(x[zoom_indices]),
                               np.nanmax(x[zoom_indices]) + 0.1 * np.nanstd(x[zoom_indices]))
                ax[1].set_xlabel('Time (seconds)', fontsize=14)
                ax[1].set_ylabel('Amplitude', fontsize=14)
                ax[1].set_title(f"Zoomed-in View: {start_time}s to {end_time}s", fontsize=14)

        if own_fig:
            plt.tight_layout()
            plt.show()

    
    if show:
        _plot(ts, x, mph, mpd, threshold, prominence, edge,
              valley, ax, peaks, title, tzoom, properties)

    # ==========================================================
    # OPTIONAL: OVERLAY OF REJECTED PEAK SEGMENTS
    # ==========================================================
    if plot_rejected_overlay and not rejected_segments.empty:
        aligned = rejected_segments.values.T
        time_aligned = np.linspace(
            peak_trange[0],
            peak_trange[0] + peak_trange[1],
            aligned.shape[1]
        )

        mean_rejected = np.nanmean(aligned, axis=0)

        plt.figure(figsize=(10, 6))

        for seg in aligned:
            plt.plot(time_aligned, seg, color='gray', alpha=0.4, linewidth=1)

        plt.plot(
            time_aligned,
            mean_rejected,
            color='red',
            linewidth=4,
            label='Mean rejected peak'
        )

        if auc_peak_window is not None:
            auc_mask = (
                (time_aligned >= -auc_peak_window/2) &
                (time_aligned <=  auc_peak_window/2)
            )
            plt.fill_between(
                time_aligned[auc_mask],
                mean_rejected[auc_mask],
                0,
                color='red',
                alpha=0.3,
                label='AUC window'
            )

        plt.title('Overlay of Rejected Peak Segments')
        plt.xlabel('Time Relative to Peak (s)')
        plt.ylabel('Amplitude')
        plt.xlim([-2,2])
        plt.legend()
        plt.tight_layout()
        plt.show()


    if plot_histogram and peak_amps.size > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(peak_amps, bins='auto', color='purple', alpha=0.7, edgecolor='white')
        plt.title('Histogram of Peak Amplitudes')
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    if plot_overlay and data_for_stats is not None:
        aligned_array = data_for_stats.values.T
        time_aligned = np.linspace(peak_trange[0], peak_trange[0] + peak_trange[1], aligned_array.shape[1])
        mean_segment = np.nanmean(aligned_array, axis=0)

        plt.figure(figsize=(10, 6))
        n = aligned_array.shape[0]
        cmap = plt.cm.get_cmap('Greens', n)

        for i, seg in enumerate(aligned_array):
            plt.plot(time_aligned, seg, color=cmap(i), linewidth=1.5)

        plt.plot(time_aligned, mean_segment, color='red', linewidth=5, label='Average Peak')

        if auc_peak_window is not None:
            auc_mask = (time_aligned >= -auc_peak_window/2) & (time_aligned <= auc_peak_window/2)
            plt.fill_between(time_aligned[auc_mask], mean_segment[auc_mask], 0,
                             color='magenta', alpha=0.6, label='AUC (Mean Trace)', zorder=5)

        plt.title('Overlay of Aligned Peaks')
        plt.xlabel('Time Relative to Peak (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

        
    return (
        peaks,
        peak_freq,
        peak_amps,
        np.array(aucs),
        peak_segments_raw,
        peak_segments_baselined
    )



# peak detection using scipy --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 08.14.25: modified to save peri-peak segments in a dataframe, and add a baseline correction step similar to epoc_streams
# modified to account for buffer regions in IntA SA segments, which should be ignored for peak measures. 
# modified to include artifact rejection to improve peak data quality
# documentation updated 08.10.26


def detect_peaks(
    ts, 
    new_fs, 
    x,
    mph=None, 
    mpd=100, 
    threshold=0, 
    prominence=None, 
    kpsh=False, 
    valley=False, 
    show=False, 
    ax=None, 
    title=True, 
    tzoom=None,
    plot_histogram=False,
    plot_overlay=False, 
    overlay_window=5,
    auc_peak_window=2,
    peak_trange=None,           # e.g., [-5, 10]
    baseline_trange=None,       # e.g., [-5, 3]
    pre_buffer_sec=0,           # seconds to exclude at start
    post_buffer_sec=0,          # seconds to exclude at end
    max_slope=None,             # dF/F per second (e.g., 10)
    min_width_sec=None,         # e.g., 0.05
    max_width_sec=None,         # optional
    max_amp_z=None,              # robust z-score (e.g., 6)
    show_rejected = False,
    plot_rejected_overlay=False
):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, peak_widths
    from scipy.stats import median_abs_deviation

    # ---------------------
    # Input cleanup
    # ---------------------
    # ensure that the input data is an array, even if a single number
    # double precision floating point representation, precise to ~16 decimal places
    x = np.atleast_1d(x).astype('float64')
    nan_mask = np.isnan(x) 
    if np.any(nan_mask):
        x[nan_mask] = np.nanmin(x) - 1

    x_proc = -x if valley else x

    # dictionary of key word arguments (kwargs)
    kwargs = {
        'height': mph,
        'distance': mpd,
        'threshold': threshold,
        'prominence': prominence
    }
  
    # ---------------------
    # Sampling rate
    # ---------------------
    if len(ts) < 2:
        raise ValueError("Time array too short.")
    fs = new_fs

    # ---------------------
    # Buffering
    # the buffer_sec variables are inherited from the stream trimming code
    # ---------------------
    start_idx = int(pre_buffer_sec * fs)
    end_idx = len(x) - int(post_buffer_sec * fs)
    if start_idx >= end_idx:
        raise ValueError("Buffers exclude entire signal.")

    x_proc_sub = x_proc[start_idx:end_idx]

    # ---------------------
    # Peak detection
    # ---------------------
    # **kwargs unpacks the key word arguments dictionary
    
    peaks_sub, properties = find_peaks(x_proc_sub, **kwargs)
    peaks = peaks_sub + start_idx

    # ==========================================================
    # ARTIFACT REJECTION
    # ==========================================================

    # Initialize rejected peaks
    rejected_peaks = np.array([], dtype=int)

    if len(peaks) > 0:
        # start with assumption of acceptance for all detected peaks
        valid_mask = np.ones(len(peaks), dtype=bool)

        # ---------- 1. SLOPE CHECK ----------
        if max_slope is not None:
            dx = np.gradient(x, 1/fs)  # dF/F per second using np.gradient, which calculates numerical derivative of an array
            for i, p in enumerate(peaks):
                # a slice object is used to define the index bounds of dx to examine, basic python built-in
                local = slice(max(p-2, 0), min(p+3, len(dx))) 
                if np.nanmax(np.abs(dx[local])) > max_slope:
                    valid_mask[i] = False
                    
        # ---------- 2. WIDTH CHECK ----------
        if min_width_sec is not None or max_width_sec is not None:
            # use the scipy function peak_widths
            widths, _, _, _ = peak_widths(x_proc, peaks, rel_height=0.5)  
            widths_sec = widths / fs

            if min_width_sec is not None:
                # the &= operator is a compound assignment operator for element-wise logical AND
                # in other words, update valid_mask with new rejection criteria
                # in such cases, new False elements overwrite previously True elements, thus creating successive filtering of accepted peaks 
                valid_mask &= widths_sec >= min_width_sec
            if max_width_sec is not None:
                valid_mask &= widths_sec <= max_width_sec

        # ---------- 3. AMPLITUDE OUTLIER CHECK ----------
        if max_amp_z is not None and len(peaks) > 5:
            peak_vals = x[peaks]
            med = np.nanmedian(peak_vals)
            # the median absolute deviation measures how "spread out" the peak amplitudes are
            # difference of each peak from median, then the median of those differences
            # prevents skewing of processing by outlier amplitudes
            # the 1.4826 is a scaling factor between median abs deviation and the standard deviation of a normal distribution
            mad = median_abs_deviation(peak_vals, nan_policy='omit')
            if mad > 0:
                z = (peak_vals - med) / (1.4826 * mad)
                valid_mask &= np.abs(z) <= max_amp_z

        n_rejected = np.sum(~valid_mask)
        if n_rejected > 0:
            print(f"🚫 Rejected {n_rejected} artifactual peaks")

        accepted_peaks = peaks[valid_mask]
        rejected_peaks = peaks[~valid_mask]
        peaks = accepted_peaks

    # ==========================================================
    # PERI-PEAK EXTRACTION
    # ==========================================================
    peak_segments_raw = pd.DataFrame()
    peak_segments_baselined = None
    data_for_stats = None

    if peak_trange is not None and len(peaks) > 0:
        tr_start, tr_duration = peak_trange
        tr_end = tr_start + tr_duration
        n_samples = int(tr_duration * fs)

        for peak_idx in peaks:
            peak_time = ts[peak_idx]

            seg_start_time = peak_time + tr_start
            seg_end_time = peak_time + tr_end

            seg_start_idx = np.searchsorted(ts, seg_start_time)
            seg_end_idx = np.searchsorted(ts, seg_end_time)

            if seg_start_idx < start_idx or seg_end_idx > end_idx:
                continue

            segment = x[seg_start_idx:seg_end_idx]
            if len(segment) != n_samples:
                continue 

            series = pd.Series(segment, name=f"peak_{peak_time:.3f}s")
            peak_segments_raw = pd.concat(
                [peak_segments_raw, series.reset_index(drop=True)], axis=1
            )

        # ---------- Baseline correction ----------
        if not peak_segments_raw.empty:
            if baseline_trange is not None:
                baseline_start, baseline_dur = baseline_trange
                baseline_end = baseline_start + baseline_dur

                baseline_start_idx = int((baseline_start - tr_start) * fs)
                baseline_end_idx = int((baseline_end - tr_start) * fs)

                if baseline_start_idx < 0 or baseline_end_idx > peak_segments_raw.shape[0]:
                    raise ValueError("Baseline range out of bounds.")

                baselines = peak_segments_raw.iloc[
                    baseline_start_idx:baseline_end_idx
                ].mean()

                peak_segments_baselined = peak_segments_raw.subtract(baselines, axis=1)
                data_for_stats = peak_segments_baselined
            else:
                data_for_stats = peak_segments_raw

    # ==========================================================
    # PERI-PEAK EXTRACTION (REJECTED PEAKS)
    # ==========================================================
    rejected_segments = pd.DataFrame()

    if plot_rejected_overlay and peak_trange is not None and len(rejected_peaks) > 0:
        tr_start, tr_duration = peak_trange
        tr_end = tr_start + tr_duration
        n_samples = int(tr_duration * fs)

        for peak_idx in rejected_peaks:
            peak_time = ts[peak_idx]

            seg_start_time = peak_time + tr_start
            seg_end_time   = peak_time + tr_end

            seg_start_idx = np.searchsorted(ts, seg_start_time)
            seg_end_idx   = np.searchsorted(ts, seg_end_time)

            if seg_start_idx < start_idx or seg_end_idx > end_idx:
                continue

            segment = x[seg_start_idx:seg_end_idx]
            if len(segment) != n_samples:
                continue

            rejected_segments = pd.concat(
                [rejected_segments,
                 pd.Series(segment, name=f"rej_{peak_time:.3f}s").reset_index(drop=True)],
                axis=1
            )

    # ==========================================================
    # PEAK STATS
    # ==========================================================
    if data_for_stats is not None and not data_for_stats.empty:
        #peak_amps = data_for_stats.max(axis=0).values
        # updated peak amp measure
        center = data_for_stats.shape[0] // 2
        peak_amps = data_for_stats.iloc[center, :].values
        
        aucs = []
        if auc_peak_window is not None:
            half = int((auc_peak_window / 2) * fs)
            center = data_for_stats.shape[0] // 2
            auc_start = center - half
            auc_end = center + half

            auc_time = np.linspace(
                -auc_peak_window/2, auc_peak_window/2, auc_end - auc_start
            )

            for col in data_for_stats.columns:
                trace = data_for_stats[col].iloc[auc_start:auc_end]
                aucs.append(np.trapz(trace, auc_time))
        else:
            aucs = np.array([])

        duration = ts[end_idx - 1] - ts[start_idx]
        peak_freq = len(peak_amps) / duration if duration > 0 else 0
    else:
        peak_amps = x[peaks] if len(peaks) > 0 else np.array([])
        peak_freq = len(peaks) / (ts[end_idx-1] - ts[start_idx]) if end_idx > start_idx else 0
        aucs = np.array([])

    # ==========================================================
    # OPTIONAL PLOTTING
    # ==========================================================
    
    def _plot(ts, x, mph, mpd, threshold, prominence, valley, ax, ind, title, tzoom=None, properties=None):
        """Plot results with optional zoomed-in view and prominence lines."""
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 10))
            own_fig = True
        else:
            own_fig = False

        label = 'valleys' if valley else 'peaks'
        color_signal = [0.1, 0.7, 0.2]
        color_peaks = [1, 0, 1]

        ax[0].plot(ts, x, color=color_signal, lw=1)
        if ind.size:
            ax[0].plot(ts[ind], x[ind], '+', mfc=None, mec=color_peaks, mew=2, ms=8, label=f'{len(ind)} {label}')
            ax[0].legend(loc='best', framealpha=.5, numpoints=1)

        ax[0].set_xlim(ts[0], ts[-1])
        ax[0].set_ylim(np.nanmin(x) - 0.1 * np.nanstd(x), np.nanmax(x) + 0.1 * np.nanstd(x))
        ax[0].set_xlabel('Time (seconds)', fontsize=14)
        ax[0].set_ylabel('Amplitude', fontsize=14)

        if title:
            if not isinstance(title, str):
                mode = 'Valley detection' if valley else 'Peak detection'
                title = f"{mode} (mph={mph}, mpd={mpd}, threshold={threshold}, prominence={prominence}"
            ax[0].set_title(title)

        if tzoom is not None:
            start_time, end_time = tzoom
            zoom_indices = np.where((ts >= start_time) & (ts <= end_time))[0]

            if zoom_indices.size > 0:
                ax[1].plot(ts[zoom_indices], x[zoom_indices], color=color_signal, lw=1)
                zoomed_in_peaks = ind[(ts[ind] >= start_time) & (ts[ind] <= end_time)]
                if zoomed_in_peaks.size:
                    ax[1].plot(ts[zoomed_in_peaks], x[zoomed_in_peaks], '+', mfc=None, mec=color_peaks, mew=2, ms=8,
                               label=f'{len(zoomed_in_peaks)} {label}')
                    ax[1].legend(loc='best', framealpha=.5, numpoints=1)

                    if properties and 'prominences' in properties:
                        # Get indices of zoomed-in peaks relative to original peak array
                        zoom_mask = (ts[ind] >= start_time) & (ts[ind] <= end_time)
                        zoomed_in_indices = np.where(zoom_mask)[0]  # these are relative to `ind`

                        for j, idx in enumerate(zoomed_in_indices):
                            peak_idx = ind[idx]
                            y_top = x[peak_idx]
                            y_bottom = y_top - properties['prominences'][idx]
                            ax[1].vlines(ts[peak_idx], y_bottom, y_top, color='orange', linestyle='dashed', label='Prominence' if j == 0 else "")

                ax[1].set_xlim(start_time, end_time)
                ax[1].set_ylim(np.nanmin(x[zoom_indices]) - 0.1 * np.nanstd(x[zoom_indices]),
                               np.nanmax(x[zoom_indices]) + 0.1 * np.nanstd(x[zoom_indices]))
                ax[1].set_xlabel('Time (seconds)', fontsize=14)
                ax[1].set_ylabel('Amplitude', fontsize=14)
                ax[1].set_title(f"Zoomed-in View: {start_time}s to {end_time}s", fontsize=14)

        if own_fig:
            plt.tight_layout()
            plt.show()

    
    if show:
        _plot(ts, x, mph, mpd, threshold, prominence,
              valley, ax, peaks, title, tzoom, properties)

    # ==========================================================
    # OPTIONAL: OVERLAY OF REJECTED PEAK SEGMENTS
    # ==========================================================
    if plot_rejected_overlay and not rejected_segments.empty:
        aligned = rejected_segments.values.T
        time_aligned = np.linspace(
            peak_trange[0],
            peak_trange[0] + peak_trange[1],
            aligned.shape[1]
        )

        mean_rejected = np.nanmean(aligned, axis=0)

        plt.figure(figsize=(10, 6))

        for seg in aligned:
            plt.plot(time_aligned, seg, color='gray', alpha=0.4, linewidth=1)

        plt.plot(
            time_aligned,
            mean_rejected,
            color='red',
            linewidth=4,
            label='Mean rejected peak'
        )

        if auc_peak_window is not None:
            auc_mask = (
                (time_aligned >= -auc_peak_window/2) &
                (time_aligned <=  auc_peak_window/2)
            )
            plt.fill_between(
                time_aligned[auc_mask],
                mean_rejected[auc_mask],
                0,
                color='red',
                alpha=0.3,
                label='AUC window'
            )

        plt.title('Overlay of Rejected Peak Segments')
        plt.xlabel('Time Relative to Peak (s)')
        plt.ylabel('Amplitude')
        plt.xlim([-2,2])
        plt.legend()
        plt.tight_layout()
        plt.show()


    if plot_histogram and peak_amps.size > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(peak_amps, bins='auto', color='purple', alpha=0.7, edgecolor='white')
        plt.title('Histogram of Peak Amplitudes')
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    if plot_overlay and data_for_stats is not None:
        aligned_array = data_for_stats.values.T
        time_aligned = np.linspace(peak_trange[0], peak_trange[0] + peak_trange[1], aligned_array.shape[1])
        mean_segment = np.nanmean(aligned_array, axis=0)

        plt.figure(figsize=(10, 6))
        n = aligned_array.shape[0]
        cmap = plt.cm.get_cmap('Greens', n)

        for i, seg in enumerate(aligned_array):
            plt.plot(time_aligned, seg, color=cmap(i), linewidth=1.5)

        plt.plot(time_aligned, mean_segment, color='red', linewidth=5, label='Average Peak')

        if auc_peak_window is not None:
            auc_mask = (time_aligned >= -auc_peak_window/2) & (time_aligned <= auc_peak_window/2)
            plt.fill_between(time_aligned[auc_mask], mean_segment[auc_mask], 0,
                             color='magenta', alpha=0.6, label='AUC (Mean Trace)', zorder=5)

        plt.title('Overlay of Aligned Peaks')
        plt.xlabel('Time Relative to Peak (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

        
    return (
        peaks,
        peak_freq,
        peak_amps,
        np.array(aucs),
        peak_segments_raw,
        peak_segments_baselined
    )








# extract stream data flanking each epoc event, average and plot ---------------------------------------------------------------------

# added the 'event' input
# updated 04.29.25 to adjust a few things

    # Issue	Fix
    # Wrong use of abs() in epoch range	Replace with proper signed math
    # NaN check drops columns redundantly	Move dropna() out of the loop
    # Stats computed before NaN drop	Move after drop or use skipna
    # Out-of-bounds indexing not checked	Add guard for start, end
    # Legend handling hardcoded	Use label= and ax.legend()

    # this allows a subset of events to be examined

# 06.24.25:  modification to measure mean fluorescence and AUC aligned to events of interest
# 08.14.25: modified to include a baseline subtraction step to better align perievent traces, and to add approach window
# 11.13.25: updated to include two AUC windows
# 11.14.25: includes events that fall beyond bounds instead of discarding, pads with NaNs 
# 01.31.26: updated to loop over data and event types, and include randoms
# 08.21.26: deleted associated random event and circular shift functions, as I didn't end up using them
# 08.24.26: updating documentation


def epoc_streams(
    epoc_ts_and_indices,
    trange,  # e.g. [-10, 25] is a start of 10 seconds prior to event and 25 sec duration
    stream,
    new_fs,
    ts,
    event,
    tzoom,
    cue_duration,
    subset=None,
    cue_window=(0, 4),
    auc_pre_window=(-3, 0),
    auc_post_window=(0, 4),
    approach_window=None,
    plot_auc_region=False,
    baseline_trange=None,
    subset_plots = None,
    pre_buffer_sec = None,
    label=None,
    ):

    # Initialize DataFrame to store concatenated epochs
    GCaMP_465_epocs = pd.DataFrame()

    # Use a boolean mask to filter the relevant epoc indices, 
    #e.g. only the indices from rows in which event = "Drug Infusion"
    
    event_indices = epoc_ts_and_indices['epoc_indices'][epoc_ts_and_indices['event'] == event]

    # Apply subset restriction if specified
    if subset is not None:
        if isinstance(subset, int):
            event_indices = event_indices.iloc[:subset]
        elif isinstance(subset, str) and subset.startswith("last"):
            num = int(subset.replace("last", ""))
            event_indices = event_indices.iloc[-num:]
        elif isinstance(subset, (tuple, list)) and len(subset) == 2:
            event_indices = event_indices.iloc[subset[0]:subset[1]]
        else:
            raise ValueError("Invalid subset format.")

    # ----------------------------------------------------------
    # Pad out-of-bounds epochs with NaNs rather than discard
    # ----------------------------------------------------------
    win_len = int(trange[1] * new_fs)

    for onset in event_indices:
        start = int(onset + trange[0] * new_fs)
        end   = start + win_len

        epoch = np.full(win_len, np.nan)

        valid_start = max(start, 0)
        valid_end   = min(end, len(stream))

        epoch_start_idx = valid_start - start
        epoch_end_idx   = epoch_start_idx + (valid_end - valid_start)

        epoch[epoch_start_idx:epoch_end_idx] = stream[valid_start:valid_end]

        epoch_series = pd.Series(epoch, name=f'onset_{(onset / new_fs - pre_buffer_sec):.3f}s')
        GCaMP_465_epocs = pd.concat([GCaMP_465_epocs,
                                     epoch_series.reset_index(drop=True)], axis=1)

    print("Number of events found:", len(event_indices))

    if GCaMP_465_epocs.empty:
        print("No valid epochs found. Skipping stats and plotting.")
        return (
            GCaMP_465_epocs,          # epochs
            None,                     # baselined epochs
            pd.DataFrame(),           # mean/std/sem df
            np.array([]),             # mean trace
            np.array([]),             # std trace
            np.array([]),             # sem trace
            None,                     # fig_5
            None,                     # fig_15
            (np.nan, np.nan),         # cue stats
            (np.nan, np.nan),         # auc pre
            (np.nan, np.nan),         # auc post
            (np.nan, np.nan),         # approach
            None                      # subset_figs  ✅ REQUIRED
        )


    if GCaMP_465_epocs.shape[1] <= 2:
        print(f"Warning: Only {GCaMP_465_epocs.shape[1]} valid epoch(s). Plotting will proceed.")

    # Detect & drop all-NaN columns (bad events)
    if GCaMP_465_epocs.isna().any().any():
        na_cols = GCaMP_465_epocs.columns[GCaMP_465_epocs.isna().all()]
        if len(na_cols) > 0:
            print("Dropping all-NaN event columns:", na_cols.tolist())
        GCaMP_465_epocs = GCaMP_465_epocs.drop(columns=na_cols)


    # ------------------------------------------------------
    # Baseline subtraction with valid baseline requirement
    # ------------------------------------------------------
    if baseline_trange is not None:
        baseline_start_time = baseline_trange[0]
        baseline_end_time   = baseline_trange[1]

        # indices to use for baseline subtraction
        baseline_start_idx = int((baseline_start_time - trange[0]) * new_fs)
        baseline_end_idx   = int((baseline_end_time   - trange[0]) * new_fs)

        if baseline_start_idx < 0 or baseline_end_idx > GCaMP_465_epocs.shape[0]:
            raise ValueError("Baseline range is out of bounds relative to extracted segments.")

        baseline_slice = GCaMP_465_epocs.iloc[baseline_start_idx:baseline_end_idx]

        # --- NEW: require ≥10% non-NaN per event ---
        valid_fraction = baseline_slice.notna().sum(axis=0) / len(baseline_slice)

        # columns failing validation:
        bad_cols = valid_fraction[valid_fraction < 0.1].index.tolist()
        if len(bad_cols) > 0:
            print(f"Dropping {len(bad_cols)} events: insufficient baseline samples (<10%).")
            GCaMP_465_epocs = GCaMP_465_epocs.drop(columns=bad_cols)
            baseline_slice   = baseline_slice.drop(columns=bad_cols)

        # Now safe to compute baseline means and subtract for each column / epoch
        baselines = baseline_slice.mean()
        GCaMP_465_epocs_baselined = GCaMP_465_epocs.subtract(baselines, axis=1)

    else:
        print("no baseline correction requested")
        GCaMP_465_epocs_baselined = None

    
    data_to_use = GCaMP_465_epocs_baselined if GCaMP_465_epocs_baselined is not None else GCaMP_465_epocs

    # the mean trace is computed along axis = 1, aka mean across columns/events at each timepoint
    mean_epoc_stream = np.nanmean(data_to_use, axis=1)
   
    num_epochs = data_to_use.shape[1]

    if num_epochs >= 2:
        std_epoc_stream  = np.nanstd(data_to_use, axis=1, ddof=1)
        sem_epoc_stream  = stats.sem(data_to_use, axis=1, nan_policy="omit")
    else:
        std_epoc_stream = np.full(data_to_use.shape[0], np.nan)
        sem_epoc_stream = np.full(data_to_use.shape[0], np.nan)
        print("⚠ Only one valid epoch left — STD and SEM cannot be computed.")
    
    #---------------------
    # plotting, full stream
    
    ticks = epoc_ts_and_indices['epoc_ts'][epoc_ts_and_indices['event'] == event]
    drugavail = epoc_ts_and_indices['epoc_ts'][epoc_ts_and_indices['event'] == 'drug available onset']

    fig_15, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    ax1.plot(ts, stream, color=[0.1,0.7,0.2])
    ax1.set_xlim(tzoom)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal')
    ax1.plot(ticks, np.full_like(ticks, 1.5*np.max(stream)), '|', label='event', color='w')
    ax1.plot(drugavail, np.full_like(drugavail, 1.55*np.max(stream)), '|', label='drug available onset', color='g')
    ax1.legend(loc='upper right')
    if label is not None:
        ax1.set_title(label)

    # --- plot event ticks ---
    # Plot event ticks for chosen event
    for t in ticks:
        if tzoom[0] <= t <= tzoom[1]:
            ax1.plot(t, np.nanmax(stream), '|', markersize=18, color='k', zorder=10)

    # Plot drug availability ticks
    for t in drugavail:
        if tzoom[0] <= t <= tzoom[1]:
            ax1.plot(t, np.nanmax(stream), '|', markersize=18, color='c', zorder=10)
    
    

    #---------------------
    # plot perievent epochs
    
    epoc_ts = trange[0] + np.arange(0, len(mean_epoc_stream)) / new_fs
        
    fig_5, ax2 = plt.subplots(1, 1, figsize=(10, 7))
    if label is not None:
        ax2.set_title(f"{label}\nPeri-event aligned traces", fontsize=12)

     
    # ------------------------------
    # Grey single-event traces
    # ------------------------------
    ax2.plot(epoc_ts, data_to_use, color=(0.4, 0.4, 0.4), linewidth=0.5, zorder=1)

    ax2.axhline(0, color='white', linestyle=':', linewidth=1)
    ax2.plot([0, 0],
             [np.nanmin(data_to_use.values), np.nanmax(data_to_use.values)],
             'r', linewidth=3)

    # ------------------------------
    # Mean + SEM
    # ------------------------------
    ax2.fill_between(epoc_ts,
                     mean_epoc_stream + sem_epoc_stream,
                     mean_epoc_stream - sem_epoc_stream,
                     facecolor=[1, 1, 0], alpha=0.4, zorder=11)
    ax2.plot(epoc_ts, mean_epoc_stream, color=[1, 1, 0], linewidth=3, zorder=12)

    # Tight y-axis based on mean ± SEM
    # Compute ymin and ymax safely
    if data_to_use.shape[1] == 0:
        # No valid epochs
        print("⚠ No valid epochs. Using default y-limits [0, 1].")
        ymin, ymax = 0, 1
    elif data_to_use.shape[1] == 1:
        # Only one valid epoch
        print("⚠ Only one valid epoch. Using min/max of that epoch for y-limits.")
        epoch_vals = data_to_use.iloc[:, 0].values
        epoch_vals = epoch_vals[np.isfinite(epoch_vals)]  # drop NaNs
        if len(epoch_vals) == 0:
            ymin, ymax = 0, 1
        else:
            ymin, ymax = np.min(epoch_vals), np.max(epoch_vals)
    else:
        # Two or more epochs: mean ± SEM
        ymin = np.nanmin(mean_epoc_stream - sem_epoc_stream)
        ymax = np.nanmax(mean_epoc_stream + sem_epoc_stream)

    # Fallback in case still NaN
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = 0, 1

    # Set axis with padding
    ax2.set_ylim(ymin - 0.5*(ymax - ymin), ymax + 0.5*(ymax - ymin))

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Signal')

    # ------------------------------
    # Data-relative bars based on mean trace
    # ------------------------------
    mean_max = np.nanmax(mean_epoc_stream)
    bar_bottom = ymax * 1.5
    bar_top    = ymax * 1.6
    label_y    = bar_top + 0.02 #* mean_max

    # Cue bar
    ax2.fill_between([0, cue_duration], bar_bottom, bar_top, color='yellow', alpha=0.3)
    ax2.text(0.05, label_y, 'cue light', color='yellow', va='bottom')

    # Approach bar
    if approach_window is not None:
        ax2.fill_between([approach_window[0], approach_window[1]], bar_bottom, bar_top,
                         color='magenta', alpha=0.3)
        ax2.text(approach_window[0], label_y, 'approach', color='magenta', va='bottom')

    
    # bounds of the AUC windows
    
    pre_start_idx = np.searchsorted(epoc_ts, auc_pre_window[0])
    pre_end_idx   = np.searchsorted(epoc_ts, auc_pre_window[1])
    post_start_idx = np.searchsorted(epoc_ts, auc_post_window[0])
    post_end_idx   = np.searchsorted(epoc_ts, auc_post_window[1])
    
    # ------------------------------
    # AUC shading ON TOP of mean trace
    # ------------------------------
    if plot_auc_region:
        # PRE
        ax2.fill_between(epoc_ts[pre_start_idx:pre_end_idx],
                         mean_epoc_stream[pre_start_idx:pre_end_idx],
                         0,
                         color='green', alpha=0.5,
                         label='Pre-event AUC', zorder=13)
        # POST
        ax2.fill_between(epoc_ts[post_start_idx:post_end_idx],
                         mean_epoc_stream[post_start_idx:post_end_idx],
                         0,
                         color='blue', alpha=0.5,
                         label='Post-event AUC', zorder=13)

        ax2.legend(loc='upper right')


    # ------------------------------
    # Optional subset plots (still data-relative)
    # ------------------------------
    subset_figs = None
    if subset_plots is not None:
        subset_figs = []
        data_to_use_for_subset = data_to_use.copy()
        epoc_ts_subset = epoc_ts

        gradient_colors = [[1,0,0],[0,1,0], [0,0,1]]

        for i, (start_idx, end_idx) in enumerate(subset_plots):
            if start_idx < 0 or end_idx > data_to_use_for_subset.shape[1]:
                print(f"⚠ Subset {i} out of bounds. Skipping.")
                continue

            subset_data = data_to_use_for_subset.iloc[:, start_idx:end_idx]
            num_events = subset_data.shape[1]

            cmap = LinearSegmentedColormap.from_list(f'subset_gradient_{i}', gradient_colors, N=num_events)
            colors = [cmap(j/(num_events-1)) for j in range(num_events)] if num_events > 1 else [to_rgb(gradient_colors[0])]

            fig_subset, ax_subset = plt.subplots(figsize=(10, 6))
            for j, col in enumerate(subset_data.columns):
                ax_subset.plot(epoc_ts_subset, subset_data[col], color=colors[j], linewidth=1.5, zorder=j+1)

            ax_subset.axhline(0, color='white', linestyle=':', linewidth=1)
            ax_subset.plot([0, 0],
                           [np.nanmin(subset_data.values), np.nanmax(subset_data.values)],
                           'r', linewidth=2)
            ax_subset.set_title(f"{label}\nSubset event traces {start_idx}–{end_idx}")


            # Bars relative to raw data
            subset_max = np.nanmax(subset_data.values)
            bar_bottom = subset_max * 2
            bar_top = subset_max * 2.2
            label_y = bar_top + 0.03 * subset_max

            # Cue bar
            ax_subset.fill_between([0, cue_duration], bar_bottom, bar_top, color='yellow', alpha=0.3)
            ax_subset.text(0.05, label_y, 'cue light', color='yellow', va='bottom')

            # Approach bar
            if approach_window is not None:
                ax_subset.fill_between([approach_window[0], approach_window[1]], bar_bottom, bar_top,
                                       color='magenta', alpha=0.3)
                ax_subset.text(-3, label_y, 'approach', color='magenta', va='bottom')

            ax_subset.set_title(f'Subset event traces {start_idx} to {end_idx}')
            ax_subset.set_xlabel('Time (s)')
            ax_subset.set_ylabel('Signal')
            plt.tight_layout()

            subset_figs.append(fig_subset)
    
    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------
    GCaMP_465_epocs_mean_std_sem = pd.DataFrame({
        'mean_epoc_stream': mean_epoc_stream,
        'std_epoc_stream':  std_epoc_stream,
        'sem_epoc_stream':  sem_epoc_stream
    })

    # Cue window stats
    cue_start_idx = np.searchsorted(epoc_ts, cue_window[0])
    cue_end_idx   = np.searchsorted(epoc_ts, cue_window[1])
    cue_vals = data_to_use.iloc[cue_start_idx:cue_end_idx]
    cue_window_mean = np.nanmean(cue_vals.values)
    cue_means_per_event = np.nanmean(cue_vals, axis=0)
    if len(cue_means_per_event) >= 2:
        cue_window_sem = stats.sem(cue_means_per_event, nan_policy="omit")
    else:
        cue_window_sem = np.nan

    # use np.trapz to measure the area under the curve
    # inputs are the steam data and the epoc_ts in the identified indices
    # Pre-event AUC
    pre_vals = []
    for col in data_to_use.columns:
        trace = data_to_use[col].iloc[pre_start_idx:pre_end_idx]
        ts_seg = epoc_ts[pre_start_idx:pre_end_idx]
        pre_vals.append(np.trapz(trace, ts_seg))

    auc_pre_values = np.array(pre_vals)
    auc_pre_mean = np.nanmean(auc_pre_values)
    if len(auc_pre_values) >= 2:
        auc_pre_sem = stats.sem(auc_pre_values, nan_policy='omit')
    else:
        auc_pre_sem = np.nan

    # Post-event AUC
    post_vals = []
    for col in data_to_use.columns:
        trace = data_to_use[col].iloc[post_start_idx:post_end_idx]
        ts_seg = epoc_ts[post_start_idx:post_end_idx]
        post_vals.append(np.trapz(trace, ts_seg))

    auc_post_values = np.array(post_vals)
    auc_post_mean = np.nanmean(auc_post_values)
    if len(auc_post_values) >= 2:
        auc_post_sem = stats.sem(auc_post_values, nan_policy='omit')
    else:
        auc_post_sem = np.nan

    # Approach window stats
    if approach_window is not None:
        a_start = np.searchsorted(epoc_ts, approach_window[0])
        a_end   = np.searchsorted(epoc_ts, approach_window[1])
        a_vals = data_to_use.iloc[a_start:a_end]
        approach_window_mean = np.nanmean(a_vals.values)
        approach_means = np.nanmean(a_vals, axis=0)

        if len(approach_means) >= 2:
            approach_window_sem = stats.sem(approach_means, nan_policy='omit')
        else:
            approach_window_sem = np.nan
            
    else:
        approach_window_mean = np.nan
        approach_window_sem  = np.nan

    return (
        GCaMP_465_epocs,
        GCaMP_465_epocs_baselined,
        GCaMP_465_epocs_mean_std_sem,
        mean_epoc_stream,
        std_epoc_stream,
        sem_epoc_stream,
        fig_5,
        fig_15,
        (cue_window_mean, cue_window_sem),
        (auc_pre_mean, auc_pre_sem),
        (auc_post_mean, auc_post_sem),
        (approach_window_mean, approach_window_sem),
        subset_figs, 
    )




# concatenate partial multiple recording segments from single animals ---------------------------------------------------------------------    
# 01.23.26, update to include individual epoc data in outputs for later mixed model analysis potentially
# 02.02.26 udpate to show within animal means, across animal means as separate plots 
# updating 02.19.26 to add new output for event latency, count by file
# updating documentation 08.25.26

'''
run_perievent_pipeline is a large orchestration function, which at it's core uses perievent_analysis() to analyze preprocessed event-aligned photometry data

key steps:

1. Organize matched Feather files by rat and generate file-level event summaries.
2. tier 1: WITHIN-RAT:
   For each rat, call `perievent_analysis()` on that rat's files.
   Store per-rat epocs, statistics, traces, and optional figures.
3. tier 2: ACROSS-TRIAL:
   Call `perievent_analysis()` once on all matched files.
   Generate pooled trial-level epocs, statistics, summaries, and figures.
4. tier 3: ACROSS-ANIMAL:
   Aggregate the per-rat results from Step 2 to calculate
   animal-level means, SEMs, time-resolved traces, and summary statistics.
5. Save requested tables/figures and return all results in `pipeline_results`.

    matched files
         │ 
         ├──► group by rat
         │
         ▼
    ┌──────────────────────────────┐
    │ WITHIN-RAT                   │
    │ perievent_analysis() × rats  │
    └──────────────┬───────────────┘
                   │
                   ▼
            per-rat results
                   │
                   ▼
    ┌──────────────────────────────┐
    │ ACROSS-ANIMAL                │
    │ aggregate per-rat results    │
    └──────────────────────────────┘

    matched files (all)
         │
         ▼
    ┌──────────────────────────────┐
    │ ACROSS-TRIAL                 │
    │ perievent_analysis()         │
    │ on all files                 │
    └──────────────────────────────┘

    Returns a large dictionary containing within-rat, across-trial, across-animal, summary-statistic, and figure outputs.


'''


def run_perievent_pipeline(
    matched_files,
    save_dir,
    processed_phase,
    eventname,
    subset=None,
    *,
    compute_kwargs,
    overwrite=False,
    save_within_rat=False,
    master_xlim = None,
    master_ylim = None,
    excel_file_path = None,
    plot_group_comparison=True,
    plot_baseline_significance=True,
    paperfigs = False,
    paperfig_path = None,
    master_figsize = (2,2),
    plot_onset_detection=False,
):
    import os
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    from pathlib import Path
    import re
    from scipy import stats
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.patches import Patch
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    #########################################################################################################

    def configure_figure_style(paperfigs=False, master_figsize=(8,8)):
    
        if paperfigs:
            plt.style.use("default")

            plt.rcParams.update({
                "figure.facecolor": "none",
                "axes.facecolor": "none",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "0.85",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.figsize": master_figsize,
                "savefig.facecolor": "none",
                "font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
            })

        else:
            plt.style.use("dark_background")
            
    configure_figure_style(
        paperfigs=paperfigs,
        master_figsize=master_figsize
    )
    

    def configure_axis(ax):
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.minorticks_off()

    def get_output_path(default_path, paperfigs, paperfig_path):

        if paperfigs:
            os.makedirs(paperfig_path, exist_ok=True)
            return paperfig_path
        else:
            os.makedirs(default_path, exist_ok=True)
            return default_path
        
    fig_output_path = get_output_path(
        save_dir,
        paperfigs,
        paperfig_path
    )
    
    #########################################################################################################
    
    # ---------------------------------
    # Handle optional analysis window
    # ---------------------------------

    compute_kwargs_local = compute_kwargs.copy()

    # -----------------------------
    # Determine final time range
    # -----------------------------
    analysis_trange = compute_kwargs_local.pop("analysis_trange", None)

    if analysis_trange is not None:
        start, duration = analysis_trange
    else:
        start, duration = compute_kwargs_local.get("trange", [-10, 25])

    # If trange is meant as [start, duration], compute end
    end = start + duration

    # Save final trange back
    trange_to_use = [start, end]
    compute_kwargs_local["trange"] = trange_to_use

    fs = compute_kwargs_local.get("new_fs", 20.0)  # sampling frequency
    n_samples = int(duration * fs)  # number of samples based on duration

    ts = np.linspace(start, end, n_samples, endpoint=False)  # use endpoint=False for exact duration


        
    # ----------------------
    # Helper: safe feather/csv save
    # ----------------------
    def safe_save(df, path):
        if df is None:
            return False
        if os.path.exists(path) and not overwrite:
            print(f"↪️  Reusing existing file: {os.path.basename(path)}")
            return False
        df = df.reset_index(drop=True)
        df.columns = df.columns.astype(str)
        df.to_feather(path)
        print(f"💾 Saved: {os.path.basename(path)}")
        return True

    def safe_save_table(df, base_path):
        """
        Save a DataFrame safely to Feather and CSV, ensuring Feather-compatible types.
        Converts all object columns to strings to avoid ArrowTypeError.
        """
        import os
        if df is None:
            return False

        feather_path = base_path + ".feather"
        csv_path     = base_path + ".csv"

        # Reset index and ensure column names are strings
        df = df.reset_index(drop=True)
        df.columns = df.columns.astype(str)

        # Convert any object columns to string to prevent ArrowTypeError
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        # Save Feather
        if not os.path.exists(feather_path) or overwrite:
            df.to_feather(feather_path)
            print(f"💾 Saved: {os.path.basename(feather_path)}")
        else:
            print(f"↪️  Reusing existing file: {os.path.basename(feather_path)}")

        # Save CSV
        if not os.path.exists(csv_path) or overwrite:
            df.to_csv(csv_path, index=False)
            print(f"💾 Saved: {os.path.basename(csv_path)}")
        else:
            print(f"↪️  Reusing existing file: {os.path.basename(csv_path)}")

        return True


    def safe_save_txt(text, path):
        if text is None:
            return False
        if os.path.exists(path) and not overwrite:
            print(f"↪️  Reusing existing file: {os.path.basename(path)}")
            return False
        with open(path, "w") as f:
            f.write(text)
        print(f"💾 Saved: {os.path.basename(path)}")
        return True

    
    def safe_save_fig(fig, path_base):
        """
        Saves figure as PNG, PDF, and SVG automatically.
        `path_base` should NOT include extension.
        Example: safe_save_fig(fig, "my_figure")
        """

        if fig is None:
            return False

        # --- Illustrator-friendly font settings ---
        mpl.rcParams['pdf.fonttype'] = 42      # Keep text editable in PDF
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['svg.fonttype'] = 'none'  # Keep text editable in SVG

        saved_any = False

        formats = {
            "png": {"dpi": 600},
            "pdf": {},
            "svg": {}
        }

        for ext, kwargs in formats.items():
            full_path = f"{path_base}.{ext}"

            if os.path.exists(full_path) and not overwrite:
                print(f"↪️  Reusing existing figure: {os.path.basename(full_path)}")
                continue

            fig.savefig(
                full_path,
                bbox_inches='tight',
                transparent= True,
                facecolor= None,
                edgecolor="none",
                **kwargs
            )

            print(f"🖼️  Saved: {os.path.basename(full_path)}")
            saved_any = True

        return saved_any


    # ----------------------
    # Parse rat from filename
    # ----------------------
    def parse_rat(filename):
        fname = Path(filename).stem
        rat_match = re.search(r'ACW_coh5_IT_[fm]\d+', fname)
        return rat_match.group(0) if rat_match else "unknown_rat"

    # ----------------------
    # Group files by rat
    # ----------------------
    rat_files = defaultdict(list)
    for f in matched_files:
        rat_files[parse_rat(f)].append(f)

    print(f"\n🐀 Detected {len(rat_files)} rat(s): {list(rat_files.keys())}")

    # ----------------------
    # Initialize storage
    # ----------------------
    per_rat_stats = {}
    per_rat_results = {}
    per_animal_traces = []
    per_rat_epoc_tables = []

    # ----------------------
    # NEW OUTPUT: File-level summary table with first onset times and event count e.g. for correlation with drug_avail signals
    # ----------------------
    file_summary_rows = []

    onset_pattern = re.compile(r"onset_([0-9.]+)s")

    for rat in sorted(rat_files.keys()):
        for f in sorted(rat_files[rat]):

            df = pd.read_feather(f)

            # Extract onset times from column headers
            onset_times = []
            for col in df.columns:
                match = onset_pattern.search(str(col))
                if match:
                    onset_times.append(float(match.group(1)))

            if len(onset_times) > 0:
                first_event_onset_sec = min(onset_times)
            else:
                first_event_onset_sec = np.nan

            n_events = len(onset_times)

            file_summary_rows.append({
                "rat_id": rat,
                "matched_file": Path(f).name,
                "first_event_onset_sec": first_event_onset_sec,
                "n_events": n_events
            })

    file_summary_df = (
        pd.DataFrame(file_summary_rows)
        .sort_values(["rat_id", "matched_file"])
        .reset_index(drop=True)
    )

    # Save file-level summary
    file_summary_base = os.path.join(
        save_dir,
        f"{processed_phase}_{eventname}_file_level_summary"
    )

    safe_save_table(file_summary_df, file_summary_base)
        
    # -------------------
    # Tier 1: per-rat processing
    # -------------------
    #compute_kwargs_local = compute_kwargs.copy()
    #analysis_trange = compute_kwargs_local.pop("analysis_trange", None)

    
    for rat, files in rat_files.items():
        (
            combined_epocs,
            combined_baselined,
            combined_stats,
            mean_stream,
            std_stream,
            sem_stream,
            _fig_trial,
            _fig_across,
            _fig_within,
            _fig_onset,
            fig_trials,
            (cue_mean, cue_sem),
            (auc_pre_mean, auc_pre_sem),
            (auc_post_mean, auc_post_sem),
            (approach_mean, approach_sem),
            (onset_mean, onset_sem), 
            (slope_mean, slope_sem),  
            (time_to_peak_mean, time_to_peak_sem),
            (onset_peak_amp_mean, onset_peak_amp_sem),
            (overall_peak_amp_mean, overall_peak_amp_sem),
            (overall_peak_time_mean, overall_peak_time_sem),
            per_epoc_stats_df
        ) = perievent_analysis(
            feather_files=files,
            subset=subset,
            **compute_kwargs_local,
            plot=False,
            plot_onset_detection=False,
            master_xlim = master_xlim,
            master_ylim = master_ylim,
            overall_peak_window = (0,8),
            rat_name = rat,
            excel_file_path = excel_file_path,
            plot_group_comparison=plot_group_comparison,
            plot_baseline_significance=plot_baseline_significance,
            paperfigs = paperfigs,
            master_figsize = master_figsize
        )

            
        # Add rat metadata
        per_epoc_stats_df["rat_id"] = rat
        per_rat_epoc_tables.append(per_epoc_stats_df)

        per_rat_results[rat] = {
            "epocs": combined_epocs,
            "baselined_epocs": combined_baselined,
            "stats": combined_stats,
            "per_epoc_stats": per_epoc_stats_df
        }

        per_rat_stats[rat] = {
            "cue_stats": (cue_mean, cue_sem),
            "approach_stats": (approach_mean, approach_sem),
            "auc_pre_stats": (auc_pre_mean, auc_pre_sem),
            "auc_post_stats": (auc_post_mean, auc_post_sem),

            "onset_stats": (onset_mean, onset_sem),
            "slope_stats": (slope_mean, slope_sem),
            "time_to_peak_stats": (time_to_peak_mean, time_to_peak_sem),
            "onset_peak_amp_stats": (onset_peak_amp_mean, onset_peak_amp_sem),
            
            "overall_peak_amp_stats": (overall_peak_amp_mean, overall_peak_amp_sem),
            "overall_peak_time_stats": (overall_peak_time_mean, overall_peak_time_sem),
        }

        # Store per-animal traces
        mean_trace = np.nanmean(combined_baselined.values, axis=1)

        n_trials = combined_baselined.shape[1]

        if n_trials > 1:
            sem_trace = (
                np.nanstd(combined_baselined.values, axis=1, ddof=1)
                / np.sqrt(n_trials)
            )
        else:
            # Single-epoc case (e.g. program_start) → no within-animal variability
            sem_trace = np.zeros_like(mean_trace)

        per_animal_traces.append({
            "rat": rat,
            "mean": mean_trace,
            "sem": sem_trace
        })


        # Save per-rat files
        rat_base = os.path.join(save_dir, f"{processed_phase}_{rat}_{eventname}")
        rat_base_fig = os.path.join(fig_output_path, f"{processed_phase}_{rat}_{eventname}")
        safe_save_table(combined_baselined, rat_base + "_epocs_bslnd")
        safe_save_table(combined_stats, rat_base + "_stats")
        safe_save_table(per_epoc_stats_df, rat_base + "_per_epoc_stats")
       

        # per-rat figures can optionally be saved here
        if _fig_onset is not None:      
            safe_save_fig(_fig_onset, rat_base_fig + "_onset_slope")     # temp to assure this is working
            
        if fig_trials is not None:
            safe_save_fig(fig_trials, rat_base_fig + "_individual_trials")
            
        
    # -----------------------
    # Build within-animal averaged epocs DataFrames
    # -----------------------
    #rat_str = "-".join(sorted(per_rat_results.keys()))
    rat_str = "_".join([r.split('_')[-1] for r in sorted(per_rat_results.keys())])
    within_base = os.path.join(
        save_dir,
        f"{processed_phase}_{rat_str}_{eventname}_within_rat_epoc_bslnd"
    )

    # Mean trace per rat
    within_animal_epocs_mean = pd.DataFrame(
        {trace["rat"]: trace["mean"] for trace in per_animal_traces}
    )
    within_animal_epocs_mean.insert(0, "time", ts)

    # SEM trace per rat
    within_animal_epocs_sem = pd.DataFrame(
        {trace["rat"]: trace["sem"] for trace in per_animal_traces}
    )
    within_animal_epocs_sem.insert(0, "time", ts)

    safe_save_table(within_animal_epocs_mean, within_base + "_means")
    safe_save_table(within_animal_epocs_sem, within_base + "_sem")

    #########
    import matplotlib.pyplot as plt
    
    for rat in per_rat_stats.keys():
        trace_dict = next(t for t in per_animal_traces if t["rat"] == rat)
        mean_trace = trace_dict["mean"]
        sem_trace = trace_dict["sem"]

        rat_stats = per_rat_stats[rat]
        onset_mean = rat_stats["onset_stats"][0]
        slope_mean = rat_stats["slope_stats"][0]

        if plot_onset_detection and per_animal_traces is not None and len(per_animal_traces) > 0:
            fig, ax = plt.subplots(figsize=master_figsize if paperfigs else (6,4))
            ax.plot(ts, mean_trace, label="Mean Trace")
            ax.fill_between(ts, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3, linewidth=0,
                        edgecolor='none')

            # Event vertical line
            ax.axvline(0, color="red", linestyle="-", label="Event")

            # Onset vertical line
            if not np.isnan(onset_mean):
                ax.axvline(onset_mean, linestyle="--", color="magenta", label="Mean Onset")

                # Slice window starting just before onset
                mask = ts >= onset_mean
                t_win = ts[mask]
                y_win = mean_trace[mask]

                # Interpolate y at exact onset
                y0 = np.interp(onset_mean, ts, mean_trace)

                # Peak-based cutoff
                onset_peak_val = np.nanmax(y_win)
                threshold = 0.6 * onset_peak_val
                above_thresh_idx = np.where(y_win >= threshold)[0]

                if len(above_thresh_idx) > 0:
                    t_fit = t_win[:above_thresh_idx[0]+1]
                    slope_line = slope_mean * (t_fit - onset_mean) + y0  # start at interpolated y
                    ax.plot(t_fit, slope_line, color="green", label="Mean Slope")

            ax.set_title(f"{rat} - Within-Rat Onset & Slope")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal (z-score)")
            ax.set_xlim(-0.5, 2)
            ax.legend()
            plt.show()
        
        
    #########
    # -------------------
    # Tier 2: across-trial
    # -------------------
    all_files = [f for files in rat_files.values() for f in files]
    (
        combined_epocs_all,
        combined_baselined_all,
        combined_stats_all,
        mean_all,
        std_all,
        sem_all,
        fig_trial_all,
        fig_across_all,
        fig_within_all,
        fig_onset_all,
        fig_trials_all,
        (cue_mean_all, cue_sem_all),
        (auc_pre_mean_all, auc_pre_sem_all),
        (auc_post_mean_all, auc_post_sem_all),
        (approach_mean_all, approach_sem_all),
        (onset_mean_all, onset_sem_all),
        (slope_mean_all, slope_sem_all),
        (time_to_peak_mean_all, time_to_peak_sem_all),
        (onset_peak_amp_mean_all, onset_peak_amp_sem_all),
        (overall_peak_amp_mean_all, overall_peak_amp_sem_all),
        (overall_peak_time_mean_all, overall_peak_time_sem_all),
        per_epoc_stats_all
    ) = perievent_analysis(
        feather_files=all_files,
        subset=subset,
        **compute_kwargs_local,
        plot=True,
        per_animal_traces=per_animal_traces,
        plot_onset_detection=False,
        plot_individual_trials = True,
        master_xlim = master_xlim,
        master_ylim = master_ylim,
        overall_peak_window = (0,8),
        excel_file_path = excel_file_path,
        plot_group_comparison=plot_group_comparison,
        plot_baseline_significance=plot_baseline_significance,
        paperfigs = paperfigs,
        master_figsize = master_figsize

    )


    per_epoc_stats_all = pd.concat(per_rat_epoc_tables, ignore_index=True)
    #rat_str = "_".join(sorted(per_rat_results.keys()))
    rat_str = "_".join([r.split('_')[-1] for r in sorted(per_rat_results.keys())])
    
    # ---------- Across-trial summary ----------
    combined_event_dict_trial = {
        "event name": eventname,
        "rats included": rat_str,
        "event count": combined_epocs_all.shape[1],
        "subset": subset,
        "approach window (s)": str(compute_kwargs["approach_window"]),
        "mean approach response": f"{approach_mean_all:.3f}",
        "SEM approach response": f"{approach_sem_all:.3f}",
        "cue window (s)": str(compute_kwargs["cue_window"]),
        "mean cue response": f"{cue_mean_all:.3f}",
        "SEM cue response": f"{cue_sem_all:.3f}",
        "pre-event AUC window (s)": str(compute_kwargs["auc_pre_window"]),
        "mean pre-event AUC": f"{auc_pre_mean_all:.3f}",
        "SEM pre-event AUC": f"{auc_pre_sem_all:.3f}",
        "post-event AUC window (s)": str(compute_kwargs["auc_post_window"]),
        "mean post-event AUC": f"{auc_post_mean_all:.3f}",
        "SEM post-event AUC": f"{auc_post_sem_all:.3f}",
        "mean onset latency (s)": f"{onset_mean_all:.3f}",
        "SEM onset latency (s)": f"{onset_sem_all:.3f}",
        "mean rise slope": f"{slope_mean_all:.3f}",
        "SEM rise slope": f"{slope_sem_all:.3f}",
        "mean time to peak (s)": f"{time_to_peak_mean_all:.3f}",
        "SEM time to peak (s)": f"{time_to_peak_sem_all:.3f}",
        "mean onset peak amplitude": f"{onset_peak_amp_mean_all:.3f}",
        "SEM onset peak amplitude": f"{onset_peak_amp_sem_all:.3f}",
        "overall peak window (s)": str(compute_kwargs.get("overall_peak_window", (0,8))),
        "mean overall peak amplitude": f"{overall_peak_amp_mean_all:.3f}",
        "SEM overall peak amplitude": f"{overall_peak_amp_sem_all:.3f}",
        "mean overall peak timestamp (s)": f"{overall_peak_time_mean_all:.3f}",
        "SEM overall peak timestamp (s)": f"{overall_peak_time_sem_all:.3f}",
    }

    combined_events_info_trial = (
        pd.DataFrame.from_dict(combined_event_dict_trial, orient="index")
          .reset_index()
    )
    combined_events_info_trial.columns = ["combined perievent analysis", "value"]

    # Matched files
    matched_file_rows = pd.DataFrame({
        "combined perievent analysis": ["matched file"] * len(all_files),
        "value": [Path(f).name for f in all_files]
    })
    combined_events_info_trial = pd.concat([matched_file_rows, combined_events_info_trial], ignore_index=True).astype(str)

    # Save across-trial files
    base_all_trial = os.path.join(save_dir, f"{processed_phase}_{rat_str}_{eventname}_across_trial")
    base_all_trial_fig = os.path.join(fig_output_path, f"{processed_phase}_{rat_str}_{eventname}_across_trial")

    safe_save_table(combined_baselined_all, base_all_trial + "_epocs_bslnd")
    safe_save_table(combined_stats_all, base_all_trial + "_stats")
    safe_save_table(per_epoc_stats_all, base_all_trial + "_PER_EPOC_summary_stats")
    safe_save_table(combined_events_info_trial, base_all_trial + "_summary")
    safe_save_txt(combined_events_info_trial.to_string(index=False), base_all_trial + "_summary.txt")
    
    safe_save_fig(fig_trial_all, base_all_trial_fig + "_across_trials")
    safe_save_fig(fig_within_all, base_all_trial_fig + "_within_rats")

    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # ------------------------------------------------------------
    # FIGURE — Heatmap of ALL TRIALS (per-rat trial y-axis, right-aligned rat IDs)
    # ------------------------------------------------------------
    
    
    x_smooth_sigma = 2
    gap_between_rats = 10
    y_offset = 0

    yticks_trial = []
    yticklabels_trial = []
    yticks_rat = []
    yticklabels_rat = []

    # Stack traces to compute color scale
    within_means = np.vstack([trace["mean"] for trace in per_animal_traces])
    if master_xlim is not None:
        x_mask = (ts >= master_xlim[0]) & (ts <= master_xlim[1])
        ts_plot = ts[x_mask]
        within_means = within_means[:, x_mask]
    else:
        ts_plot = ts

    vlim2 = np.nanmax(np.abs(within_means))
    vlim = np.nanpercentile(np.abs(within_means), 98)
    
    ##############
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # --- Custom diverging colormap ---
    deep_green = np.array([36, 106, 72]) / 255
    green = np.array([54, 158, 90]) / 255
    purple = np.array([60, 30, 90]) / 255
    white = np.array([1, 1, 1])
    deep_redpurple = np.array([40, 15, 45]) / 255

    custom_cmap = LinearSegmentedColormap.from_list(
        "GreenPurple",
        [deep_redpurple, purple, white, green, deep_green],
        N=256
    )

    # Proper zero-centered normalization
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    ##############
    
    

    # Create figure with GridSpec: heatmap (left), colorbar (right)
    fig_all_trials = plt.figure(figsize=(2,0.75) if paperfigs else (11,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax_left = fig_all_trials.add_subplot(gs[0])
    ax_cbar = fig_all_trials.add_subplot(gs[1])


    '''
    # Compute a representative peak per rat using the mean across trials
    rat_peak_scores = {}
    for rat, trace in per_rat_results.items():
        df_rat = trace["baselined_epocs"]
        mean_trace = df_rat.mean(axis=1)  # mean across trials (row-wise)
        if master_xlim is not None:
            mean_trace = mean_trace[x_mask]  # apply x-limits if needed
        rat_peak_scores[rat] = mean_trace.max()  # use max of the mean trace

    # Sort rats by this representative peak (descending)
    sorted_rats = sorted(rat_peak_scores.keys(), key=lambda r: rat_peak_scores[r], reverse=False)
    '''
    
    # ------------------------------------------------------------
    # Determine rat order
    # ------------------------------------------------------------

    subject_order = ["m4","f5","m8","f2","m3","m9","m5"]
    #subject_order = None
    
    if subject_order is not None:

        # Extract short IDs from rat names
        rat_short = {rat: rat.split("_")[-1] for rat in per_rat_results.keys()}

        # Map manual order to full rat names
        manual_map = {v: k for k, v in rat_short.items()}

        sorted_rats = []
        for sid in subject_order:
            if sid in manual_map:
                sorted_rats.append(manual_map[sid])
            else:
                print(f"⚠️ Subject {sid} not found in data")

        # Add any rats not specified at the end
        remaining = [r for r in per_rat_results.keys() if r not in sorted_rats]
        sorted_rats.extend(remaining)
        sorted_rats = sorted_rats[::-1]

    else:
        # Default behavior: sort by peak amplitude
        rat_peak_scores = {}
        rat_mean_scores = {}
        
        for rat, trace in per_rat_results.items():
            df_rat = trace["baselined_epocs"]
            mean_trace = df_rat.mean(axis=1)

            if master_xlim is not None:
                mean_trace = mean_trace[x_mask]

            rat_peak_scores[rat] = mean_trace.max()
            rat_mean_scores[rat] = mean_trace.mean()
        
        '''
        sorted_rats = sorted(
            rat_peak_scores.keys(),
            key=lambda r: rat_peak_scores[r],
            reverse=False
        '''
            
        sorted_rats = sorted(
            rat_mean_scores.keys(),
            key=lambda r: rat_mean_scores[r],
            reverse=False
            
        )

    # Plot each rat in sorted order
    y_offset = 0
    yticks_trial, yticklabels_trial = [], []
    yticks_rat, yticklabels_rat = [], []

    for rat in sorted_rats:
        df_rat = per_rat_results[rat]["baselined_epocs"]

        # Sort trials within rat by peak amplitude (descending)
        #peak_vals = df_rat.min(axis=0)     # negpeak   
        peak_vals = df_rat.max(axis=0)   #peak
        sorted_cols = peak_vals.sort_values(ascending=False).index
        #df_sorted = df_rat[sorted_cols]                         ###### sort by max amplitude (when timing doesn't matter, e.g. noncont tests)
        df_sorted = df_rat                             ###### keep original trial order (chronological)
        

        # Apply heatmap_xlim
        if master_xlim is not None:
            df_sorted = df_sorted.loc[x_mask, :]

        # Gaussian smoothing
        if x_smooth_sigma > 0:
            df_sorted = df_sorted.apply(lambda col: gaussian_filter1d(col.values, sigma=x_smooth_sigma), axis=0)

        n_trials = df_sorted.shape[1]
        ts_plot = ts[x_mask] if master_xlim is not None else ts

        # Plot trials block (flip vertically so largest trial peaks at top)
        extent = [ts_plot[0], ts_plot[-1], y_offset, y_offset + n_trials]
        
        im = ax_left.imshow(
            df_sorted.T[::-1],
            aspect='auto',
            extent=extent,
            origin='lower',
            #cmap='viridis',
            cmap = custom_cmap,
            vmin=-vlim2,
            vmax=vlim2,
            interpolation='nearest'
        )
        
      
        # ==========================================================
        # LEFT Y-AXIS: 3 ticks, floor-rounded to nearest 5
        # First label always starts at 5
        # ==========================================================

        # Evenly spaced trial numbers
        trial_vals = np.linspace(1, n_trials, 3)

        # Floor to nearest 5
        trial_vals_floor5 = (5 * np.floor(trial_vals / 5)).astype(int)

        # Force first label to be 5
        trial_vals_floor5[0] = 5

        # Remove duplicates just in case (rare edge case)
        trial_vals_floor5 = np.unique(trial_vals_floor5)

        # Convert trial numbers to y positions (account for vertical flip)
        y_positions = y_offset + (n_trials - trial_vals_floor5)

        yticks_trial.extend(y_positions)
        yticklabels_trial.extend(trial_vals_floor5)

        # Right axis: rat label at top of block
        yticks_rat.append(y_offset + n_trials - 2)
        yticklabels_rat.append(rat)

        y_offset += n_trials + gap_between_rats

    # Configure left y-axis
    ax_left.set_ylabel('Trial # (per rat)')
    ax_left.set_yticks(yticks_trial)
    ax_left.set_yticklabels(yticklabels_trial)
    ax_left.set_ylim(0, y_offset)
    ax_left.set_xlabel('Time (s)')
    ax_left.set_xlim(ts_plot[0], ts_plot[-1])
    ax_left.set_title('All Trials Heatmap (per-animal trial count)')
    ax_left.axvline(0, color='r', linewidth=1)
    
    '''
    # Right y-axis for rat IDs
    ax_right = ax_left.twinx()
    ax_right.set_ylim(ax_left.get_ylim())
    ax_right.set_yticks(yticks_rat)
    ax_right.set_yticklabels(yticklabels_rat, rotation=0, ha='right')
    ax_right.set_ylabel('Rat ID')
    ax_right.tick_params(axis='y', pad=100)  # move labels 10 pts away from axis

    # Colorbar on separate axis
    fig_all_trials.colorbar(im, cax=ax_cbar, label='fluorescence (z-score)')
    '''
    
    # Right y-axis: rat labels, **outside plot**
    for i, rat_label in enumerate(yticklabels_rat):
        y = yticks_rat[i]
        ax_left.text(ts_plot[-1] + 0.5, y, rat_label, va='center', ha='left', fontsize=8)

    # Adjust the x-limits to give space for labels
    ax_left.set_xlim(ts_plot[0], ts_plot[-1] + 3)  # +3 adds padding for labels

    # Colorbar: create a separate axis to the right
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_left)
    cax = divider.append_axes("right", size="5%", pad=0.5)  # pad=0.5 gives space after labels
    fig_all_trials.colorbar(im, cax=cax, label='Fluorescence (z-score)')


    plt.tight_layout()
    plt.show()

    # Save figure
    heatmap_base = os.path.join(fig_output_path, f"{processed_phase}_{rat_str}_{eventname}_all_trials_heatmap")
    safe_save_fig(fig_all_trials, heatmap_base)
    
    
    # ------------------------------------------------------------
    # FIGURE — WITHIN-ANIMAL HEATMAP (MATCHES ALL-TRIALS STYLE)
    # ------------------------------------------------------------

    # Use SAME vlim and sorted_rats from trial heatmap
    fig_heatmap_within_rat = plt.figure(figsize=(2,0.75) if paperfigs else (11,6))
    #fig_heatmap_within_rat = plt.figure(figsize=(1.7,0. 6375) if paperfigs else (11,6))    # smaller version for cues, nonconts etc. fig
    gs2 = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax_left2 = fig_heatmap_within_rat.add_subplot(gs2[0])
    ax_cbar2 = fig_heatmap_within_rat.add_subplot(gs2[1])

    if within_animal_epocs_mean is None or within_animal_epocs_mean.empty:
        print("⚠️ No within-animal mean data available for heatmap.")
    else:

        epoc_ts = within_animal_epocs_mean["time"].to_numpy()

        # Keep same x-limits logic
        if master_xlim is not None:
            x_mask2 = (epoc_ts >= master_xlim[0]) & (epoc_ts <= master_xlim[1])
            epoc_ts_plot = epoc_ts[x_mask2]
        else:
            x_mask2 = slice(None)
            epoc_ts_plot = epoc_ts

        # Reorder rows to match sorted_rats from first heatmap
        rat_labels = sorted_rats
        heatmap_within_rat_data = (
            within_animal_epocs_mean[rat_labels]
            .to_numpy()
            .T
        )

        heatmap_within_rat_data = heatmap_within_rat_data[:, x_mask2]

        # Optional: same smoothing
        x_smooth_sigma_within_rat = 0
        if x_smooth_sigma_within_rat > 0:
            heatmap_within_rat_data = np.array([
                gaussian_filter1d(row, sigma=x_smooth_sigma_within_rat)
                for row in heatmap_within_rat_data
            ])

        # Plot (NO flipping needed — one row per rat)
        extent2 = [
            epoc_ts_plot[0],
            epoc_ts_plot[-1],
            0,
            heatmap_within_rat_data.shape[0]
        ]
        
        '''
        im2 = ax_left2.imshow(
            heatmap_within_rat_data,
            aspect='auto',
            extent=extent2,
            origin='lower',
            #cmap='viridis',
            cmap='seismic',
            vmin=-vlim,   # ← SAME SCALE AS FIRST HEATMAP
            vmax=vlim,
            interpolation='nearest' 
        )
        '''
        
        im2 = ax_left2.imshow(
            heatmap_within_rat_data,
            aspect='auto',
            extent=extent2,
            origin='lower',
            cmap=custom_cmap,
            #cmap='viridis',
            norm=norm,
            interpolation='nearest'
        )

        # Match styling
        ax_left2.axvline(0, color='r', linewidth=1)
        ax_left2.set_xlabel('Time (s)')
        ax_left2.set_xlim(epoc_ts_plot[0], epoc_ts_plot[-1])
        ax_left2.set_xticks([-5, 0, 5, 10, epoc_ts_plot[-1]])
        ax_left2.set_xticklabels([-5, 0, 5, 10, 15]) 
        ax_left2.set_ylabel('Rat ID')
        if not paperfigs:
            ax_left2.set_title('Within-Animal Heatmap') 

        # Y ticks centered per row
        yticks = np.arange(len(rat_labels)) + 0.5
        ax_left2.set_yticks(yticks)

        if paperfigs:
            ax_left2.set_yticklabels([])        # remove subject IDs
            ax_left2.set_ylabel("Subjects")
            ax_left2.tick_params(axis='y', length=0)  # remove tick marks
        else:
            ax_left2.set_yticklabels(rat_labels)

        ax_left2.set_ylim(0, len(rat_labels))      
        
        cbar = fig_heatmap_within_rat.colorbar(im2,cax=ax_cbar2)
        
        
        
        # Paperfigs condition
        if paperfigs:
            # Remove default vertical label
            cbar.ax.set_ylabel('')

            # Add horizontal label above the colorbar
            # Use axes coordinates (0-1), x=0.5 centers it horizontally
            ax_cbar2.text(
                0.5, 1.05,           # x, y in axes fraction coordinates
                'z-score',           # label text
                ha='center', va='bottom',
                fontsize=8,
                transform=ax_cbar2.transAxes
            )

            # Always update ticks
            vmin, vmax = -vlim, vlim
            vmin_tick = np.ceil(vmin * 2) / 2
            vmax_tick = np.floor(vmax * 2) / 2
            ticks = [t for t in (vmin_tick, 0, vmax_tick) if vmin <= t <= vmax]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

            # Optional: move ticks below label
            ax_cbar2.xaxis.set_ticks_position('top')
        else:
            # Normal vertical label
            cbar.set_label('Fluorescence (z-score)', fontsize=8)

        fig_heatmap_within_rat.tight_layout()

    plt.show()

    # Save figure
    heatmap_within_rat_base = os.path.join(
        fig_output_path,
        f"{processed_phase}_{rat_str}_{eventname}_within_rat_heatmap"
    )
    safe_save_fig(fig_heatmap_within_rat, heatmap_within_rat_base)
    
    
    # -------------------
    # -------------------
    # Tier 3: across-animal (CORRECTED)
    # -------------------

    # ---------------------------------
    # Build per-animal (per-rat) summary
    # ---------------------------------
    per_animal_epoc_stats = []
    for rat, stats_dict in per_rat_stats.items():
        per_animal_epoc_stats.append({
            "rat_id": rat,
            "cue_mean": stats_dict["cue_stats"][0],
            "approach_mean": stats_dict["approach_stats"][0],
            "auc_pre_mean": stats_dict["auc_pre_stats"][0],
            "auc_post_mean": stats_dict["auc_post_stats"][0],
            "onset_mean": stats_dict["onset_stats"][0],
            "slope_mean": stats_dict["slope_stats"][0],
            "time_to_peak_mean": stats_dict["time_to_peak_stats"][0],
            "onset_peak_amp_mean": stats_dict["onset_peak_amp_stats"][0],
            "overall_peak_amp_mean": stats_dict["overall_peak_amp_stats"][0],
            "overall_peak_time_mean": stats_dict["overall_peak_time_stats"][0],
        })

    per_animal_stats_df = pd.DataFrame(per_animal_epoc_stats)

    n_animals = len(per_animal_stats_df)

    # ---------------------------------
    # Across-animal scalar stats
    # ---------------------------------
    cue_mean_animal = per_animal_stats_df["cue_mean"].mean()
    if n_animals > 1:
        cue_sem_animal  = per_animal_stats_df["cue_mean"].std(ddof=1) / np.sqrt(n_animals)
    else:
        cue_sem_animal = 0.0
        
    approach_mean_animal = per_animal_stats_df["approach_mean"].mean()
    if n_animals > 1:
        approach_sem_animal  = per_animal_stats_df["approach_mean"].std(ddof=1) / np.sqrt(n_animals)
    else:
        approach_sem_animal = 0.0
        
    auc_pre_mean_animal = per_animal_stats_df["auc_pre_mean"].mean()
    
    if n_animals > 1:
        auc_pre_sem_animal  = per_animal_stats_df["auc_pre_mean"].std(ddof=1) / np.sqrt(n_animals)
    else:
        auc_pre_sem_animal = 0.0
        
    auc_post_mean_animal = per_animal_stats_df["auc_post_mean"].mean()
    if n_animals > 1:
        auc_post_sem_animal  = per_animal_stats_df["auc_post_mean"].std(ddof=1) / np.sqrt(n_animals)
    else:
        auc_post_sem_animal = 0.0
        
    if n_animals > 1:
        onset_sem_animal = per_animal_stats_df["onset_mean"].std(ddof=1) / np.sqrt(n_animals)
        slope_sem_animal = per_animal_stats_df["slope_mean"].std(ddof=1) / np.sqrt(n_animals)
    else:
        onset_sem_animal = 0.0
        slope_sem_animal = 0.0

    onset_mean_animal = per_animal_stats_df["onset_mean"].mean()
    slope_mean_animal = per_animal_stats_df["slope_mean"].mean()
    
    time_to_peak_mean_animal = per_animal_stats_df["time_to_peak_mean"].mean()
    onset_peak_amp_mean_animal = per_animal_stats_df["onset_peak_amp_mean"].mean()

    if n_animals > 1:
        time_to_peak_sem_animal = per_animal_stats_df["time_to_peak_mean"].std(ddof=1) / np.sqrt(n_animals)
        onset_peak_amp_sem_animal = per_animal_stats_df["onset_peak_amp_mean"].std(ddof=1) / np.sqrt(n_animals)
    else:
        time_to_peak_sem_animal = 0.0
        onset_peak_amp_sem_animal = 0.0
        
    overall_peak_amp_mean_animal = per_animal_stats_df["overall_peak_amp_mean"].mean()
    overall_peak_time_mean_animal = per_animal_stats_df["overall_peak_time_mean"].mean()

    if n_animals > 1:
        overall_peak_amp_sem_animal = (
            per_animal_stats_df["overall_peak_amp_mean"].std(ddof=1)
            / np.sqrt(n_animals)
        )
        overall_peak_time_sem_animal = (
            per_animal_stats_df["overall_peak_time_mean"].std(ddof=1)
            / np.sqrt(n_animals)
        )
    else:
        overall_peak_amp_sem_animal = 0.0
        overall_peak_time_sem_animal = 0.0

    # ---------------------------------
    # Across-animal time-resolved stats
    # ---------------------------------
    all_means = np.vstack([trace["mean"] for trace in per_animal_traces])

    mean_across_animal = np.nanmean(all_means, axis=0)
    if n_animals > 1:
        sem_across_animal = (
            np.nanstd(all_means, axis=0, ddof=1)
            / np.sqrt(n_animals)
        )
    else:
        sem_across_animal = np.zeros_like(mean_across_animal)

    across_animal_stats = pd.DataFrame({
        "mean_epoc_stream": mean_across_animal,
        "sem_epoc_stream":  sem_across_animal
    })


    across_animal_epocs = pd.DataFrame({
        "time": ts,
        "mean_trace": mean_across_animal
    })


    ########### onset and slope figure
    
    ########### Across-Animal Onset & Slope Figure
    fig_across_animal_onset, ax = plt.subplots(
        figsize=master_figsize if paperfigs else (6,4)
    )

    # Plot mean trace
    ax.plot(ts, mean_across_animal, label="Mean Trace")

    # SEM shading
    ax.fill_between(
        ts,
        mean_across_animal - sem_across_animal,
        mean_across_animal + sem_across_animal,
        alpha=0.3, linewidth=0, edgecolor='none'
    )

    # Event vertical line
    ax.axvline(0, color="red", linestyle="-", label="Event")

    # Onset vertical line
    if not np.isnan(onset_mean_animal):
        ax.axvline(onset_mean_animal, linestyle="--", color="magenta", label="Mean Onset")

        # Include small window around onset for accurate interpolation
        mask = (ts >= onset_mean_animal - 0.01) & (ts <= onset_mean_animal + 4)
        t_win = ts[mask]
        y_win = mean_across_animal[mask]

        # Interpolate y at exact onset
        y0 = np.interp(onset_mean_animal, t_win, y_win)

        # Peak-based cutoff for slope line
        onset_peak_val = np.nanmax(y_win)
        threshold = 0.6 * onset_peak_val
        above_thresh_idx = np.where(y_win >= threshold)[0]

        if len(above_thresh_idx) > 0:
            t_fit = t_win[:above_thresh_idx[0]+1]
            slope_line = slope_mean_animal * (t_fit - onset_mean_animal) + y0
            ax.plot(t_fit, slope_line, color="green", label="Mean Slope")

    ax.set_title("Across-Animal Onset & Rising Slope")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (z-score)")
    ax.set_xlim(-0.5, 2.5)
    ax.legend()
    plt.show()

    
    
    #################################
    
    # ---------------------------------
    # Across-animal summary table
    # ---------------------------------
    combined_event_dict_animal = {
        "event name": eventname,
        "rats included": "-".join(sorted(per_rat_results.keys())),
        "animal count": n_animals,
        "subset": subset,

        "approach window (s)": str(compute_kwargs["approach_window"]),
        "mean approach response": f"{approach_mean_animal:.3f}",
        "SEM approach response": f"{approach_sem_animal:.3f}",

        "post window (s)": str(compute_kwargs["cue_window"]),
        "mean post response": f"{cue_mean_animal:.3f}",
        "SEM post response": f"{cue_sem_animal:.3f}",

        "pre-event AUC window (s)": str(compute_kwargs["auc_pre_window"]),
        "mean pre-event AUC": f"{auc_pre_mean_animal:.3f}",
        "SEM pre-event AUC": f"{auc_pre_sem_animal:.3f}",

        "post-event AUC window (s)": str(compute_kwargs["auc_post_window"]),
        "mean post-event AUC": f"{auc_post_mean_animal:.3f}",
        "SEM post-event AUC": f"{auc_post_sem_animal:.3f}",
        
        "mean onset latency (s)": f"{onset_mean_animal:.3f}",
        "SEM onset latency (s)": f"{onset_sem_animal:.3f}",
        "mean rise slope": f"{slope_mean_animal:.3f}",
        "SEM rise slope": f"{slope_sem_animal:.3f}",
        
        "mean time to peak (s)": f"{time_to_peak_mean_animal:.3f}",
        "SEM time to peak (s)": f"{time_to_peak_sem_animal:.3f}",
        "mean onset peak amplitude": f"{onset_peak_amp_mean_animal:.3f}",
        "SEM onset peak amplitude": f"{onset_peak_amp_sem_animal:.3f}",
        
        "overall peak window (s)": str(compute_kwargs.get("overall_peak_window", (0,8))),
        "mean overall peak amplitude": f"{overall_peak_amp_mean_animal:.3f}",
        "SEM overall peak amplitude": f"{overall_peak_amp_sem_animal:.3f}",
        "mean overall peak timestamp (s)": f"{overall_peak_time_mean_animal:.3f}",
        "SEM overall peak timestamp (s)": f"{overall_peak_time_sem_animal:.3f}",
    }

    combined_events_info_animal = (
        pd.DataFrame.from_dict(combined_event_dict_animal, orient="index")
          .reset_index()
    )
    combined_events_info_animal.columns = ["combined perievent analysis", "value"]
    combined_events_info_animal = combined_events_info_animal.astype(str)

    # ---------------------------------
    # Save across-animal outputs
    # ---------------------------------
    base_all_animal = os.path.join(
        save_dir,
        f"{processed_phase}_{rat_str}_{eventname}_across_rat"
    )
    
    base_all_animal_fig = os.path.join(
        fig_output_path,
        f"{processed_phase}_{rat_str}_{eventname}_across_rat"
    )


    safe_save_table(across_animal_epocs, base_all_animal + "_epocs_bslnd")
    safe_save_table(across_animal_stats, base_all_animal + "_stats")
    safe_save_table(per_animal_stats_df, base_all_animal + "_PER_ANIMAL_summary_stats")
    safe_save_table(combined_events_info_animal, base_all_animal + "_summary")
    safe_save_txt(
        combined_events_info_animal.to_string(index=False),
        base_all_animal + "_summary.txt"
    )
    
    safe_save_fig(
        fig_across_all,
        base_all_animal_fig
    )
    
    safe_save_fig(fig_across_animal_onset,
                  base_all_animal_fig + "_onset_slope")


    print("Across-trial SEM (approach):", approach_sem_all)
    print("Across-animal SEM (approach):", approach_sem_animal)

    print("\n✅ Perievent pipeline complete")

    fig_all= {
        "trial": fig_trial_all,
        "across": fig_across_all,
        "within": fig_within_all
    }
    
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------
    
    if paperfigs:
        # ---------------------------
        # Figure setup
        # ---------------------------
        '''
        fig_stack = plt.figure(figsize=(1.4, 1.925))  # total height = top 1.4 + bottom 0.525 + some space
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1.4, 0.525],
            width_ratios=[20, 1],
            hspace=0.15,   # more vertical space
            wspace=0.05
        )

        '''
        # smaller size figure below for cues, noncont, etc. but a
        #scale = 0.85
        #scale = 1.42857143
        scale = 1
        
        
        fig_stack = plt.figure(figsize=(1.4*scale, 1.925*scale))

        '''
        gs = fig_stack.add_gridspec(
            2, 2,
            height_ratios=[1.4, 0.525],
            width_ratios=[20, 1],
            hspace=0.15,
            wspace=0.05
        )
        '''
        
        height_ratios = np.array([1.4, 0.525]) * scale
        width_ratios = np.array([20, 1]) * scale

        gs = fig_stack.add_gridspec(
            2, 2,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0.15,
            wspace=0.05
        )

        ax_trace = fig_stack.add_subplot(gs[0, 0])
        ax_heat = fig_stack.add_subplot(gs[1, 0], sharex=ax_trace)
        ax_cbar = fig_stack.add_subplot(gs[1, 1])

        # ---------------------------
        # Across-animal trace
        # ---------------------------
        dark_green = np.array([36,106,72])/255
        ax_trace.plot(ts, mean_across_animal, color=dark_green, lw=1.5, label="Mean across subjects")
        ax_trace.fill_between(
            ts,
            mean_across_animal - sem_across_animal,
            mean_across_animal + sem_across_animal,
            color=dark_green,
            alpha=0.4,
            linewidth=0,
            label="SEM"
        )
        ax_trace.axvline(0, color="red", lw=1, zorder=0)
        ax_trace.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8, zorder=0)


        # Remove x-axis
        ax_trace.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_trace.set_ylabel("Fluorescence (z-score)")

        # Set y-limits and ticks
        ax_trace.set_ylim(master_ylim)
        ylim_span = master_ylim[1] - master_ylim[0]

        if ylim_span <= 2.0:
            # ticks every 0.5 but constrained to master_ylim
            yticks = np.arange(master_ylim[0], master_ylim[1]+1e-6, 0.5)
        else:
            # ticks every 1.0 but constrained to master_ylim
            yticks = np.arange(master_ylim[0], master_ylim[1]+1e-6, 1.0)

        ax_trace.set_yticks(yticks)
        
        
        ax_trace.spines['bottom'].set_visible(False)
        
        # ---------------------------
        # Excel significance markers (dynamic vertical spacing)
        # ---------------------------
        if excel_file_path is not None:
            try:
                sig_excel = pd.read_excel(excel_file_path, sheet_name=None)
                y_top = master_ylim[1]

                # Count total rows of markers (group comparisons + baseline)
                n_group_sheets = len([name for name in sig_excel.keys() if "vs" in name.lower()]) if plot_group_comparison else 0
                n_baseline_cols = len(sig_excel["Significance From Baseline"].columns) if (plot_baseline_significance and "Significance From Baseline" in sig_excel) else 0
                total_rows = max(n_group_sheets + n_baseline_cols, 1)

                # 15% of axis for all markers, stack them evenly
                total_marker_height = 0.15 * (master_ylim[1] - master_ylim[0])
                spacing = total_marker_height / max(total_rows, 1)
                height = y_top + (total_rows - 1 - i) * spacing  # topmost row highest                
                
                legend_handles = []
                legend_labels = []

                # GROUP COMPARISONS
                if plot_group_comparison:
                    comparison_sheets = [name for name in sig_excel.keys() if "vs" in name.lower()]
                    n_sheets = len(comparison_sheets)
                    # Define key RGB colors for gradient: purple -> red -> orange
                    key_colors = np.array([
                        [128, 0, 128],    # purple
                        [255, 0, 0],      # red
                        [255, 165, 0]     # orange
                    ]) / 255  # normalize to 0-1

                    # Create a colormap spanning these colors
                    from matplotlib.colors import LinearSegmentedColormap
                    gradient_cmap = LinearSegmentedColormap.from_list("purple_red_orange", key_colors, N=256)
                    # Sample n_sheets colors evenly along the gradient
                    #colors = [gradient_cmap(i / max(n_sheets-1, 1)) for i in range(n_sheets)]

                    # 2️⃣ List all possible pairwise comparisons among 4 groups
                    all_comparisons = [
                        "Group 1 vs Group 2",
                        "Group 1 vs Group 3",
                        "Group 1 vs Group 4",
                        "Group 2 vs Group 3",
                        "Group 2 vs Group 4",
                        "Group 3 vs Group 4"
                    ]
                    # 3️⃣ Sample colors evenly along the gradient
                    n_total = len(all_comparisons)
                    comparison_colors = {comp: gradient_cmap(i / (n_total - 1)) for i, comp in enumerate(all_comparisons)}

                    for i, sheet_name in enumerate(comparison_sheets):
                        sig_vals = sig_excel[sheet_name].iloc[:,0].to_numpy()
                        if len(sig_vals) > 1:
                            sig_times = np.linspace(epoc_ts.min(), epoc_ts.max(), len(sig_vals))
                            interp_vals = np.interp(ts, sig_times, sig_vals)
                            mask = interp_vals > 0.5
                            height = y_top * (0.95 - i*spacing)  # scale dynamically

                            sc = ax_trace.scatter(
                                ts[mask],
                                height*np.ones(np.sum(mask)),
                                marker='o',
                                color=comparison_colors.get(sheet_name, (0.5,0.5,0.5)),  # fallback gray if missing
                                linewidths=0,
                                edgecolors='none',
                                s=7,
                                label=sheet_name
                            )

                            legend_handles.append(sc)
                            legend_labels.append(sheet_name)

                # BASELINE SIGNIFICANCE
                if plot_baseline_significance and "Significance From Baseline" in sig_excel:
                    sig_base = sig_excel["Significance From Baseline"]
                    cmap = plt.get_cmap("tab10")
                    for j, col in enumerate(sig_base.columns):
                        sig_vals = sig_base[col].to_numpy()
                        if len(sig_vals) > 1:
                            sig_times = np.linspace(epoc_ts.min(), epoc_ts.max(), len(sig_vals))
                            interp_vals = np.interp(ts, sig_times, sig_vals)
                            mask = interp_vals > 0.5
                            baseline_offset = n_group_sheets
                            height = y_top * (0.95 - (baseline_offset + j)*spacing)
                            sc = ax_trace.scatter(
                                ts[mask],
                                height*np.ones(np.sum(mask)),
                                marker='o',
                                #color=cmap(j%10),
                                color=dark_green,
                                linewidths=0,
                                edgecolors='none',
                                s=7,
                                label=f"{col} baseline"
                            )
                            #legend_handles.append(sc)
                            #legend_labels.append(f"{col} baseline")               
                
                # Legend off to the side
                # Create a "patch" representing SEM
                sem_patch = Rectangle((0,0), 1, 1, facecolor=dark_green, alpha=0.4, edgecolor='none')
                # Create a line representing the mean
                mean_line = Line2D([0,1],[0,0], color=dark_green, lw=1.5)
                # --- Baseline significance markers (collapsed into one) ---
                baseline_handle = Line2D(
                    [0], [0],
                    marker='o',
                    color='none',
                    markerfacecolor=dark_green,
                    markeredgecolor='none',  # remove black outline
                    markeredgewidth=0,
                    markersize=3,             # match scatter s=10 (~3pt)
                    linestyle='None',
                    label='Event-related transient'
                )

                # Use a tuple as a single legend entry
  
                all_handles = [(sem_patch, mean_line)] + [baseline_handle] + legend_handles
                all_labels  = ["Mean across subjects"] + ["Event-related transient"] + legend_labels

                
                ax_trace.legend(
                    handles=all_handles,
                    labels=all_labels,
                    handler_map={tuple: HandlerTuple(ndivide=None)},
                    loc='center left',
                    bbox_to_anchor=(1.35, 0.5),
                    fontsize=7,
                    frameon=True,
                    markerscale=2
                )
                

            except Exception as e:
                print("⚠️ Could not overlay Excel significance:", e)

        # ---------------------------
        # Heatmap
        # ---------------------------
        im = ax_heat.imshow(
            heatmap_within_rat_data,
            aspect='auto',
            extent=extent2,
            origin='lower',
            cmap=custom_cmap,
            norm=norm,
            interpolation='nearest'
        )
        ax_heat.axvline(0, color="red", lw=1)
        ax_heat.set_xlabel("Time (s)")
        ax_heat.set_xticks([-5,0,5,10,15])
        ax_heat.set_yticks([])  # remove y ticks
        ax_heat.set_ylabel("Subjects")

        # ---------------------------
        # Colorbar
        # ---------------------------
        cbar = fig_stack.colorbar(im, cax=ax_cbar)
        cbar.ax.set_ylabel("")
        
        ax_cbar.set_position([
            ax_heat.get_position().x1 + 0.05,  # small offset to the right
            ax_heat.get_position().y0,
            0.03,                              # width of colorbar
            ax_heat.get_position().height
        ])
        
        ax_cbar.text(
            0.5, 1.12,
            "z-score",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax_cbar.transAxes
        )
        cbar.outline.set_visible(False)

        fig_stack.tight_layout()
        
        safe_save_fig(
            fig_stack,
            os.path.join(fig_output_path, f"{processed_phase}_{rat_str}_{eventname}_stacked")
        )
        plt.show()        
        
    
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------
 
    return {
        # --- Within-rat / per-rat ---
        "per_rat": per_rat_results,
        "per_rat_stats": per_rat_stats,  # cue/approach/AUC per rat
        "per_epoc_stats_within_rat": per_epoc_stats_all,
        "within_animal_epocs_mean": within_animal_epocs_mean,
        "within_animal_epocs_sem": within_animal_epocs_sem,
        "file_level_epoc_latency_count": file_summary_df,

        # --- Across-trial ---
        "combined_epocs_across_trial": combined_epocs_all,
        "combined_epocs_baselined_across_trial": combined_baselined_all,
        "combined_stats_across_trial": combined_stats_all,
        "combined_events_info_across_trial": combined_events_info_trial,

        # --- Across-animal ---
        "across_animal_stats": across_animal_stats,   # time-resolved mean/SEM
        "across_animal_epocs": across_animal_epocs,   # mean trace
        "per_animal_stats": per_animal_stats_df,      # per-rat summary
        "combined_events_info_across_animal": combined_events_info_animal,

        # --- Figures ---
        "fig_all": fig_all,

        # --- Summary stats for convenience ---
        "cue_stats_across_trial": (cue_mean_all, cue_sem_all),
        "approach_stats_across_trial": (approach_mean_all, approach_sem_all),
        "auc_pre_stats_across_trial": (auc_pre_mean_all, auc_pre_sem_all),
        "auc_post_stats_across_trial": (auc_post_mean_all, auc_post_sem_all),

        "cue_stats_across_animal": (cue_mean_animal, cue_sem_animal),
        "approach_stats_across_animal": (approach_mean_animal, approach_sem_animal),
        "auc_pre_stats_across_animal": (auc_pre_mean_animal, auc_pre_sem_animal),
        "auc_post_stats_across_animal": (auc_post_mean_animal, auc_post_sem_animal),
        
        "onset_stats_across_trial": (onset_mean_all, onset_sem_all),
        "slope_stats_across_trial": (slope_mean_all, slope_sem_all),
        "onset_stats_across_animal": (onset_mean_animal, onset_sem_animal),
        "slope_stats_across_animal": (slope_mean_animal, slope_sem_animal),
        
        "time_to_peak_stats_across_trial": (time_to_peak_mean_all, time_to_peak_sem_all),
        "onset_peak_amp_stats_across_trial": (onset_peak_amp_mean_all, onset_peak_amp_sem_all),
        "overall_peak_amp_stats_across_trial": (overall_peak_amp_mean_all, overall_peak_amp_sem_all),
        "overall_peak_time_stats_across_trial": (overall_peak_time_mean_all, overall_peak_time_sem_all),

        "time_to_peak_stats_across_animal": (time_to_peak_mean_animal, time_to_peak_sem_animal),
        "onset_peak_amp_stats_across_animal": (onset_peak_amp_mean_animal, onset_peak_amp_sem_animal),
        "overall_peak_amp_stats_across_animal": (overall_peak_amp_mean_animal, overall_peak_amp_sem_animal),
        "overall_peak_time_stats_across_animal": (overall_peak_time_mean_animal, overall_peak_time_sem_animal),
    }





# concatenate multiple recording segments for stats and plotting ---------------------------------------------------------------------   

# updating documentation 09.12.26

def perievent_analysis(
        *,
        feather_folder=None,
        file_pattern=None,
        feather_files=None,
        new_fs=20.0,
        ts=None,
        trange=None,
        baseline_trange=None,
        cue_window=None,
        auc_pre_window=None,
        auc_post_window=None,
        approach_window=None,
        plot_auc_region=False,
        plot=True,
        per_animal_traces=None,
        per_animal_labels=None,
        subset = None,
        onset_search_window=(0, 3),
        deriv_threshold=0.05,
        deriv_smooth_window=7,
        deriv_smooth_poly=6,
        consecutive_points=2,
        peak_fraction=0.5,
        min_peak_amplitude=-1,
        amp_onset_fraction=0.1,
        plot_onset_detection = False,
        plot_individual_trials = False,
        master_xlim = None,
        master_ylim = None,
        overall_peak_window = (0,8),
        rat_name = None,
        excel_file_path = None,
        plot_group_comparison=False,
        plot_baseline_significance=False,
        paperfigs = False,
        master_figsize = (8, 8)
):
    """
    Loads and concatenates GCaMP_465 epocs feather files, computes statistics,
    plots peri-event responses, AND computes per-epoc baselined window stats.

    Returns
    -------
    (
        combined_epocs,                  # raw concatenated data
        combined_epocs_baselined,        # baseline-subtracted data
        epoc_stats,                      # mean/SEM/std for plotting
        mean_epoc_stream,
        std_epoc_stream,
        sem_epoc_stream,
        fig_trial,
        fig_across,
        fig_within,
        fig_onset,
        fig_trials,
        (cue_mean, cue_sem),
        (auc_pre_mean, auc_pre_sem),
        (auc_post_mean, auc_post_sem),
        (approach_mean, approach_sem),
        (onset_mean, onset_sem),
        (slope_mean, slope_sem),
        (time_to_peak_mean, time_to_peak_sem),
        (onset_peak_amp_mean, onset_peak_amp_sem),
        (overall_peak_amp_mean, overall_peak_amp_sem),
        (overall_peak_time_mean, overall_peak_time_sem),
        per_epoc_stats_df               # per-column stats
    )
    """
    import os, glob, re
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from statsmodels.api import RLM, add_constant
    from scipy.stats import linregress
    from matplotlib.ticker import MultipleLocator

    
    # this is used to standardize ticks in publication figures - may need to be adjusted based on specific data 
    def configure_axis(ax):
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.minorticks_off()

    # ---------------------
    # Gather files
    # ---------------------
    if feather_files is not None:
        feather_files = list(feather_files)
    else:
        if feather_folder is None:
            raise ValueError("Either 'feather_files' or 'feather_folder' must be provided")
        if isinstance(feather_folder, str):
            feather_folder = [feather_folder]
        feather_files = []
        for folder in feather_folder:
            if file_pattern is None:
                raise ValueError("'file_pattern' must be provided if feather_files is None")
            feather_files.extend(glob.glob(os.path.join(folder, file_pattern)))

    if not feather_files:
        raise FileNotFoundError("No feather files found")

    # ---------------------
    # Load and concatenate
    # ---------------------
    epoc_dfs = []
    epoc_rats = []
    epoc_sources_per_df = []

    for f in feather_files:
        df = pd.read_feather(f)
        
        # ---------------------
        # Optional subsetting
        # ---------------------
        if subset is not None:
            n_cols = df.shape[1]

            if isinstance(subset, int):
                if subset > 0:
                    # First N
                    df = df.iloc[:, :min(subset, n_cols)]
                elif subset < 0:
                    # Last N
                    df = df.iloc[:, max(0, n_cols + subset):]

            elif isinstance(subset, (list, tuple)) and len(subset) == 2:
                start, end = subset

                # Convert 1-indexed inclusive to 0-indexed slice
                start_idx = max(start - 1, 0)
                end_idx   = min(end, n_cols)

                if start_idx < end_idx:
                    df = df.iloc[:, start_idx:end_idx]
                else:
                    df = df.iloc[:, 0:0]  # empty safely

            else:
                raise ValueError("subset must be int, negative int, or [start, end]")

        if df.shape[1] == 0:
            continue

        # pathlib .stem extracts the file name, stripping the extension.
        # pathlib treats file paths as objects with associated methods
        fname = Path(f).stem
        
        # for each file in the feather_files list, extract the animal ID 
        rat_match = re.search(r'ACW_coh\d+_IT_[fm]\d+', fname)  # updated this, previously was just "coh5"
        rat_id = rat_match.group(0) if rat_match else "unknown_rat"

        df.columns = [f"{rat_id}_{col}" for col in df.columns]
        
        # populate the epoc_dfs list, becomes a list of dataframes from each file
        # epoc_rats keeps track of the animals being analyzed
        epoc_dfs.append(df)
        epoc_rats.append(rat_id)

        # Track source file for each column, i.e. same file name repeated x number of events
        epoc_sources_per_df.append([Path(f).name] * df.shape[1])
        
    if len(epoc_dfs) == 0:
        raise ValueError("No epocs remaining after subsetting/baseline filtering.")

    # aggregate the list of dataframes from feather_files into a new combined dataframe
    combined_epocs = pd.concat(epoc_dfs, axis=1).reset_index(drop=True)

    # Deduplicate columns if needed, can end up with duplicates
    def dedupe_columns(cols):
        seen = {} # empty dictionary
        out = [] # empty list
        for c in cols:
            if c not in seen:
                seen[c] = 0
                out.append(c)
            else:
                seen[c] += 1
                out.append(f"{c}__{seen[c]}")
        return out

    combined_epocs.columns = dedupe_columns(combined_epocs.columns)

    # ---------------------
    # Baseline correction
    # ---------------------
    if baseline_trange is not None:
        baseline_start_idx = int((baseline_trange[0] - trange[0]) * new_fs)
        baseline_end_idx   = int((baseline_trange[1] - trange[0]) * new_fs)

        filtered_epoc_dfs = []
        filtered_sources = []

        # first, check is enough baseline exists (e.g. event didn't occurr too soon in recording segment)
        for df, src_list in zip(epoc_dfs, epoc_sources_per_df):

            baseline_slice = df.iloc[baseline_start_idx:baseline_end_idx]
            valid_fraction = baseline_slice.notna().sum(axis=0) / len(baseline_slice)
            keep_cols = valid_fraction[valid_fraction >= 0.1].index

            filtered_df = df[keep_cols]
            filtered_epoc_dfs.append(filtered_df)

            # Filter matching sources correctly
            keep_mask = df.columns.isin(keep_cols)
            filtered_sources.extend(np.array(src_list)[keep_mask].tolist())

        if len(filtered_epoc_dfs) == 0 or all(df.shape[1] == 0 for df in filtered_epoc_dfs):
            raise ValueError("No epocs remaining after baseline filtering.")

        combined_epocs = pd.concat(filtered_epoc_dfs, axis=1).reset_index(drop=True)
        combined_epocs.columns = dedupe_columns(combined_epocs.columns)

        baseline_slice = combined_epocs.iloc[baseline_start_idx:baseline_end_idx]
        baselines = baseline_slice.mean()  # mean in the baseline interval, for each column / event
        combined_epocs_baselined = combined_epocs.subtract(baselines, axis=1)
        data_to_use = combined_epocs_baselined

        epoc_sources = filtered_sources  # <- update the sources list
    else:
        combined_epocs_baselined = combined_epocs.copy()
        data_to_use = combined_epocs_baselined
        # No filtering happened, so just flatten the per-file source lists as-is
        epoc_sources = [src for src_list in epoc_sources_per_df for src in src_list]


    # the below index slicing and time vector construction assume that the steam data begins at ts = trange[0]
    # in my typical case, -10 seconds from the event. 
    #    This should be true if the perivent data was preprocessed using my epoc_streams function, 
    #    which takes trange as an input defined as [x, y] where x is the second preceding the event, and y is the total duration 
    #    e.g. [-10, 25] in the preprocessing extracts segments beginning 10 seconds before the event and 15 seconds following the event
    #    the trange defined in the compute_kwargs for the analysis code should match that used in the preprocessing. 
    
    # After baseline correction
    if trange is not None:
        start_idx = max(int((trange[0] - trange[0]) * new_fs), 0)  # 0 offset
        end_idx   = min(int((trange[1] - trange[0]) * new_fs), data_to_use.shape[0])
        data_to_use = data_to_use.iloc[start_idx:end_idx]

    # ---------------------
    # Time vector
    # ---------------------
    epoc_ts = trange[0] + np.arange(len(data_to_use)) / new_fs

    #epoc_ts = np.linspace(trange[0], trange[1], data_to_use.shape[0])

    '''
    Key variables at this point:
    
    epoc_dfs: list of dataframes, one per input file, headers renamed to include rat IDs
    
    epoc_rats, epoc_sources_per_df: lists of rat IDs and source file names
    
    combined_epocs and combined_epocs_baselined: key outputs, dataframes containing all events collected from the input list of files
    
    data_to_use: key variable used for downstream steps, typically a baseline subtracted dataframe with columns corresponding to each event
    
    epoc_ts: the time vector built from trange according to the sampling rate
    
    '''

    # ============================================================
    # 🔬 ONSET + SLOPE DETECTION
    # ============================================================
    
    # this onset and slope detection doesn't work well unless there is a pretty clear and robust positive transient
    
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.tools import add_constant
    from scipy.signal import savgol_filter
    import numpy as np
    
    dt = epoc_ts[1] - epoc_ts[0]
    
    def detect_onset_and_slope(
            y,   # numpy array corresponding to one column of the data_to_use dataframe
            epoc_ts, 
            dt, 
            onset_search_window,
            deriv_threshold,
            deriv_smooth_window,
            deriv_smooth_poly,
            consecutive_points,
            peak_fraction,
            min_peak_amplitude,
            amp_onset_fraction
    ):
        """
        Detect onset, rising slope, time-to-peak, and peak amplitude
        using derivative + amplitude logic with full numerical safeguards.

        Returns
        -------
        onset_time : float
        slope : float
        time_to_peak : float
        peak_val : float
        """

        import numpy as np
        from scipy.signal import savgol_filter
        from statsmodels.robust.robust_linear_model import RLM
        from statsmodels.tools import add_constant

        # -----------------------------
        # Restrict to onset search window
        # -----------------------------
        # use a Numpy booleean array 
        
        mask = (epoc_ts >= onset_search_window[0]) & (epoc_ts <= onset_search_window[1])
        if mask.sum() < 3:
            return np.nan, np.nan, np.nan, np.nan

        y_win = y[mask]
        ts_win = epoc_ts[mask]

        # -----------------------------
        # Remove NaNs early
        # -----------------------------
        if np.any(np.isnan(y_win)):
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Smooth signal for peak detection
        # this lessens influence of noise in the signal
        # -----------------------------
        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y_win, sigma=2)

        # -----------------------------
        # Peak detection (on smoothed signal)
        # -----------------------------
        peak_idx = np.nanargmax(y_smooth) # returns index of the max value
        onset_peak_val = y_smooth[peak_idx]
        onset_peak_time = ts_win[peak_idx]
        
        if onset_peak_val < min_peak_amplitude:
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Validate Savitzky–Golay window
        # -----------------------------
        max_window = len(y_win)
        if max_window % 2 == 0:
            max_window -= 1  # must be odd

        window_length = min(deriv_smooth_window, max_window)

        if window_length <= deriv_smooth_poly:
            return np.nan, np.nan, np.nan, np.nan

        if window_length < 3:
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Compute derivative safely
        # the idea is to compute a smoothed derivative, i.e. rate of change at each time point
        # -----------------------------
        try:
            dydt = savgol_filter(
                y_win,
                window_length=window_length,
                polyorder=deriv_smooth_poly,
                deriv=1,
                delta=dt
            )
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

        if np.all(np.isnan(dydt)):
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Dynamic derivative threshold
        # -----------------------------
        max_dydt = np.nanmax(dydt)
        dynamic_threshold = max(deriv_threshold, 0.1 * max_dydt) 
        above = dydt > dynamic_threshold

        # -----------------------------
        # Derivative-based onset
        # moving window looks for consecutive points that exceed the derivative threshold
        # consecutive points so noisy spikes don't count
        # -----------------------------
        onset_idx_deriv = None
        for i in range(len(above) - consecutive_points + 1):
            if np.all(above[i:i + consecutive_points]):
                onset_idx_deriv = i
                break

        # -----------------------------
        # Amplitude-based onset
        # -----------------------------
        amp_threshold = amp_onset_fraction * onset_peak_val
        onset_candidates_amp = np.where(y_win >= amp_threshold)[0]
        onset_idx_amp = onset_candidates_amp[0] if len(onset_candidates_amp) > 0 else None

        # -----------------------------
        # Combine logic
        # whichever happens first, derivative or amplitude-based onset, is used for the onset time
        # -----------------------------
        if onset_idx_deriv is None and onset_idx_amp is None:
            return np.nan, np.nan, np.nan, np.nan

        if onset_idx_deriv is None:
            onset_idx = onset_idx_amp
        elif onset_idx_amp is None:
            onset_idx = onset_idx_deriv
        else:
            onset_idx = min(onset_idx_deriv, onset_idx_amp)

        onset_time = ts_win[onset_idx]

        # -----------------------------
        # Time-to-peak
        # -----------------------------
        time_to_peak = onset_peak_time - onset_time

        if time_to_peak <= 0:
            return onset_time, np.nan, np.nan, onset_peak_val

        # -----------------------------
        # Rising slope fit window
        # -----------------------------
        threshold_val = peak_fraction * onset_peak_val
        above_thresh = np.where(y_win >= threshold_val)[0]
        valid_end = above_thresh[above_thresh > onset_idx]
        end_idx = valid_end[0] if len(valid_end) > 0 else peak_idx

        if end_idx <= onset_idx:
            return onset_time, np.nan, time_to_peak, onset_peak_val

        x_fit = ts_win[onset_idx:end_idx + 1]
        y_fit = y_win[onset_idx:end_idx + 1]

        if len(x_fit) < 3:
            return onset_time, np.nan, time_to_peak, onset_peak_val

        # -----------------------------
        # Robust linear fit
        # this comes from statsmodels
        # -----------------------------
        try:
            X = add_constant(x_fit)
            rlm_model = RLM(y_fit, X)  # constructs a robust linear model, should downweight noisy outliers in the data
            slope = rlm_model.fit().params[1]  # performs the fit, returns results, from which the slope is extracted
        except Exception:
            slope = np.nan

        return onset_time, slope, time_to_peak, onset_peak_val

    # dictionaries, eventually populated with one entry per event
    onset_dict = {}
    slope_dict = {}
    time_to_peak_dict = {}
    onset_peak_amp_dict = {}
    
    for col in data_to_use.columns:
        y = data_to_use[col].values  # for each event/column, extract .values (pandas series --> numpy array)
        onset, slope, ttp, onset_peak_amp = detect_onset_and_slope(
            y, epoc_ts, dt,
            onset_search_window=onset_search_window,
            deriv_threshold=deriv_threshold,
            deriv_smooth_window=deriv_smooth_window,
            deriv_smooth_poly=deriv_smooth_poly,
            consecutive_points=consecutive_points,
            peak_fraction=peak_fraction,
            min_peak_amplitude=min_peak_amplitude,
            amp_onset_fraction=amp_onset_fraction
        )
        
        # dictionaries consist of {column name: value} pairings
        # these four dictionaries will end up populating the per_epoc_stats_df below
        onset_dict[col] = onset
        slope_dict[col] = slope
        time_to_peak_dict[col] = ttp
        onset_peak_amp_dict[col] = onset_peak_amp


    # ============================================================
    # 🔬 OVERALL PEAK AMPLITUDE (within defined window)
    # ============================================================
    
    overall_peak_amp_dict = {}
    overall_peak_time_dict = {}

    if overall_peak_window is not None:
        w_start, w_end = overall_peak_window
        win_mask = (epoc_ts >= w_start) & (epoc_ts <= w_end)

        for col in data_to_use.columns:
            y = data_to_use[col].values
            y_win = y[win_mask]
            ts_win = epoc_ts[win_mask]

            if len(y_win) == 0 or np.all(np.isnan(y_win)):
                overall_peak_amp_dict[col] = np.nan
                overall_peak_time_dict[col] = np.nan
                continue

            peak_idx = np.nanargmax(y_win)
            overall_peak_amp_dict[col] = y_win[peak_idx]
            overall_peak_time_dict[col] = ts_win[peak_idx]
    else:
        for col in data_to_use.columns:
            overall_peak_amp_dict[col] = np.nan
            overall_peak_time_dict[col] = np.nan
    
    
    # ---------------------
    # Per-epoc stats
    # ---------------------
    per_epoc_stats_df = pd.DataFrame(index=data_to_use.columns)
    per_epoc_stats_df.index.name = "epoc_id"

    # Map onset/slope
    per_epoc_stats_df["response_latency_sec"] = [onset_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["rising_slope"] = [slope_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["time_to_peak_sec"] = [time_to_peak_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["onset_peak_amplitude"] = [onset_peak_amp_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["overall_peak_amp"] = [overall_peak_amp_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["overall_peak_amp_timestamp"] = [overall_peak_time_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]

    # ---------------------
    
    # compute mean + SEM over current set of events
    # in tier 1 of the run_perievent_analysis function, these are effectively per-rat values
    # in tier 2, would be across all trials/rats
    # ---------------------
    
    latency_vals = per_epoc_stats_df["response_latency_sec"].values
    slope_vals = per_epoc_stats_df["rising_slope"].values
    time_to_peak_vals = per_epoc_stats_df["time_to_peak_sec"].values
    onset_peak_amp_vals = per_epoc_stats_df["onset_peak_amplitude"].values
    

    onset_mean = np.nanmean(latency_vals)
    onset_sem = stats.sem(latency_vals, nan_policy="omit")

    slope_mean = np.nanmean(slope_vals)
    slope_sem = stats.sem(slope_vals, nan_policy="omit")
    
    time_to_peak_mean = np.nanmean(time_to_peak_vals)
    time_to_peak_sem = stats.sem(time_to_peak_vals, nan_policy="omit")

    onset_peak_amp_mean = np.nanmean(onset_peak_amp_vals)
    onset_peak_amp_sem = stats.sem(onset_peak_amp_vals, nan_policy="omit")

    overall_peak_vals = per_epoc_stats_df["overall_peak_amp"].values
    overall_peak_time_vals = per_epoc_stats_df["overall_peak_amp_timestamp"].values

    overall_peak_amp_mean = np.nanmean(overall_peak_vals)
    overall_peak_amp_sem  = stats.sem(overall_peak_vals, nan_policy="omit")

    overall_peak_time_mean = np.nanmean(overall_peak_time_vals)
    overall_peak_time_sem  = stats.sem(overall_peak_time_vals, nan_policy="omit")

    per_rat_onset_stats = {
        "latency_mean": onset_mean,
        "latency_sem": onset_sem,
        "slope_mean": slope_mean,
        "slope_sem": slope_sem,
        "time_to_peak_mean": time_to_peak_mean,
        "time_to_peak_sem":  time_to_peak_sem,
        "onset_peak_amp_mean": onset_peak_amp_mean,
        "onset_peak_amp_sem": onset_peak_amp_sem,
        "overall_peak_amp_mean": overall_peak_amp_mean,
        "overall_peak_amp_sem": overall_peak_amp_sem,
        "overall_peak_time_mean": overall_peak_time_mean,
        "overall_peak_time_sem": overall_peak_time_sem,
    }

    # ============================================================
    # OPTIONAL VISUALIZATION
    # ============================================================
    # ============================================================
    # 📊 INDIVIDUAL TRIAL + PEAK TIMING PLOT (PER ANIMAL)
    # ============================================================

    fig_trials = None

    
    if plot_individual_trials:

        fig_trials, ax1 = plt.subplots(figsize=master_figsize if paperfigs else (11,6))

        if paperfigs:
            configure_axis(ax1)
            
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        n_trials = len(data_to_use.columns)
        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0, vmax=n_trials - 1)

        peak_times = []
        peak_vals  = []

        for i, col in enumerate(data_to_use.columns):

            trace = data_to_use[col].values

            # Plot trace
            ax1.plot(
                epoc_ts,
                trace,
                color=cmap(norm(i)),
                alpha=0.8,
                linewidth=1
            )

            # Get peak time from your existing table
            peak_time = overall_peak_time_dict[col]
            peak_val  = overall_peak_amp_dict[col]

            peak_times.append(peak_time)
            peak_vals.append(peak_val)

            # Plot peak marker
            ax1.scatter(
                peak_time,
                peak_val,
                color=cmap(norm(i)),
                edgecolor="white",
                s=80,
                zorder=5
            )

        # Mean trace
        mean_epoc_stream_plot = np.nanmean(data_to_use, axis=1)
        ax1.plot(
            epoc_ts,
            mean_epoc_stream_plot,
            color="red",
            linewidth=3,
            label="Mean"
        )

        ax1.axvline(0, color="red", linestyle="--", linewidth=1, zorder=0)

        # Optional: annotate jitter (SD of peak times)
        peak_sd = np.nanstd(peak_times)
        ax1.text(
            0.02, 0.95,
            f"Peak Time SD = {peak_sd:.3f}s",
            transform=ax1.transAxes,
            verticalalignment="top"
        )

        title_name = rat_name if rat_name is not None else "Animal"
        ax1.set_title(f"{title_name} - Individual Trials with Peaks")

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Signal")

        plt.tight_layout()

    fig_onset = None

    if plot_onset_detection and per_animal_traces is not None and len(per_animal_traces) > 0:
        with plt.ioff():  # prevent automatic display
            fig_onset, ax = plt.subplots(figsize=(8,6))
            for col in data_to_use.columns:
                y = data_to_use[col].values
                ax.plot(epoc_ts, y, alpha=0.3)

                onset = onset_dict[col]
                slope = slope_dict[col]

                if not np.isnan(onset) and not np.isnan(slope):
                    ax.axvline(onset, linestyle='--', alpha=0.5)
                    ax.axvline(0, linestyle="-", color="red", label="event")

                    y0 = np.interp(onset, epoc_ts, y)
                    x_line = np.linspace(onset, onset + 1.0, 50)
                    y_line = y0 + slope * (x_line - onset)
                    ax.plot(x_line, y_line, linewidth=2)

            ax.set_title("Response onset and slope detection")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Fluorescence (z-score)")
            ax.set_xlim(-0.5,3)

            
    # ---------------------
    # Mean / std / SEM of the perievent data
    # ---------------------
    n_trials = data_to_use.shape[1]
    
    mean_epoc_stream = np.nanmean(data_to_use, axis=1)

    if n_trials > 1:
        std_epoc_stream = np.nanstd(data_to_use, axis=1, ddof=1)
        sem_epoc_stream = stats.sem(data_to_use, axis=1, nan_policy='omit')
    else:
        std_epoc_stream = np.zeros_like(mean_epoc_stream)
        sem_epoc_stream = np.zeros_like(mean_epoc_stream)

    # ---------------------
    # Window-level stats
    # ---------------------
    def window_stats(window):
        if window is None:
            return np.nan, np.nan
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        wdata = data_to_use.iloc[start_idx:end_idx]
        if wdata.empty:
            return np.nan, np.nan
        mean_val = wdata.mean().mean()
        sem_val  = stats.sem(wdata.mean(), nan_policy='omit')
        return mean_val, sem_val

    cue_mean, cue_sem = window_stats(cue_window)
    approach_mean, approach_sem = window_stats(approach_window)

    # ---------------------
    # AUC
    # ---------------------
    def compute_auc(window):
        if window is None:
            return np.nan, np.nan
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        auc_vals = []
        for col in data_to_use.columns:
            y = data_to_use[col].iloc[start_idx:end_idx].values
            x = epoc_ts[start_idx:end_idx]
            finite_mask = np.isfinite(y)  # double check, why the np.isfinite step? 
            if finite_mask.sum() < 2:
                continue
            auc = np.trapz(y[finite_mask], x[finite_mask])  # this will eventually need to be updated to np.trapezoid I think
            if np.isfinite(auc):
                auc_vals.append(auc)
        if len(auc_vals) == 0:
            return np.nan, np.nan
        auc_vals = np.array(auc_vals)
        mean_auc = np.mean(auc_vals)
        sem_auc  = stats.sem(auc_vals) if len(auc_vals) > 1 else np.nan
        return mean_auc, sem_auc

    auc_pre_mean, auc_pre_sem   = compute_auc(auc_pre_window)
    auc_post_mean, auc_post_sem = compute_auc(auc_post_window)

    # ---------------------
    # Per-epoc stats
    # ---------------------
    def per_epoc_window_mean(window):
        if window is None:
            return {}
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        return {col: data_to_use[col].iloc[start_idx:end_idx].mean() for col in data_to_use.columns}

    def per_epoc_auc(window):
        if window is None:
            return {}
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        out = {}
        for col in data_to_use.columns:
            y = data_to_use[col].iloc[start_idx:end_idx].values
            x = epoc_ts[start_idx:end_idx]
            if len(x) == len(y) and np.any(np.isfinite(y)):
                out[col] = float(np.trapz(y, x))
        return out

    
    # Map window means
    cue_vals = per_epoc_window_mean(cue_window)
    approach_vals = per_epoc_window_mean(approach_window)
    auc_pre_vals = per_epoc_auc(auc_pre_window)
    auc_post_vals = per_epoc_auc(auc_post_window)

    # further build the per_epoc dataframe created above containing onset, peak, and slope measures
    # now add the mean and AUC values for identified windows, for each event
    
    per_epoc_stats_df["cue_mean"] = [cue_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["approach_mean"] = [approach_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["auc_pre"] = [auc_pre_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["auc_post"] = [auc_post_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]

    # Add source file
    per_epoc_stats_df["source_file"] = epoc_sources

    # Reset index for epoc_id as a column
    per_epoc_stats_df = per_epoc_stats_df.reset_index()
    
    # ---------------------
    # Stats DataFrame (time-resolved), aka the perievent mean and SEM trace
    # ---------------------
    epoc_stats = pd.DataFrame({
        "mean_epoc_stream": mean_epoc_stream,
        "std_epoc_stream": std_epoc_stream,
        "sem_epoc_stream": sem_epoc_stream
    })

    
    
    
    # ---------------------
    # Plotting (optional)
    # ---------------------
    fig_trial = None
    fig_across = None
    fig_within = None

    mean_across_animal = None
    sem_across_animal  = None
    
    if paperfigs:
        # use consistent color maps, line widths, alpha
        mean_color = np.array([36, 106, 72]) / 255         # across-trial mean
        sem_alpha = 0.4
        cue_color = 'blue'
        approach_color = 'magenta'
        auc_pre_color = 'blue'
        auc_post_color = 'magenta'
        line_width_mean = 2
        line_width_individual = 1.5
    else:
        # legacy or interactive defaults
        mean_color = 'yellow'
        sem_alpha = 0.3
        line_width_mean = 2
        line_width_individual = 1.5


    if per_animal_traces is not None and len(per_animal_traces) > 0:
        all_means = np.array([trace["mean"] for trace in per_animal_traces])
        mean_across_animal = np.nanmean(all_means, axis=0)
        sem_across_animal  = np.nanstd(all_means, axis=0, ddof=1) / np.sqrt(all_means.shape[0])

    if plot:
        import matplotlib.cm as cm

        # ------------------------------------------------------------
        # FIGURE 1 — ACROSS-TRIAL
        # ------------------------------------------------------------
        fig_trial, ax = plt.subplots(1, 1, figsize=master_figsize if paperfigs else (11,6))
        ax.axvline(0, color='r', linewidth=1, zorder=0)
        ax.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8, zorder=0)
        
        if paperfigs:
            configure_axis(ax)

        ax.plot(epoc_ts, mean_epoc_stream, color=mean_color, linewidth=line_width_mean, label='Mean response', zorder=11)
        ax.fill_between(epoc_ts,
                        mean_epoc_stream - sem_epoc_stream,
                        mean_epoc_stream + sem_epoc_stream,
                        color=mean_color, alpha=sem_alpha, linewidth=0,
                        edgecolor='none', label='SEM', zorder=10)

        # Y-limits
        k = 8
        y_min = np.nanmin(mean_epoc_stream - k * sem_epoc_stream)
        y_max = np.nanmax(mean_epoc_stream + k * sem_epoc_stream)
        ax.set_ylim(y_min, y_max)
        top_bar = y_max * 0.95

        # Cue and approach bars
        if not paperfigs:
            for window, color, label in zip([cue_window, approach_window],
                                            ['yellow', 'magenta'],
                                            ['cue', 'approach']):
                if window is not None:
                    start, end = window
                    x = epoc_ts[(epoc_ts >= start) & (epoc_ts <= end)]
                    if x.size:
                        ax.fill_between(x, top_bar*0.98, top_bar, color=color, alpha=0.7)
                        ax.text(np.mean(x), top_bar*1.01, label, color=color,
                                ha='center', va='bottom', fontsize=8)

        # AUC shading
        if plot_auc_region:
            def safe_fill(window, color):
                start, end = window
                i0 = np.searchsorted(epoc_ts, start)
                i1 = np.searchsorted(epoc_ts, end)
                if i0 >= i1:
                    return
                y = mean_epoc_stream[i0:i1]
                if np.all(np.isnan(y)):
                    return
                y = pd.Series(y).interpolate(limit_direction='both').to_numpy()
                ax.fill_between(epoc_ts[i0:i1], 0, y, color=color, alpha=0.5, linewidth=0, edgecolor='none')

            if auc_pre_window is not None:
                safe_fill(auc_pre_window, 'green')
            if auc_post_window is not None:
                safe_fill(auc_post_window, 'blue')

        ax.set_title('Across-trial mean ± SEM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence (z-score)')
        ax.set_xlim(master_xlim)
        ax.set_ylim (master_ylim)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)


        # ------------------------------------------------------------
        # FIGURE 2 — ACROSS-ANIMAL
        # ------------------------------------------------------------
        fig_across, ax = plt.subplots(1, 1, figsize=master_figsize if paperfigs else (11,6))
        ax.axvline(0, color='r', linewidth=1, zorder=0)
        ax.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8, zorder=0)

        if paperfigs:
            configure_axis(ax)
            
        if mean_across_animal is not None:
            ax.plot(epoc_ts, mean_across_animal, color=mean_color, linewidth=line_width_mean, label='Mean across subjects')
            ax.fill_between(epoc_ts,
                            mean_across_animal - sem_across_animal,
                            mean_across_animal + sem_across_animal,
                            color=mean_color, alpha=sem_alpha, linewidth=0,
                            edgecolor='none', label='SEM')
        else:
            ax.plot(epoc_ts, mean_epoc_stream, color='yellow', linewidth=3, label='Mean response')
            ax.fill_between(epoc_ts,
                            mean_epoc_stream - sem_epoc_stream,
                            mean_epoc_stream + sem_epoc_stream,
                            color='yellow', alpha=0.4, linewidth=0,
                            edgecolor='none', label='SEM')

            
        # -----------------------------
        # Optional: Overlay Excel significance markers
        # -----------------------------
        # ==========================================================
        # OPTIONAL SIGNIFICANCE OVERLAY (Fully Generalized)
        # ==========================================================
        if excel_file_path is not None:
            try:
                sig_excel = pd.read_excel(excel_file_path, sheet_name=None)

                y_top = master_ylim[1]
                vertical_spacing = 0.05  # spacing between stacked marker rows

                # --------------------------------------------------
                # GROUP COMPARISONS (auto-detect sheets with "vs")
                # --------------------------------------------------
                if plot_group_comparison:

                    comparison_sheets = [
                        name for name in sig_excel.keys()
                        if "vs" in name.lower()
                    ]

                    n_sheets = len(comparison_sheets)
                    # Create a gradient from magenta to dark gray
                    start_color = np.array([1.0, 0.0, 1.0])  # magenta RGB
                    end_color = np.array([0.3, 0.3, 0.3])    # dark gray RGB
                    colors = [
                        start_color + (end_color - start_color) * (i / max(n_sheets - 1, 1))
                        for i in range(n_sheets)
                    ]
                    colors = [tuple(c) for c in colors]  # convert to tuples for matplotlib

                    for i, sheet_name in enumerate(comparison_sheets):

                        sig_vals = sig_excel[sheet_name].iloc[:, 0].to_numpy()

                        if len(sig_vals) > 1:

                            sig_times = np.linspace(
                                epoc_ts.min(),
                                epoc_ts.max(),
                                len(sig_vals)
                            )

                            interp_vals = np.interp(epoc_ts, sig_times, sig_vals)
                            mask = interp_vals > 0.5

                            height = y_top * (0.98 - i * vertical_spacing)

                            ax.scatter(
                                epoc_ts[mask],
                                height * np.ones(np.sum(mask)),
                                marker='o',
                                color=colors[i],
                                linewidths=0,
                                edgecolors='none',
                                s=7 if paperfigs else 100,
                                label=sheet_name
                            )

                # --------------------------------------------------
                # BASELINE SIGNIFICANCE (auto-detect columns)
                # --------------------------------------------------
                if plot_baseline_significance and "Significance From Baseline" in sig_excel:

                    sig_base = sig_excel["Significance From Baseline"]

                    # Color cycle for unlimited groups
                    cmap = plt.get_cmap("tab10")

                    for j, col in enumerate(sig_base.columns):

                        sig_vals = sig_base[col].to_numpy()

                        if len(sig_vals) > 1:

                            sig_times = np.linspace(
                                epoc_ts.min(),
                                epoc_ts.max(),
                                len(sig_vals)
                            )

                            interp_vals = np.interp(epoc_ts, sig_times, sig_vals)
                            mask = interp_vals > 0.5

                            # Stack below comparison markers
                            baseline_offset = len(comparison_sheets) * vertical_spacing
                            height = y_top * (0.98 - baseline_offset - j * vertical_spacing)

                            ax.scatter(
                                epoc_ts[mask],
                                height * np.ones(np.sum(mask)),
                                marker='o',
                                color=cmap(j % 10),
                                linewidths=0,
                                edgecolors='none',
                                s=7 if paperfigs else 100,
                                label=f"{col} baseline"
                            )

                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)

            except Exception as e:
                print("⚠️ Could not overlay Excel significance:", e)
                print("epoc_ts length:", len(epoc_ts))
        
        if not paperfigs:    
            # Cue and approach bars
            for window, color, label in zip([cue_window, approach_window],
                                            ['yellow', 'magenta'],
                                            ['cue', 'approach']):
                if window is not None:
                    start, end = window
                    x = epoc_ts[(epoc_ts >= start) & (epoc_ts <= end)]
                    if x.size:
                        ax.fill_between(x, top_bar*0.98, top_bar, color=color, alpha=0.7)
                        ax.text(np.mean(x), top_bar*1.01, label, color=color,
                                ha='center', va='bottom', fontsize=8)

        # AUC shading
        if plot_auc_region:

            def safe_fill(window, color):
                start, end = window
                i0 = np.searchsorted(epoc_ts, start)
                i1 = np.searchsorted(epoc_ts, end)
                if i0 >= i1:
                    return

                # Use across-animal mean if available
                if mean_across_animal is not None:
                    y_source = mean_across_animal
                else:
                    y_source = mean_epoc_stream

                y = y_source[i0:i1]

                if np.all(np.isnan(y)):
                    return

                y = pd.Series(y).interpolate(limit_direction='both').to_numpy()
                ax.fill_between(epoc_ts[i0:i1], 0, y, color=color, alpha=0.5, linewidth=0,
                        edgecolor='none')

            if auc_pre_window is not None:
                safe_fill(auc_pre_window, 'green')
            if auc_post_window is not None:
                safe_fill(auc_post_window, 'blue')


        ax.set_ylim(y_min, y_max)
        ax.set_title('Across-animal mean ± SEM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence (z-score)')
        ax.set_xlim(master_xlim)
        ax.set_ylim(master_ylim)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)


        # ------------------------------------------------------------
        # FIGURE 3 — WITHIN-ANIMAL
        # ------------------------------------------------------------
        fig_within = None
        if per_animal_traces is not None and len(per_animal_traces) > 0:
            fig_within, ax = plt.subplots(1, 1, figsize=master_figsize if paperfigs else (11,6))
            ax.axvline(0, color='r', linewidth=1, zorder=0)
            ax.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8, zorder=0)
            
            if paperfigs:
                configure_axis(ax)
            
            # Plot per-animal traces
       
            cmap_name = "viridis"   # <- easily change this
            cmap = cm.get_cmap(cmap_name)
            colors = cmap(np.linspace(0.1, 0.9, len(per_animal_traces)))
            
            for trace, c in zip(per_animal_traces, colors):
                mean = trace['mean']
                sem  = trace['sem']
                rat  = trace['rat']
                ax.plot(epoc_ts, mean, color=c, linewidth=line_width_individual, alpha=0.9, label=rat)
                ax.fill_between(epoc_ts, mean - sem, mean + sem, color=c, alpha=sem_alpha, linewidth=0,
                edgecolor='none',)

                
            # -----------------------------
            # Optional: Overlay Excel significance markers
            # -----------------------------
            # ==========================================================
            # OPTIONAL SIGNIFICANCE OVERLAY (Fully Generalized)
            # ==========================================================
            if excel_file_path is not None:
                try:
                    sig_excel = pd.read_excel(excel_file_path, sheet_name=None)

                    y_top = master_ylim[1]
                    vertical_spacing = 0.05  # spacing between stacked marker rows

                    # --------------------------------------------------
                    # GROUP COMPARISONS (auto-detect sheets with "vs")
                    # --------------------------------------------------
                    if plot_group_comparison:

                        comparison_sheets = [
                            name for name in sig_excel.keys()
                            if "vs" in name.lower()
                        ]

                        n_sheets = len(comparison_sheets)
                        # Create a gradient from magenta to dark gray
                        start_color = np.array([1.0, 0.0, 1.0])  # magenta RGB
                        end_color = np.array([0.3, 0.3, 0.3])    # dark gray RGB
                        colors = [
                            start_color + (end_color - start_color) * (i / max(n_sheets - 1, 1))
                            for i in range(n_sheets)
                        ]
                        colors = [tuple(c) for c in colors]  # convert to tuples for matplotlib

                        for i, sheet_name in enumerate(comparison_sheets):

                            sig_vals = sig_excel[sheet_name].iloc[:, 0].to_numpy()

                            if len(sig_vals) > 1:

                                sig_times = np.linspace(
                                    epoc_ts.min(),
                                    epoc_ts.max(),
                                    len(sig_vals)
                                )

                                interp_vals = np.interp(epoc_ts, sig_times, sig_vals)
                                mask = interp_vals > 0.5

                                height = y_top * (0.98 - i * vertical_spacing)

                                ax.scatter(
                                    epoc_ts[mask],
                                    height * np.ones(np.sum(mask)),
                                    marker='o',
                                    color=colors[i],
                                    linewidths=0,
                                    edgecolors='none',
                                    s=7 if paperfigs else 100,
                                    label=sheet_name
                                )

                    # --------------------------------------------------
                    # BASELINE SIGNIFICANCE (auto-detect columns)
                    # --------------------------------------------------
                    if plot_baseline_significance and "Significance From Baseline" in sig_excel:

                        sig_base = sig_excel["Significance From Baseline"]

                        # Color cycle for unlimited groups
                        cmap = plt.get_cmap("tab10")

                        for j, col in enumerate(sig_base.columns):

                            sig_vals = sig_base[col].to_numpy()

                            if len(sig_vals) > 1:

                                sig_times = np.linspace(
                                    epoc_ts.min(),
                                    epoc_ts.max(),
                                    len(sig_vals)
                                )

                                interp_vals = np.interp(epoc_ts, sig_times, sig_vals)
                                mask = interp_vals > 0.5

                                # Stack below comparison markers
                                baseline_offset = len(comparison_sheets) * vertical_spacing
                                height = y_top * (0.98 - baseline_offset - j * vertical_spacing)

                                ax.scatter(
                                    epoc_ts[mask],
                                    height * np.ones(np.sum(mask)),
                                    marker='o',
                                    color=cmap(j % 10),
                                    linewidths=0,
                                    edgecolors='none',
                                    s=7 if paperfigs else 100,
                                    label=f"{col} baseline"
                                )

                    ax.legend(loc='upper right', fontsize=10, markerscale=2)

                except Exception as e:
                    print("⚠️ Could not overlay Excel significance:", e)
                    print("epoc_ts length:", len(epoc_ts))

            # -----------------------------
            if not paperfigs:
                # Add cue/approach bars
                # -----------------------------
                top_bar = np.nanmax([trace["mean"] + trace["sem"] for trace in per_animal_traces])
                for window, color, label in zip([cue_window, approach_window],
                                                ['yellow', 'magenta'],
                                                ['cue', 'approach']):
                    if window is not None:
                        start, end = window
                        x = epoc_ts[(epoc_ts >= start) & (epoc_ts <= end)]
                        if x.size:
                            ax.fill_between(x, top_bar*0.98, top_bar, color=color, alpha=0.7, linewidth=0, edgecolor='none')
                            ax.text(np.mean(x), top_bar*1.01, label, color=color,
                                    ha='center', va='bottom', fontsize=8)

            ax.set_title('Within-animal mean ± SEM')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Fluorescence (z-score)')
            ax.set_xlim(master_xlim)
            ax.set_ylim(master_ylim)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)

    return (
        combined_epocs,
        combined_epocs_baselined,
        epoc_stats,
        mean_epoc_stream,
        std_epoc_stream,
        sem_epoc_stream,
        fig_trial,
        fig_across,
        fig_within,
        fig_onset,
        fig_trials,
        (cue_mean, cue_sem),
        (auc_pre_mean, auc_pre_sem),
        (auc_post_mean, auc_post_sem),
        (approach_mean, approach_sem),
        (onset_mean, onset_sem),
        (slope_mean, slope_sem),
        (time_to_peak_mean, time_to_peak_sem),
        (onset_peak_amp_mean, onset_peak_amp_sem),
        (overall_peak_amp_mean, overall_peak_amp_sem),
        (overall_peak_time_mean, overall_peak_time_sem),
        per_epoc_stats_df
    )







# above, functions reviewed    
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################




# combined analysis of peaks from multiple folders, files --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# udpated 02.23.26 to better handle multiple folders

def peaks_process_multi_folders_patterns(
    folders,
    file_patterns,
    preproc_info_txt_pattern,
    filename_prefix_base="combined_",
    target_rat=None,
    rebaseline_window=None,
    save_base=None
):
    """
    True pooled processing across multiple folders.

    Behavior:
    - If save_base is provided → always save there.
    - If save_base is None and one folder → save to folder/GCaMP_peaks_processed.
    - If save_base is None and multiple folders → error.
    - Multiple folders are truly pooled (same rat_ID merged).
    """

    from pathlib import Path
    import pandas as pd
    import shutil
    import tempfile

    # ----------------------------
    # Normalize inputs
    # ----------------------------
    if isinstance(folders, str):
        folders = [folders]

    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    multi_folder_mode = len(folders) > 1

    # ----------------------------
    # Determine save location
    # ----------------------------
    if save_base is not None:
        save_folder = Path(save_base)
    else:
        if multi_folder_mode:
            raise ValueError(
                "Multiple folders provided but save_base is None. "
                "Please specify save_base."
            )
        #save_folder = Path(folders[0]) / "GCaMP_peaks_processed"

    save_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # SINGLE FOLDER CASE
    # ------------------------------------------------------------
    if not multi_folder_mode:

        folder = Path(folders[0])

        amp_pattern = f"{file_patterns[0]}_peak_amps.feather"
        auc_pattern = f"{file_patterns[0]}_peak_aucs.feather"
        info_pattern = f"{file_patterns[0]}_peaks_info.feather"
        baselined_peak_pattern = f"{file_patterns[0]}_baselined_peak_df.feather"

        per_rat_outputs, combined_outputs_all = load_combine_and_save_peak_data_multi(
            folder,
            amp_pattern,
            auc_pattern,
            info_pattern,
            filename_prefix=f"{filename_prefix_base}",
            preproc_info_txt_pattern=preproc_info_txt_pattern,
            baselined_peak_pattern=baselined_peak_pattern,
            save_folder=save_folder,
            rebaseline_window=rebaseline_window
        )

    # ------------------------------------------------------------
    # MULTI-FOLDER TRUE POOLING
    # ------------------------------------------------------------
    else:

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            print("\n🔄 Pooling files across folders...")

            for folder in folders:
                folder = Path(folder)

                for pattern in file_patterns:
                    for suffix in [
                        "_peak_amps.feather",
                        "_peak_aucs.feather",
                        "_peaks_info.feather",
                        "_baselined_peak_df.feather"
                    ]:
                        for file in folder.glob(f"{pattern}{suffix}"):
                            shutil.copy(file, temp_dir / file.name)

                # copy preprocessing txt files
                for txt in folder.glob(preproc_info_txt_pattern):
                    shutil.copy(txt, temp_dir / txt.name)

            amp_pattern = f"{file_patterns[0]}_peak_amps.feather"
            auc_pattern = f"{file_patterns[0]}_peak_aucs.feather"
            info_pattern = f"{file_patterns[0]}_peaks_info.feather"
            baselined_peak_pattern = f"{file_patterns[0]}_baselined_peak_df.feather"

            per_rat_outputs, combined_outputs_all = load_combine_and_save_peak_data_multi(
                temp_dir,
                amp_pattern,
                auc_pattern,
                info_pattern,
                filename_prefix=f"{filename_prefix_base}",
                preproc_info_txt_pattern=preproc_info_txt_pattern,
                baselined_peak_pattern=baselined_peak_pattern,
                save_folder=save_folder,
                rebaseline_window=rebaseline_window
            )

    # ------------------------------------------------------------
    # Build per-rat summary dataframe
    # ------------------------------------------------------------
    all_per_rat_stats = {
        rat: summary_df.set_index("peak measurements")["value"]
        for rat, (summary_df, *_rest) in per_rat_outputs.items()
    }

    per_rat_df = pd.DataFrame(all_per_rat_stats) if all_per_rat_stats else pd.DataFrame()

    # ------------------------------------------------------------
    # Across-rat mean ± SEM
    # ------------------------------------------------------------
    if not per_rat_df.empty:
        summary_all_df = pd.DataFrame({
            "value_mean_across_rats": per_rat_df.mean(axis=1),
            "value_sem_across_rats": per_rat_df.sem(axis=1)
        }, index=per_rat_df.index)
    else:
        summary_all_df = pd.DataFrame()

    return per_rat_df, summary_all_df, combined_outputs_all


# combined analysis of peaks from multiple time segments --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# updated 01.23.26 to save the per-rat organized dataframe of all peaks for later mixed model analysis
# updated 01.29.26 to include csv saving
        
def load_combine_and_save_peak_data_multi(
    folder,
    amp_pattern,
    auc_pattern,
    info_pattern,
    filename_prefix,
    preproc_info_txt_pattern,
    baselined_peak_pattern,
    overwrite=True,
    save_folder=None,
    rebaseline_window = None
):
    import pandas as pd
    import numpy as np
    import os
    import glob
    import re
    import matplotlib.pyplot as plt
    from scipy import stats

    # --- Set save folder ---
    if save_folder is None:
        save_folder = folder  # fallback to original folder
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # --- Helper: extract sampling rate ---
    def extract_fs_from_txt(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if line.lower().startswith("new fs after downsample"):
                    fs = float(line.strip().split(":")[1].strip())
                    return fs
        raise ValueError(f"Sampling rate not found in {txt_path}")

    # --- Safe save helpers ---
    def safe_save(df, path, desc="DataFrame"):
        df = df.copy()
        df.columns = df.columns.astype(str)
        if df.index.name is not None or not df.index.is_integer():
            df = df.reset_index(drop=True)
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.to_feather(path)
        print(f"💾 Saved {desc} to:\n  {path}")

    def safe_save_table(df, base_path, desc="DataFrame"):
        if df is None or df.empty:
            return False
        feather_path = base_path.with_suffix(".feather")
        csv_path     = base_path.with_suffix(".csv")

        # Make a Feather-safe copy
        df_to_feather = df.copy()
        if not isinstance(df_to_feather.index, pd.RangeIndex):
            df_to_feather = df_to_feather.reset_index(drop=True)  # <-- fix

        # Save Feather
        df_to_feather.to_feather(feather_path)

        # Save CSV (keep index)
        df.to_csv(csv_path, index=True)
        print(f"💾 Saved {desc} to:\n  {feather_path}\n  {csv_path}")

        return True


    
    def print_file_list(title, files, indent=2):
        print(f"\n📂 {title} ({len(files)} file(s)):")
        if not files:
            print(" " * indent + "— none —")
        else:
            for f in sorted(files):
                print(" " * indent + os.path.basename(f))


    # --- Find files ---
    amp_files = glob.glob(os.path.join(folder, amp_pattern))
    auc_files = glob.glob(os.path.join(folder, auc_pattern))
    info_files = glob.glob(os.path.join(folder, info_pattern))
    baselined_peak_files = glob.glob(os.path.join(folder, baselined_peak_pattern))
    preproc_info_txt_files = glob.glob(os.path.join(folder, preproc_info_txt_pattern))

    print("\n================ FILES USED IN ANALYSIS ================")
    print(f"📁 Folder: {folder}")
    print(f"🔎 amp_pattern: {amp_pattern}")
    print(f"🔎 auc_pattern: {auc_pattern}")
    print(f"🔎 info_pattern: {info_pattern}")
    print(f"🔎 baselined_pattern: {baselined_peak_pattern}")
    print(f"🔎 preproc_txt_pattern: {preproc_info_txt_pattern}")

    print_file_list("Peak amplitude files", amp_files)
    print_file_list("Peak AUC files", auc_files)
    print_file_list("Peak info files", info_files)
    print_file_list("Baselined peri-peak files", baselined_peak_files)
    print_file_list("Preprocessing info TXT files", preproc_info_txt_files)

    print("========================================================\n")


    if not amp_files or not auc_files or not info_files:
        raise FileNotFoundError("Missing required input files.")
    if not preproc_info_txt_files:
        raise FileNotFoundError("❌ Could not find preproc info txt file.")

    fs = extract_fs_from_txt(preproc_info_txt_files[0])
    print(f"✅ Extracted sampling rate: fs = {fs:.3f} Hz")

    # --- Identify rats ---
    rat_files_dict = {}
    rat_pattern = re.compile(r'_(ACW_.*?)_peak_amps')
    for f in amp_files:
        m = rat_pattern.search(os.path.basename(f))
        if m:
            rat_id = m.group(1)
            if rat_id not in rat_files_dict:
                rat_files_dict[rat_id] = {'partials': [], 'single': []}
            if "_partial" in f:
                rat_files_dict[rat_id]['partials'].append(f)
            else:
                rat_files_dict[rat_id]['single'].append(f)

    rats = sorted(rat_files_dict.keys())
    print(f"\n🐀 Found {len(rats)} unique rat(s): {rats}")

    per_rat_outputs = {}
    all_amps_list, all_aucs_list, all_baselined_list, all_stacked_list = [], [], [], []

    # --- Process each rat ---
    for rat in rats:
        files_info = rat_files_dict[rat]

        combined_amps_df = pd.DataFrame()
        combined_aucs_df = pd.DataFrame()
        combined_baselined_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        stacked_df = pd.DataFrame()
        all_peak_rows = []

        # NEW: accumulate duration + peaks properly
        total_duration = 0.0
        total_peaks = 0

        # --- Process partials first (or single) ---
        input_files = files_info['partials'] + files_info['single']
        print(f"\n🐀 Rat: {rat}")
        print_file_list("Amplitude files used", input_files)

        for af in input_files:

            # Load amplitude and AUC
            amp_df = pd.read_feather(af)["peak amplitudes"].to_frame(name="peak amplitudes")
            auc_file = af.replace("_peak_amps", "_peak_aucs")
            auc_df = pd.read_feather(auc_file)["peak AUCs"].to_frame(name="peak AUCs")

            # Load info
            info_file = af.replace("_peak_amps", "_peaks_info")
            if not os.path.exists(info_file):
                raise FileNotFoundError(f"Missing info file for {af}")

            info_df_file = pd.read_feather(info_file).set_index("peak measurements")
            # Later, use info_df_file for AUC window
            auc_win_val = float(info_df_file.loc["AUC window (sec)"].values[0])
            
            # --- Duration accumulation (FIXED) ---
            if "duration of time segment (sec)" not in info_df_file.index:
                raise ValueError(f"Duration missing in {info_file}")

            file_duration = float(
                pd.to_numeric(
                    info_df_file.loc["duration of time segment (sec)"].values[0],
                    errors="coerce"
                )
            )

            total_duration += file_duration

            # --- Peak accumulation (FIXED) ---
            n_peaks = amp_df.shape[0]
            total_peaks += n_peaks

            # --- Stacked dataframe rows ---
            peak_rows = pd.DataFrame({
                "rat": rat,
                "source_file": os.path.basename(af),
                "peak_idx": np.arange(n_peaks),
                "peak amplitudes": amp_df["peak amplitudes"].values,
                "peak AUCs": auc_df["peak AUCs"].values
            })
            all_peak_rows.append(peak_rows)

            # Combine for per-rat summaries
            combined_amps_df = pd.concat([combined_amps_df, amp_df], ignore_index=True)
            combined_aucs_df = pd.concat([combined_aucs_df, auc_df], ignore_index=True)

        # --- Build stacked DF per rat ---
        if all_peak_rows:
            stacked_df = pd.concat(all_peak_rows, ignore_index=True)
            all_stacked_list.append(stacked_df)


        # --- Baselined traces ---
        baselined_files = [bf for bf in baselined_peak_files if rat in bf]
        if baselined_files:
            baselined_dfs = [pd.read_feather(bf) for bf in baselined_files]
            combined_baselined_df = pd.concat(baselined_dfs, axis=1)
            combined_baselined_df.columns = [
                f"{rat}__{col}" for col in combined_baselined_df.columns
            ]
            # --- Optional re-baselining ---
            if rebaseline_window is not None:
                start_sec, end_sec = rebaseline_window
                n_samples = combined_baselined_df.shape[0]
                # Build time vector (currently your ts is -5:5)
                ts = np.linspace(-5, 5, n_samples)  
                baseline_idx = (ts >= start_sec) & (ts <= end_sec)
                if not baseline_idx.any():
                    print(f"⚠️ Warning: rebaseline window {rebaseline_window} outside trace range")
                else:
                    baseline_values = combined_baselined_df.iloc[baseline_idx, :].mean(axis=0)
                    combined_baselined_df = combined_baselined_df - baseline_values
                    
            all_baselined_list.append(combined_baselined_df)

        # --- Recompute amplitudes / AUCs from re-baselined traces ---
        '''
        if rebaseline_window is not None and not combined_baselined_df.empty:
            # Peak amplitude: subtract baseline
            combined_amps_df = pd.DataFrame({
                "peak amplitudes": combined_baselined_df.max(axis=0)
            })

            # Time vector in seconds
            ts = np.arange(combined_baselined_df.shape[0]) / fs

            # AUC window from original info file
            auc_win_val = float(info_df_file.loc["AUC window (sec)"].values[0])
            start_win = -auc_win_val / 2
            end_win   =  auc_win_val / 2

            # Boolean mask by position
            auc_mask = (ts >= start_win) & (ts <= end_win)

            # Peak AUCs computed on re-baselined traces
            combined_aucs_df = pd.DataFrame({
                "peak AUCs": [
                    np.trapz(combined_baselined_df[col].iloc[auc_mask], ts[auc_mask])
                    for col in combined_baselined_df.columns
                ]
            })

            # Ensure numeric dtype for sem computation
            combined_aucs_df["peak AUCs"] = pd.to_numeric(combined_aucs_df["peak AUCs"], errors="coerce")
            combined_amps_df["peak amplitudes"] = pd.to_numeric(combined_amps_df["peak amplitudes"], errors="coerce")
        '''
        # --- Optional re-baselining ---
        if rebaseline_window is not None and not combined_baselined_df.empty:
            start_sec, end_sec = rebaseline_window
            n_samples = combined_baselined_df.shape[0]

            # Build time vector matching trace length
            ts = np.linspace(-5, 5, n_samples)  # adjust -5,5 if needed for actual trace

            # --- Compute baseline across the selected window ---
            baseline_idx = (ts >= start_sec) & (ts <= end_sec)
            if not baseline_idx.any():
                print(f"⚠️ Warning: rebaseline window {rebaseline_window} outside trace range")
                baseline_values = 0
            else:
                # Important: use .iloc to index rows by boolean mask
                baseline_values = combined_baselined_df.iloc[baseline_idx, :].mean(axis=0)

            # Subtract baseline from all rows
            combined_baselined_df = combined_baselined_df - baseline_values

            # --- Recompute amplitudes respecting peak polarity ---
            # Determine original polarity BEFORE overwriting amplitudes
            original_mean_amp = combined_amps_df["peak amplitudes"].mean()

            if original_mean_amp < 0:
                # Negative peaks (dips)
                peak_vals = combined_baselined_df.min(axis=0)
            else:
                # Positive peaks
                peak_vals = combined_baselined_df.max(axis=0)

            combined_amps_df = pd.DataFrame({
                "peak amplitudes": peak_vals
            })

            # --- Recompute peak AUCs ---
            # AUC window comes from info file (original setting)
            half = int((auc_win_val / 2) * fs)
            center = combined_baselined_df.shape[0] // 2
            auc_start = center - half
            auc_end = center + half

            auc_time = np.linspace(
                -auc_win_val/2,
                auc_win_val/2,
                auc_end - auc_start
            )

            combined_aucs_df = pd.DataFrame({
                "peak AUCs": [
                    np.trapz(
                        combined_baselined_df[col].iloc[auc_start:auc_end],
                        auc_time
                    )
                    for col in combined_baselined_df.columns
                ]
            })

            # Ensure numeric dtype for downstream stats
            combined_amps_df["peak amplitudes"] = pd.to_numeric(
                combined_amps_df["peak amplitudes"], errors="coerce"
            )
            combined_aucs_df["peak AUCs"] = pd.to_numeric(
                combined_aucs_df["peak AUCs"], errors="coerce"
            )


        # --- Compute per-rat summary ---
         # --- Compute per-rat summary (CORRECTED) ---
        if total_duration <= 0:
            raise ValueError(f"Total duration invalid for rat {rat}")

        rat_frequency = total_peaks / total_duration

        summary_dict = {
            "duration of time segment (sec)": total_duration,
            "number of peaks detected": total_peaks,
            "peak frequency (Hz)": rat_frequency,
            "peak amplitude mean": combined_amps_df["peak amplitudes"].mean(),
            "peak amplitude SEM": stats.sem(
                combined_amps_df["peak amplitudes"], nan_policy='omit'
            ),
            "peak AUC mean": combined_aucs_df["peak AUCs"].mean(),
            "peak AUC SEM": stats.sem(
                combined_aucs_df["peak AUCs"], nan_policy='omit'
            )
        }

        summary_df = pd.DataFrame(
            list(summary_dict.items()),
            columns=["peak measurements", "value"]
        )
        
        # Store outputs
        per_rat_outputs[rat] = (
            summary_df,
            combined_amps_df,
            combined_aucs_df,
            combined_baselined_df,
            stacked_df
        )

        all_amps_list.append(combined_amps_df)
        all_aucs_list.append(combined_aucs_df)


    # --- Across-rat combination ---
    combined_amps_all = pd.concat(all_amps_list, ignore_index=True)
    combined_aucs_all = pd.concat(all_aucs_list, ignore_index=True)
    combined_baselined_all = pd.concat(all_baselined_list, axis=1) if all_baselined_list else pd.DataFrame()
    combined_stacked_all = pd.concat(all_stacked_list, ignore_index=True) if all_stacked_list else pd.DataFrame()

    # --- Across-rat summary (mean across animals, NOT pooled) ---

    rat_summaries = [rat_tuple[0] for rat_tuple in per_rat_outputs.values()]

    rat_durations = []
    rat_peak_counts = []
    rat_frequencies = []

    for summary_df in rat_summaries:
        if summary_df.empty:
            continue

        duration_val = float(
            summary_df.loc[
                summary_df["peak measurements"] == "duration of time segment (sec)",
                "value"
            ].values[0]
        )

        peak_val = float(
            summary_df.loc[
                summary_df["peak measurements"] == "number of peaks detected",
                "value"
            ].values[0]
        )

        freq_val = float(
            summary_df.loc[
                summary_df["peak measurements"] == "peak frequency (Hz)",
                "value"
            ].values[0]
        )

        rat_durations.append(duration_val)
        rat_peak_counts.append(peak_val)
        rat_frequencies.append(freq_val)

    if len(rat_frequencies) == 0:
        raise ValueError("No valid per-rat summaries found.")

    # Totals (still useful for reporting)
    total_duration_all = sum(rat_durations)
    total_peaks_all = sum(rat_peak_counts)

    # --- CORRECT across-rat statistics ---

    # Extract per-rat amplitude and AUC means
    rat_amp_means = []
    rat_auc_means = []

    for summary_df in rat_summaries:
        if summary_df.empty:
            continue

        amp_mean = float(
            summary_df.loc[
                summary_df["peak measurements"] == "peak amplitude mean",
                "value"
            ].values[0]
        )

        auc_mean = float(
            summary_df.loc[
                summary_df["peak measurements"] == "peak AUC mean",
                "value"
            ].values[0]
        )

        rat_amp_means.append(amp_mean)
        rat_auc_means.append(auc_mean)
        
    n_rats = len(rat_frequencies)

    mean_frequency = np.mean(rat_frequencies)

    if n_rats > 1:
        # --- Across-animal SEM ---
        sem_frequency = stats.sem(rat_frequencies, nan_policy="omit")

        mean_amp = np.mean(rat_amp_means)
        sem_amp = stats.sem(rat_amp_means, nan_policy="omit")

        mean_auc = np.mean(rat_auc_means)
        sem_auc = stats.sem(rat_auc_means, nan_policy="omit")

    else:
        # --- Within-animal SEM (single rat case) ---
        sem_frequency = 0  # frequency is a single value

        mean_amp = combined_amps_all["peak amplitudes"].mean()
        sem_amp = stats.sem(
            combined_amps_all["peak amplitudes"],
            nan_policy="omit"
        )

        mean_auc = combined_aucs_all["peak AUCs"].mean()
        sem_auc = stats.sem(
            combined_aucs_all["peak AUCs"],
            nan_policy="omit"
        )


    if n_rats > 1:
        freq_sem_label = "peak frequency SEM (across rats)"
        amp_sem_label = "peak amplitude SEM (across rats)"
        auc_sem_label = "peak AUC SEM (across rats)"
    else:
        freq_sem_label = "peak frequency SEM (within rat)"
        amp_sem_label = "peak amplitude SEM (within rat)"
        auc_sem_label = "peak AUC SEM (within rat)"


    summary_dict_all = {
        "number of rats": n_rats,
        "total duration across rats (sec)": total_duration_all,
        "total peaks across rats": int(total_peaks_all),
        "peak frequency (Hz) - mean across rats": mean_frequency,
        freq_sem_label: sem_frequency,
        "peak amplitude mean": mean_amp,
        amp_sem_label: sem_amp,
        "peak AUC mean": mean_auc,
        auc_sem_label: sem_auc,
    }


    summary_all = pd.DataFrame(
        list(summary_dict_all.items()),
        columns=["peak measurements", "value"]
    )


    # --- Time vector for plotting ---
    n_samples = combined_baselined_all.shape[0] if not combined_baselined_all.empty else 0
    ts = np.linspace(-5, 5, n_samples) if n_samples>0 else np.array([])

    # --- Plot baselined traces with AUC window ---
    fig_all = None
    if not combined_baselined_all.empty:
        mean_trace_all = combined_baselined_all.mean(axis=1)
        sem_trace_all = stats.sem(combined_baselined_all, axis=1, nan_policy='omit')

        fig_all, ax = plt.subplots(figsize=(8,6))
        ax.plot(ts, combined_baselined_all, color='gray', alpha=0.4, linewidth=0.5)
        ax.plot(ts, mean_trace_all, color='yellow', linewidth=3)

        if info_files:
            info0 = pd.read_feather(info_files[0]).set_index("peak measurements")
            if "AUC window (sec)" in info0.index:
                auc_win_val = float(info0.loc["AUC window (sec)", :].values[0])
                start_win = -auc_win_val / 2
                end_win = auc_win_val / 2
                mask = (ts >= start_win) & (ts <= end_win)
                ts_win = ts[mask]
                mean_win = mean_trace_all[mask]
                ax.fill_between(ts_win, 0, mean_win, color='magenta', alpha=0.5)

        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel("Time (s) relative to peak")
        ax.set_ylabel("Baseline-corrected Fluorescence")
        ax.set_title("Combined Baselined Peri-Peak Traces (All Rats)")
        plt.tight_layout()

    # --- Build per-rat summary DataFrame with rats as columns ---
    per_rat_summary_dict = {}
    for rat, (summary_df, *_ ) in per_rat_outputs.items():
        per_rat_summary_dict[rat] = summary_df.set_index("peak measurements")["value"]

    per_rat_summary_df = pd.DataFrame({
        "peak measurements": list(next(iter(per_rat_summary_dict.values())).index),
        **per_rat_summary_dict
    })


    # --- Save per-rat outputs (all in save_folder) ---
    for rat, (summary_df, combined_amps_df, combined_aucs_df, combined_baselined_df, stacked_df) in per_rat_outputs.items():
        safe_save_table(
            combined_amps_df,
            save_folder / f"{filename_prefix}{rat}_combined_peak_amp",
            f"{rat} amplitudes"
        )
        safe_save_table(
            combined_aucs_df,
            save_folder / f"{filename_prefix}{rat}_combined_peak_aucs",
            f"{rat} AUCs"
        )
        safe_save_table(
            summary_df,
            save_folder / f"{filename_prefix}{rat}_summary",
            f"{rat} summary"
        )
        safe_save_table(
            stacked_df,
            save_folder / f"{filename_prefix}{rat}_stacked_peaks",
            f"{rat} stacked peaks"
        )


    # --- Save across-rat outputs ---
    rats_in_filename = "-".join(rats)
    safe_save_table(
        summary_all,
        save_folder / f"{filename_prefix}{rats_in_filename}_peaks_combined_info",
        "combined summary"
    )
    txt_path = save_folder / f"{filename_prefix}{rats_in_filename}_peaks_combined_events_info.txt"
    with open(txt_path, "w") as ftxt:
        ftxt.write(summary_all.to_string(index=False))
    safe_save_table(
        combined_amps_all,
        save_folder / f"{filename_prefix}{rats_in_filename}_peaks_combined_amps",
        "combined amplitudes"
    )
    safe_save_table(
        combined_aucs_all,
        save_folder / f"{filename_prefix}{rats_in_filename}_peaks_combined_aucs",
        "combined AUCs"
    )
    safe_save_table(
        combined_stacked_all,
        save_folder / f"{filename_prefix}{rats_in_filename}_stacked_peaks",
        "stacked per-peak dataframe"
    )

    # --- Save per-rat summary DataFrame (rats as columns) ---
    safe_save_table(
        per_rat_summary_df,
        save_folder / f"{filename_prefix}per_rat_summary",
        desc="per-rat summary (each rat is a column)"
    )
    
    return per_rat_outputs, (summary_all, combined_amps_all, combined_aucs_all, combined_baselined_all, combined_stacked_all, fig_all)

        
        
# combined analysis of peaks from multiple time segments --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# updated 01.23.26 to save the per-rat organized dataframe of all peaks for later mixed model analysis
# updated 01.29.26 to include csv saving
        
def load_combine_and_save_peak_data_multi_021726(
    folder,
    amp_pattern,
    auc_pattern,
    info_pattern,
    filename_prefix,
    preproc_info_txt_pattern,
    baselined_peak_pattern,
    overwrite=True
):
    import pandas as pd
    import numpy as np
    import os
    import glob
    import re
    import matplotlib.pyplot as plt
    from scipy import stats

    # --- Helper: extract sampling rate ---
    def extract_fs_from_txt(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if line.lower().startswith("new fs after downsample"):
                    fs = float(line.strip().split(":")[1].strip())
                    return fs
        raise ValueError(f"Sampling rate not found in {txt_path}")

    # --- Safe save helper for Feather ---
    def safe_save(df, path, desc="DataFrame"):
        df = df.copy()
        df.columns = df.columns.astype(str)
        if df.index.name is not None or not df.index.is_integer():
            df = df.reset_index(drop=True)
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.to_feather(path)
        print(f"💾 Saved {desc} to:\n  {path}")

    # --- Safe save helper for both Feather + CSV ---
    def safe_save_table(df, base_path, desc="DataFrame"):
        """
        Save a pandas DataFrame as both Feather and CSV with overwrite protection.
        `base_path` should NOT include extension.
        """
        if df is None or df.empty:
            return False

        feather_path = base_path + ".feather"
        csv_path     = base_path + ".csv"

        # Feather
        safe_save(df, feather_path, desc)

        # CSV
        if os.path.exists(csv_path) and not overwrite:
            print(f"↪️  Reusing existing file: {os.path.basename(csv_path)}")
        else:
            df.to_csv(csv_path, index=False)
            print(f"💾 Saved {desc} CSV: {os.path.basename(csv_path)}")

        return True
    
    def print_file_list(title, files, indent=2):
        print(f"\n📂 {title} ({len(files)} file(s)):")
        if not files:
            print(" " * indent + "— none —")
        else:
            for f in sorted(files):
                print(" " * indent + os.path.basename(f))


    # --- Find files ---
    amp_files = glob.glob(os.path.join(folder, amp_pattern))
    auc_files = glob.glob(os.path.join(folder, auc_pattern))
    info_files = glob.glob(os.path.join(folder, info_pattern))
    baselined_peak_files = glob.glob(os.path.join(folder, baselined_peak_pattern))
    preproc_info_txt_files = glob.glob(os.path.join(folder, preproc_info_txt_pattern))

    print("\n================ FILES USED IN ANALYSIS ================")
    print(f"📁 Folder: {folder}")
    print(f"🔎 amp_pattern: {amp_pattern}")
    print(f"🔎 auc_pattern: {auc_pattern}")
    print(f"🔎 info_pattern: {info_pattern}")
    print(f"🔎 baselined_pattern: {baselined_peak_pattern}")
    print(f"🔎 preproc_txt_pattern: {preproc_info_txt_pattern}")

    print_file_list("Peak amplitude files", amp_files)
    print_file_list("Peak AUC files", auc_files)
    print_file_list("Peak info files", info_files)
    print_file_list("Baselined peri-peak files", baselined_peak_files)
    print_file_list("Preprocessing info TXT files", preproc_info_txt_files)

    print("========================================================\n")


    if not amp_files or not auc_files or not info_files:
        raise FileNotFoundError("Missing required input files.")
    if not preproc_info_txt_files:
        raise FileNotFoundError("❌ Could not find preproc info txt file.")

    fs = extract_fs_from_txt(preproc_info_txt_files[0])
    print(f"✅ Extracted sampling rate: fs = {fs:.3f} Hz")

    # --- Identify rats ---
    rat_files_dict = {}
    rat_pattern = re.compile(r'_(ACW_.*?)_peak_amps')
    for f in amp_files:
        m = rat_pattern.search(os.path.basename(f))
        if m:
            rat_id = m.group(1)
            if rat_id not in rat_files_dict:
                rat_files_dict[rat_id] = {'partials': [], 'single': []}
            if "_partial" in f:
                rat_files_dict[rat_id]['partials'].append(f)
            else:
                rat_files_dict[rat_id]['single'].append(f)

    rats = sorted(rat_files_dict.keys())
    print(f"\n🐀 Found {len(rats)} unique rat(s): {rats}")

    per_rat_outputs = {}
    all_amps_list, all_aucs_list, all_baselined_list, all_stacked_list = [], [], [], []

    # --- Process each rat ---
    for rat in rats:
        files_info = rat_files_dict[rat]
        combined_amps_df = pd.DataFrame()
        combined_aucs_df = pd.DataFrame()
        combined_baselined_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        stacked_df = pd.DataFrame()
        all_peak_rows = []

        # --- Process partials first (or single) ---
        input_files = files_info['partials'] + files_info['single']
        print(f"\n🐀 Rat: {rat}")
        print_file_list("Amplitude files used", input_files)

        for af in input_files:
            # Load amplitude and AUC
            amp_df = pd.read_feather(af)["peak amplitudes"].to_frame(name="peak amplitudes")
            auc_file = af.replace("_peak_amps", "_peak_aucs")
            auc_df = pd.read_feather(auc_file)["peak AUCs"].to_frame(name="peak AUCs")

            # Load info
            info_file = af.replace("_peak_amps", "_peaks_info")
            info_df = pd.read_feather(info_file).set_index("peak measurements") if os.path.exists(info_file) else pd.DataFrame()

            # --- Stacked dataframe rows ---
            n_peaks = amp_df.shape[0]
            peak_rows = pd.DataFrame({
                "rat": rat,
                "source_file": os.path.basename(af),
                "peak_idx": np.arange(n_peaks),
                "peak amplitudes": amp_df["peak amplitudes"].values,
                "peak AUCs": auc_df["peak AUCs"].values
            })
            all_peak_rows.append(peak_rows)

            # Combine for per-rat summaries
            combined_amps_df = pd.concat([combined_amps_df, amp_df], ignore_index=True)
            combined_aucs_df = pd.concat([combined_aucs_df, auc_df], ignore_index=True)

        # --- Build stacked DF per rat ---
        if all_peak_rows:
            stacked_df = pd.concat(all_peak_rows, ignore_index=True)
            all_stacked_list.append(stacked_df)

        # --- Compute per-rat summary ---
        if not info_df.empty and "duration of time segment (sec)" in info_df.index:
            duration_vals = pd.to_numeric(info_df.loc["duration of time segment (sec)"], errors='coerce')
            duration_sum = duration_vals.sum()
        else:
            duration_sum = combined_amps_df.shape[0]/fs

        peak_count = combined_amps_df.shape[0]

        summary_dict = {
            "duration of time segment (sec)": duration_sum,
            "number of peaks detected": peak_count,
            "peak frequency (Hz)": peak_count/duration_sum if duration_sum>0 else 0,
            "peak amplitude mean": combined_amps_df["peak amplitudes"].mean(),
            "peak amplitude SEM": stats.sem(combined_amps_df["peak amplitudes"], nan_policy='omit'),
            "peak AUC mean": combined_aucs_df["peak AUCs"].mean(),
            "peak AUC SEM": stats.sem(combined_aucs_df["peak AUCs"], nan_policy='omit')
        }
        summary_df = pd.DataFrame(list(summary_dict.items()), columns=["peak measurements", "value"])

        # --- Baselined traces ---
        baselined_files = [bf for bf in baselined_peak_files if rat in bf]
        if baselined_files:
            baselined_dfs = [pd.read_feather(bf) for bf in baselined_files]
            combined_baselined_df = pd.concat(baselined_dfs, axis=1)
            combined_baselined_df.columns = [f"{rat}__{col}" for col in combined_baselined_df.columns]
            all_baselined_list.append(combined_baselined_df)

        # --- Save per-rat outputs (Feather + CSV) ---
        
        safe_save_table(combined_amps_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_amp"), f"{rat} amplitudes")
        safe_save_table(combined_aucs_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_aucs"), f"{rat} AUCs")
        safe_save_table(summary_df, os.path.join(folder, f"{filename_prefix}{rat}_summary"), f"{rat} summary")
        safe_save_table(stacked_df, os.path.join(folder, f"{filename_prefix}{rat}_stacked_peaks"), f"{rat} stacked peaks")
        
        
        per_rat_outputs[rat] = (summary_df, combined_amps_df, combined_aucs_df, combined_baselined_df, stacked_df)

        all_amps_list.append(combined_amps_df)
        all_aucs_list.append(combined_aucs_df)

    # --- Across-rat combination ---
    combined_amps_all = pd.concat(all_amps_list, ignore_index=True)
    combined_aucs_all = pd.concat(all_aucs_list, ignore_index=True)
    combined_baselined_all = pd.concat(all_baselined_list, axis=1) if all_baselined_list else pd.DataFrame()
    combined_stacked_all = pd.concat(all_stacked_list, ignore_index=True) if all_stacked_list else pd.DataFrame()

    # --- Across-rat summary ---
    duration_vals = [
        pd.to_numeric(
            rat_tuple[0].loc[rat_tuple[0]["peak measurements"]=="duration of time segment (sec)", "value"].values[0],
            errors='coerce'
        )
        for rat_tuple in per_rat_outputs.values() if not rat_tuple[0].empty
    ]
    peak_vals = [
        pd.to_numeric(
            rat_tuple[0].loc[rat_tuple[0]["peak measurements"]=="number of peaks detected", "value"].values[0],
            errors='coerce'
        )
        for rat_tuple in per_rat_outputs.values() if not rat_tuple[0].empty
    ]

    duration_sum = sum(duration_vals)
    peak_count = sum(peak_vals)

    summary_dict_all = {
        "duration of time segment (sec)": duration_sum,
        "number of peaks detected": int(peak_count),
        "peak frequency (Hz)": peak_count/duration_sum if duration_sum>0 else 0,
        "peak amplitude mean": combined_amps_all["peak amplitudes"].mean(),
        "peak amplitude SEM": stats.sem(combined_amps_all["peak amplitudes"], nan_policy='omit'),
        "peak AUC mean": combined_aucs_all["peak AUCs"].mean(),
        "peak AUC SEM": stats.sem(combined_aucs_all["peak AUCs"], nan_policy='omit')
    }

    summary_all = pd.DataFrame(list(summary_dict_all.items()), columns=["peak measurements", "value"])

    # --- Time vector for plotting ---
    n_samples = combined_baselined_all.shape[0] if not combined_baselined_all.empty else 0
    ts = np.linspace(-5, 5, n_samples) if n_samples>0 else np.array([])

    # --- Plot baselined traces with AUC window ---
    fig_all = None
    if not combined_baselined_all.empty:
        mean_trace_all = combined_baselined_all.mean(axis=1)
        sem_trace_all = stats.sem(combined_baselined_all, axis=1, nan_policy='omit')

        fig_all, ax = plt.subplots(figsize=(8,6))
        ax.plot(ts, combined_baselined_all, color='gray', alpha=0.4, linewidth=0.5)
        ax.plot(ts, mean_trace_all, color='yellow', linewidth=3)

        if info_files:
            info0 = pd.read_feather(info_files[0]).set_index("peak measurements")
            if "AUC window (sec)" in info0.index:
                auc_win_val = float(info0.loc["AUC window (sec)", :].values[0])
                start_win = -auc_win_val / 2
                end_win = auc_win_val / 2
                mask = (ts >= start_win) & (ts <= end_win)
                ts_win = ts[mask]
                mean_win = mean_trace_all[mask]
                ax.fill_between(ts_win, 0, mean_win, color='magenta', alpha=0.5)

        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel("Time (s) relative to peak")
        ax.set_ylabel("Baseline-corrected Fluorescence")
        ax.set_title("Combined Baselined Peri-Peak Traces (All Rats)")
        plt.tight_layout()

    # --- Save across-rat outputs (Feather + CSV + txt) ---
    rats_in_filename = "-".join(rats)
    safe_save_table(summary_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_info"), "combined summary")
    txt_path = os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_events_info.txt")
    with open(txt_path, "w") as ftxt:
        ftxt.write(summary_all.to_string(index=False))
    safe_save_table(combined_amps_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_amps"), "combined amplitudes")
    safe_save_table(combined_aucs_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_aucs"), "combined AUCs")
    safe_save_table(combined_stacked_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_stacked_peaks"), "stacked per-peak dataframe")

    return per_rat_outputs, (summary_all, combined_amps_all, combined_aucs_all, combined_baselined_all, combined_stacked_all, fig_all)

        
        
        
# combined analysis of peaks from multiple time segments --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# updated 01.23.26 to save the per-rat organized dataframe of all peaks for later mixed model analysis

def load_combine_and_save_peak_data_multi_012826(
    folder,
    amp_pattern,
    auc_pattern,
    info_pattern,
    filename_prefix,
    preproc_info_txt_pattern,
    baselined_peak_pattern
):
    import pandas as pd
    import numpy as np
    import os
    import glob
    import re
    import matplotlib.pyplot as plt
    from scipy import stats

    # --- Helper: extract sampling rate ---
    def extract_fs_from_txt(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if line.lower().startswith("new fs after downsample"):
                    fs = float(line.strip().split(":")[1].strip())
                    return fs
        raise ValueError(f"Sampling rate not found in {txt_path}")

    # --- Safe save helper ---
    def safe_save(df, path, desc):
        df = df.copy()
        df.columns = df.columns.astype(str)
        if df.index.name is not None or not df.index.is_integer():
            df = df.reset_index(drop=True)
        # Only convert numeric columns
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.to_feather(path)
        print(f"💾 Saved {desc} to:\n  {path}")
        
    
    def safe_save_table(df, base_path):
        """
        Save a pandas DataFrame as both Feather and CSV with overwrite protection.
        `base_path` should NOT include extension.
        """
        if df is None:
            return False

        feather_path = base_path + ".feather"
        csv_path     = base_path + ".csv"

        # Feather
        safe_save(df, feather_path)

        # CSV
        if os.path.exists(csv_path) and not overwrite:
            print(f"↪️  Reusing existing file: {os.path.basename(csv_path)}")
        else:
            df.to_csv(csv_path, index=False)
            print(f"💾 Saved: {os.path.basename(csv_path)}")

        return True



    # --- Find files ---
    amp_files = glob.glob(os.path.join(folder, amp_pattern))
    auc_files = glob.glob(os.path.join(folder, auc_pattern))
    info_files = glob.glob(os.path.join(folder, info_pattern))
    baselined_peak_files = glob.glob(os.path.join(folder, baselined_peak_pattern))
    preproc_info_txt_files = glob.glob(os.path.join(folder, preproc_info_txt_pattern))

    if not amp_files or not auc_files or not info_files:
        raise FileNotFoundError("Missing required input files.")
    if not preproc_info_txt_files:
        raise FileNotFoundError("❌ Could not find preproc info txt file.")

    fs = extract_fs_from_txt(preproc_info_txt_files[0])
    print(f"✅ Extracted sampling rate: fs = {fs:.3f} Hz")

    # --- Identify rats ---
    rat_files_dict = {}
    rat_pattern = re.compile(r'_(ACW_.*?)_peak_amps')
    for f in amp_files:
        m = rat_pattern.search(os.path.basename(f))
        if m:
            rat_id = m.group(1)
            if rat_id not in rat_files_dict:
                rat_files_dict[rat_id] = {'partials': [], 'single': []}
            if "_partial" in f:
                rat_files_dict[rat_id]['partials'].append(f)
            else:
                rat_files_dict[rat_id]['single'].append(f)

    rats = sorted(rat_files_dict.keys())
    print(f"\n🐀 Found {len(rats)} unique rat(s): {rats}")

    per_rat_outputs = {}
    all_amps_list, all_aucs_list, all_baselined_list, all_stacked_list = [], [], [], []

    # --- Process each rat ---
    for rat in rats:
        files_info = rat_files_dict[rat]
        combined_amps_df = pd.DataFrame()
        combined_aucs_df = pd.DataFrame()
        combined_baselined_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        stacked_df = pd.DataFrame()

        all_peak_rows = []

        # --- Process partials first (or single) ---
        input_files = files_info['partials'] + files_info['single']
        for af in input_files:
            # Load amplitude and AUC
            amp_df = pd.read_feather(af)["peak amplitudes"].to_frame(name="peak amplitudes")
            auc_file = af.replace("_peak_amps", "_peak_aucs")
            auc_df = pd.read_feather(auc_file)["peak AUCs"].to_frame(name="peak AUCs")

            # Load info
            info_file = af.replace("_peak_amps", "_peaks_info")
            info_df = pd.read_feather(info_file).set_index("peak measurements") if os.path.exists(info_file) else pd.DataFrame()

            # --- Stacked dataframe rows ---
            n_peaks = amp_df.shape[0]
            peak_rows = pd.DataFrame({
                "rat": rat,
                "source_file": os.path.basename(af),
                "peak_idx": np.arange(n_peaks),
                "peak amplitudes": amp_df["peak amplitudes"].values,
                "peak AUCs": auc_df["peak AUCs"].values
            })
            all_peak_rows.append(peak_rows)

            # Combine for per-rat summaries
            combined_amps_df = pd.concat([combined_amps_df, amp_df], ignore_index=True)
            combined_aucs_df = pd.concat([combined_aucs_df, auc_df], ignore_index=True)

        # --- Build stacked DF per rat ---
        if all_peak_rows:
            stacked_df = pd.concat(all_peak_rows, ignore_index=True)
            all_stacked_list.append(stacked_df)

        # --- Compute per-rat summary ---
        if not info_df.empty and "duration of time segment (sec)" in info_df.index:
            duration_vals = pd.to_numeric(info_df.loc["duration of time segment (sec)"], errors='coerce')
            duration_sum = duration_vals.sum()
        else:
            duration_sum = combined_amps_df.shape[0]/fs

        peak_count = combined_amps_df.shape[0]

        summary_dict = {
            "duration of time segment (sec)": duration_sum,
            "number of peaks detected": peak_count,
            "peak frequency (Hz)": peak_count/duration_sum if duration_sum>0 else 0,
            "peak amplitude mean": combined_amps_df["peak amplitudes"].mean(),
            "peak amplitude SEM": stats.sem(combined_amps_df["peak amplitudes"], nan_policy='omit'),
            "peak AUC mean": combined_aucs_df["peak AUCs"].mean(),
            "peak AUC SEM": stats.sem(combined_aucs_df["peak AUCs"], nan_policy='omit')
        }
        summary_df = pd.DataFrame(list(summary_dict.items()), columns=["peak measurements", "value"])


        # --- Baselined traces ---
        baselined_files = [bf for bf in baselined_peak_files if rat in bf]
        if baselined_files:
            baselined_dfs = [pd.read_feather(bf) for bf in baselined_files]
            combined_baselined_df = pd.concat(baselined_dfs, axis=1)
            combined_baselined_df.columns = [f"{rat}__{col}" for col in combined_baselined_df.columns]
            all_baselined_list.append(combined_baselined_df)

        # --- Save per-rat outputs ---
        '''
        safe_save(combined_amps_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_amp.feather"), "combined amplitudes")
        safe_save(combined_aucs_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_aucs.feather"), "combined AUCs")
        safe_save(summary_df, os.path.join(folder, f"{filename_prefix}{rat}_summary.feather"), "summary")
        safe_save(stacked_df, os.path.join(folder, f"{filename_prefix}{rat}_stacked_peaks.feather"), "stacked per-peak dataframe")
        '''
        per_rat_outputs[rat] = (summary_df, combined_amps_df, combined_aucs_df, combined_baselined_df, stacked_df)

        all_amps_list.append(combined_amps_df)
        all_aucs_list.append(combined_aucs_df)

    # --- Across-rat combination ---
    combined_amps_all = pd.concat(all_amps_list, ignore_index=True)
    combined_aucs_all = pd.concat(all_aucs_list, ignore_index=True)
    combined_baselined_all = pd.concat(all_baselined_list, axis=1) if all_baselined_list else pd.DataFrame()
    combined_stacked_all = pd.concat(all_stacked_list, ignore_index=True) if all_stacked_list else pd.DataFrame()

    # --- Across-rat summary ---
    duration_vals = [
        pd.to_numeric(
            rat_tuple[0].loc[rat_tuple[0]["peak measurements"]=="duration of time segment (sec)", "value"].values[0],
            errors='coerce'
        )
        for rat_tuple in per_rat_outputs.values() if not rat_tuple[0].empty
    ]

    peak_vals = [
        pd.to_numeric(
            rat_tuple[0].loc[rat_tuple[0]["peak measurements"]=="number of peaks detected", "value"].values[0],
            errors='coerce'
        )
        for rat_tuple in per_rat_outputs.values() if not rat_tuple[0].empty
    ]


    duration_sum = sum(duration_vals)
    peak_count = sum(peak_vals)

    summary_dict_all = {
        "duration of time segment (sec)": duration_sum,
        "number of peaks detected": int(peak_count),
        "peak frequency (Hz)": peak_count/duration_sum if duration_sum>0 else 0,
        "peak amplitude mean": combined_amps_all["peak amplitudes"].mean(),
        "peak amplitude SEM": stats.sem(combined_amps_all["peak amplitudes"], nan_policy='omit'),
        "peak AUC mean": combined_aucs_all["peak AUCs"].mean(),
        "peak AUC SEM": stats.sem(combined_aucs_all["peak AUCs"], nan_policy='omit')
    }

    summary_all = pd.DataFrame(list(summary_dict_all.items()), columns=["peak measurements", "value"])

    # --- Time vector for plotting ---
    n_samples = combined_baselined_all.shape[0] if not combined_baselined_all.empty else 0
    ts = np.linspace(-5, 5, n_samples) if n_samples>0 else np.array([])

    # --- Plot baselined traces with AUC window ---
    fig_all = None
    if not combined_baselined_all.empty:
        mean_trace_all = combined_baselined_all.mean(axis=1)
        sem_trace_all = stats.sem(combined_baselined_all, axis=1, nan_policy='omit')

        fig_all, ax = plt.subplots(figsize=(8,6))

        # Plot all individual traces
        ax.plot(ts, combined_baselined_all, color='gray', alpha=0.4, linewidth=0.5)

        # Plot mean trace
        ax.plot(ts, mean_trace_all, color='yellow', linewidth=3)

        # --- Fill area under the mean trace within AUC window ---
        if info_files:
            info0 = pd.read_feather(info_files[0]).set_index("peak measurements")
            if "AUC window (sec)" in info0.index:
                # Convert string to float
                auc_win_val = float(info0.loc["AUC window (sec)", :].values[0])
                start_win = -auc_win_val / 2
                end_win = auc_win_val / 2

                # Restrict ts and mean_trace_all to the AUC window
                mask = (ts >= start_win) & (ts <= end_win)
                ts_win = ts[mask]
                mean_win = mean_trace_all[mask]

                # Fill under the mean trace in that window
                ax.fill_between(ts_win, 0, mean_win, color='magenta', alpha=0.5)


        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel("Time (s) relative to peak")
        ax.set_ylabel("Baseline-corrected Fluorescence")
        ax.set_title("Combined Baselined Peri-Peak Traces (All Rats)")
        plt.tight_layout()


    # --- Save across-rat outputs ---
    rats_in_filename = "-".join(rats)
    safe_save(summary_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_info.feather"), "combined summary")
    txt_path = os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_events_info.txt")
    with open(txt_path, "w") as ftxt:
        ftxt.write(summary_all.to_string(index=False))
    safe_save(combined_amps_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_amps.feather"), "combined amplitudes")
    safe_save(combined_aucs_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_peaks_combined_aucs.feather"), "combined AUCs")
    safe_save(combined_stacked_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_stacked_peaks.feather"), "stacked per-peak dataframe")

    return per_rat_outputs, (summary_all, combined_amps_all, combined_aucs_all, combined_baselined_all, combined_stacked_all, fig_all)


         

# combined analysis of peaks from multiple time segments --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# updated 01.22.26 to handle multiple rats

def load_combine_and_save_peak_data_multi_012226(
    folder,
    amp_pattern,
    auc_pattern,
    info_pattern,
    filename_prefix,
    preproc_info_txt_pattern,
    baselined_peak_pattern
):
    import pandas as pd
    import numpy as np
    import os
    import glob
    import re
    import matplotlib.pyplot as plt
    from scipy import stats

    # --- Helper: extract sampling rate ---
    def extract_fs_from_txt(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if line.lower().startswith("new fs after downsample"):
                    # Extract the number after the colon
                    fs = float(line.strip().split(":")[1].strip())
                    return fs
        raise ValueError(f"Sampling rate not found in {txt_path}")


    # --- Robust safe save function ---
    def safe_save(df, path, desc):
        # Make a copy
        df = df.copy()

        # Ensure all column names are strings
        df.columns = df.columns.astype(str)

        # Reset index if needed (Feather prefers integer index)
        if df.index.name is not None or not df.index.is_integer():
            df = df.reset_index(drop=True)

        # Convert object columns to numeric if possible, otherwise to string
        for col in df.columns:
            if df[col].dtype == "object":
                # First try numeric conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # If any NaNs remain where original had strings, convert those to string
                mask = df[col].isna() & df[col].notnull()
                if mask.any():
                    # Convert whole column to string to preserve mixed content
                    df[col] = df[col].astype(str)

        # Feather requires all column types to be compatible
        # Save to feather
        df.to_feather(path)
        print(f"💾 Saved {desc} to:\n  {path}")



    # --- Find all files ---
    amp_files = glob.glob(os.path.join(folder, amp_pattern))
    auc_files = glob.glob(os.path.join(folder, auc_pattern))
    info_files = glob.glob(os.path.join(folder, info_pattern))
    baselined_peak_files = glob.glob(os.path.join(folder, baselined_peak_pattern))
    preproc_info_txt_files = glob.glob(os.path.join(folder, preproc_info_txt_pattern))

    if not amp_files or not auc_files or not info_files:
        raise FileNotFoundError("Missing required input files.")
    if not preproc_info_txt_files:
        raise FileNotFoundError("❌ Could not find preproc info txt file.")

    fs = extract_fs_from_txt(preproc_info_txt_files[0])
    print(f"✅ Extracted sampling rate: fs = {fs:.3f} Hz")

    # --- Identify rats and partials ---
    rat_files_dict = {}
    rat_pattern = re.compile(r'_ACW_(.*?)_peak_amps')
    for f in amp_files:
        m = rat_pattern.search(os.path.basename(f))
        if m:
            rat_id = m.group(1)
            if rat_id not in rat_files_dict:
                rat_files_dict[rat_id] = {'partials': [], 'single': [], 'combined': []}
            if "_partial" in f:
                rat_files_dict[rat_id]['partials'].append(f)
            elif "_combined" in f:
                rat_files_dict[rat_id]['combined'].append(f)
            else:
                rat_files_dict[rat_id]['single'].append(f)

    rats = sorted(rat_files_dict.keys())
    print(f"\n🐀 Found {len(rats)} unique rat(s): {rats}")

    # --- Prepare lists for across-animal combination ---
    all_summary_list, all_amps_list, all_aucs_list, all_baselined_list = [], [], [], []
    per_rat_outputs = {}

    # --- Process each rat ---
    # --- inside load_combine_and_save_peak_data_multi ---

    # --- Process each rat ---
    # --- Process each rat ---
    for rat in rats:
        files_info = rat_files_dict[rat]
        has_partials = len(files_info['partials']) > 0
        has_single = len(files_info['single']) > 0

        combined_amps_df = pd.DataFrame()
        combined_aucs_df = pd.DataFrame()
        combined_baselined_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        merged_info = pd.DataFrame()

        # --- Rats with partials (merge and save per-rat) ---
        if has_partials:
            print(f"\n🐀 Processing rat {rat} (partials found, combining within-rat)")

            # Load all partial files
            amp_dfs = [pd.read_feather(f)["peak amplitudes"].to_frame(name="peak amplitudes") for f in files_info['partials']]
            auc_dfs = [pd.read_feather(f.replace("_peak_amps", "_peak_aucs"))["peak AUCs"].to_frame(name="peak AUCs") for f in files_info['partials']]
            info_dfs = [pd.read_feather(f.replace("_peak_amps", "_peaks_info")).set_index("peak measurements")
                        for f in files_info['partials'] if os.path.exists(f.replace("_peak_amps", "_peaks_info"))]

            # Combine partials
            combined_amps_df = pd.concat(amp_dfs, ignore_index=True)
            combined_aucs_df = pd.concat(auc_dfs, ignore_index=True)
            if info_dfs:
                merged_info = pd.concat(info_dfs, axis=1)
                summary_series = merged_info.bfill(axis=1).iloc[:, 0]
                duration_sum = merged_info.loc["duration of time segment (sec)"].astype(float).sum()
                peak_count = merged_info.loc["number of peaks detected"].astype(float).sum()
            else:
                summary_series = pd.Series(dtype=float)
                duration_sum = combined_amps_df.shape[0] / fs
                peak_count = combined_amps_df.shape[0]

            # Recalculate summary stats
            summary_series["duration of time segment (sec)"] = duration_sum
            summary_series["number of peaks detected"] = int(peak_count)
            summary_series["peak frequency (Hz)"] = peak_count / duration_sum if duration_sum > 0 else 0
            summary_series["peak amplitude mean"] = combined_amps_df["peak amplitudes"].mean()
            summary_series["peak amplitude SEM"] = stats.sem(combined_amps_df["peak amplitudes"], nan_policy='omit')
            summary_series["peak AUC mean"] = combined_aucs_df["peak AUCs"].mean()
            summary_series["peak AUC SEM"] = stats.sem(combined_aucs_df["peak AUCs"], nan_policy='omit')

            summary_df = summary_series.reset_index()
            summary_df.columns = ["peak measurements", "value"]

            # Combine baselined traces
            baselined_files = [f for f in baselined_peak_files if rat in f]
            if baselined_files:
                baselined_dfs = [pd.read_feather(f) for f in baselined_files]
                combined_baselined_df = pd.concat(baselined_dfs, axis=1)
                combined_baselined_df.columns = [f"{rat}__{col}" for col in combined_baselined_df.columns]

            # Save per-rat combined files (partials only)
            safe_save(combined_amps_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_amp.feather"), "combined amplitudes")
            safe_save(combined_aucs_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_aucs.feather"), "combined AUCs")
            safe_save(summary_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peaks_summary.feather"), "summary (feather)")
            txt_path = os.path.join(folder, f"{filename_prefix}{rat}_combined_peaks_summary.txt")
            with open(txt_path, "w") as ftxt:
                ftxt.write(summary_df.to_string(index=False))
            print(f"💾 Saved summary (text) to:\n  {txt_path}")

        # --- Rats with only single files ---
        elif has_single:
            f = files_info['single'][0]
            print(f"\n🐀 Using single file for rat {rat}: {os.path.basename(f)}")

            combined_amps_df = pd.read_feather(f)["peak amplitudes"].to_frame(name="peak amplitudes")
            combined_aucs_df = pd.read_feather(f.replace("_peak_amps", "_peak_aucs"))["peak AUCs"].to_frame(name="peak AUCs")

            info_file = f.replace("_peak_amps", "_peaks_info")
            if os.path.exists(info_file):
                merged_info = pd.read_feather(info_file).set_index("peak measurements")
                summary_series = merged_info.bfill(axis=1).iloc[:, 0]
                duration_sum = merged_info.loc["duration of time segment (sec)"].astype(float).sum()
                peak_count = merged_info.loc["number of peaks detected"].astype(float).sum()
            else:
                merged_info = pd.DataFrame()
                summary_series = pd.Series(dtype=float)
                duration_sum = combined_amps_df.shape[0] / fs
                peak_count = combined_amps_df.shape[0]

            # Recalculate summary stats
            summary_series["duration of time segment (sec)"] = duration_sum
            summary_series["number of peaks detected"] = int(peak_count)
            summary_series["peak frequency (Hz)"] = peak_count / duration_sum if duration_sum > 0 else 0
            summary_series["peak amplitude mean"] = combined_amps_df["peak amplitudes"].mean()
            summary_series["peak amplitude SEM"] = stats.sem(combined_amps_df["peak amplitudes"], nan_policy='omit')
            summary_series["peak AUC mean"] = combined_aucs_df["peak AUCs"].mean()
            summary_series["peak AUC SEM"] = stats.sem(combined_aucs_df["peak AUCs"], nan_policy='omit')

            summary_df = summary_series.reset_index()
            summary_df.columns = ["peak measurements", "value"]

            # Combine baselined traces if present
            baselined_files = [bf for bf in baselined_peak_files if rat in bf]
            if baselined_files:
                baselined_dfs = [pd.read_feather(bf) for bf in baselined_files]
                combined_baselined_df = pd.concat(baselined_dfs, axis=1)
                combined_baselined_df.columns = [f"{rat}__{col}" for col in combined_baselined_df.columns]

        else:
            raise FileNotFoundError(f"❌ No files found for rat {rat}.")

        # --- Track outputs for across-animal combination ---
        per_rat_outputs[rat] = (summary_df, combined_amps_df, combined_aucs_df, combined_baselined_df)
        all_amps_list.append(combined_amps_df)
        all_aucs_list.append(combined_aucs_df)
        if not combined_baselined_df.empty:
            all_baselined_list.append(combined_baselined_df)


    # --- Combine across all rats ---
    combined_amps_all = pd.concat(all_amps_list, ignore_index=True)
    combined_aucs_all = pd.concat(all_aucs_list, ignore_index=True)
    combined_baselined_all = pd.concat(all_baselined_list, axis=1) if all_baselined_list else pd.DataFrame()

    # --- Recompute summary statistics across all rats ---
    summary_dict_all = {}

    duration_sum = 0
    peak_count = 0

    for rat in rats:
        summary_df, amps_df, aucs_df, baselined_df = per_rat_outputs[rat]
        if not summary_df.empty:
            # Extract duration and peak count if present
            duration_val = summary_df.loc[summary_df["peak measurements"]=="duration of time segment (sec)", "value"]
            peak_val = summary_df.loc[summary_df["peak measurements"]=="number of peaks detected", "value"]
            if not duration_val.empty:
                duration_sum += float(duration_val.values[0])
            if not peak_val.empty:
                peak_count += float(peak_val.values[0])

    summary_dict_all["duration of time segment (sec)"] = duration_sum
    summary_dict_all["number of peaks detected"] = int(peak_count)
    summary_dict_all["peak frequency (Hz)"] = peak_count / duration_sum if duration_sum > 0 else 0
    summary_dict_all["peak amplitude mean"] = combined_amps_all["peak amplitudes"].mean()
    summary_dict_all["peak amplitude SEM"] = stats.sem(combined_amps_all["peak amplitudes"], nan_policy='omit')
    summary_dict_all["peak AUC mean"] = combined_aucs_all["peak AUCs"].mean()
    summary_dict_all["peak AUC SEM"] = stats.sem(combined_aucs_all["peak AUCs"], nan_policy='omit')

    summary_all = pd.DataFrame(list(summary_dict_all.items()), columns=["peak measurements", "value"])


    # --- Generate ts vector aligned at 0 ---
    n_samples = combined_baselined_all.shape[0] if not combined_baselined_all.empty else 0
    ts = np.linspace(-5, 5, n_samples) if n_samples > 0 else np.array([])

    # --- Plot across-animal baselined traces ---
    if not combined_baselined_all.empty:
        mean_trace_all = combined_baselined_all.mean(axis=1)
        sem_trace_all = stats.sem(combined_baselined_all, axis=1, nan_policy="omit")

        fig_all, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ts, combined_baselined_all, color='gray', alpha=0.4, linewidth=0.5)
        ax.plot(ts, mean_trace_all, color='yellow', linewidth=3, label='Mean Response')
        ax.fill_between(ts, mean_trace_all + sem_trace_all, mean_trace_all - sem_trace_all,
                        color='yellow', alpha=0.4, label='SEM')
        ax.axvline(0, color='red', linestyle='--', label='Peak')
        ax.set_xlabel("Time (s) relative to peak")
        ax.set_ylabel("Baseline-corrected Fluorescence")
        ax.set_title("Combined Baselined Peri-Peak Traces (All Rats)")
        ax.legend()
        plt.tight_layout()
    else:
        fig_all = None

    # --- Save across-animal outputs ---
    rats_in_filename = "-".join(rats)
    safe_save(summary_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_combined_events_info.feather"), "combined summary (all rats)")
    txt_path = os.path.join(folder, f"{filename_prefix}{rats_in_filename}_combined_events_info.txt")
    with open(txt_path, "w") as ftxt:
        ftxt.write(summary_all.to_string(index=False))
    safe_save(combined_amps_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_combined_events_info_amps.feather"), "combined amplitudes")
    safe_save(combined_aucs_all, os.path.join(folder, f"{filename_prefix}{rats_in_filename}_combined_events_info_aucs.feather"), "combined AUCs")


    return per_rat_outputs, (summary_all, combined_amps_all, combined_aucs_all, combined_baselined_all, fig_all)

        
# combined analysis of peaks from multiple time segments --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# updated 08.08.25 to include AUC information
# updated 08.14.25 to handle peak segment dataframes


def extract_fs_from_txt(txt_path):
    import re
    with open(txt_path, "r") as f:
        for line in f:
            if "new fs after downsample" in line.lower():
                try:
                    fs = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                    return fs
                except Exception as e:
                    print(f"Failed to parse fs from line: {line.strip()}")
    raise ValueError("Could not find sampling rate (fs) in txt file.")
    
    

def load_combine_and_save_peak_data(
    folder,
    amp_pattern,
    auc_pattern,
    info_pattern,
    filename_prefix,
    preproc_info_txt_pattern,
    baselined_peak_pattern
):
    import pandas as pd
    import numpy as np
    import os
    import glob
    import re
    import matplotlib.pyplot as plt
    from scipy import stats

    amp_files = glob.glob(os.path.join(folder, amp_pattern))
    auc_files = glob.glob(os.path.join(folder, auc_pattern))
    info_files = glob.glob(os.path.join(folder, info_pattern))
    baselined_peak_files = glob.glob(os.path.join(folder, baselined_peak_pattern))
    preproc_info_txt_files = glob.glob(os.path.join(folder, preproc_info_txt_pattern))

    if not amp_files or not auc_files or not info_files:
        raise FileNotFoundError("Missing required input files.")
    if not preproc_info_txt_files:
        raise FileNotFoundError("❌ Could not find preproc info txt file.")

    # === Logging ===
    print(f"\n📁 Found {len(amp_files)} amplitude files.")
    for i, file in enumerate(amp_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    print(f"\n📁 Found {len(auc_files)} AUC files.")
    for i, file in enumerate(auc_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    print(f"\n📁 Found {len(info_files)} info files.")
    for i, file in enumerate(info_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    print(f"\n📁 Found {len(baselined_peak_files)} baselined peak files.")
    print(f"📁 Found {len(preproc_info_txt_files)} preproc info txt files.\n")

    # === Rat name ===
    rat_match = re.search(r'_ACW_(.*?)_peak_amps', os.path.basename(amp_files[0]))
    rat = f"ACW_{rat_match.group(1)}" if rat_match else "rat"
    print(f"\n🐀 Detected rat name: {rat}\n")

    # === Combine amp & AUC ===
    combined_amps_df = pd.concat([pd.read_feather(f)["peak amplitudes"] for f in amp_files], ignore_index=True).to_frame()
    combined_aucs_df = pd.concat([pd.read_feather(f)["peak AUCs"] for f in auc_files], ignore_index=True).to_frame()
    combined_amps_df.columns = ["peak amplitudes"]
    combined_aucs_df.columns = ["peak AUCs"]

    # === Combine Info ===
    info_dfs = []
    for f in info_files:
        df = pd.read_feather(f)
        if 'peak measurements' in df.columns:
            df = df.set_index('peak measurements')
            info_dfs.append(df)

    merged_info = pd.concat(info_dfs, axis=1)
    summary_series = merged_info.bfill(axis=1).iloc[:, 0]

    duration_sum = merged_info.loc["duration of time segment (sec)"].astype(float).sum()
    peak_count = merged_info.loc["number of peaks detected"].astype(float).sum()
    summary_series["duration of time segment (sec)"] = duration_sum
    summary_series["number of peaks detected"] = int(peak_count)
    summary_series["peak frequency (Hz)"] = peak_count / duration_sum if duration_sum > 0 else 0

    summary_series["peak amplitude mean"] = combined_amps_df.mean().iloc[0]
    summary_series["peak amplitude SEM"] = stats.sem(combined_amps_df["peak amplitudes"])
    summary_series["peak AUC mean"] = combined_aucs_df.mean().iloc[0]
    summary_series["peak AUC SEM"] = stats.sem(combined_aucs_df["peak AUCs"], nan_policy = 'omit')

    summary_df = summary_series.reset_index()
    summary_df.columns = ["peak measurements", "value"]
    summary_df["value"] = summary_df["value"].astype(str)

    # === Extract fs ===
    fs = extract_fs_from_txt(preproc_info_txt_files[0])
    print(f"✅ Extracted sampling rate: fs = {fs:.3f} Hz")

    # === Extract peri-peak window ===
    trange = None
    for f in info_files:
        df = pd.read_feather(f)
        for col in df.columns:
            match_rows = df[df[col].astype(str).str.contains("peri-peak window", case=False, na=False)]
            if not match_rows.empty:
                for val in match_rows.iloc[0]:
                    if isinstance(val, str) and "[" in val and "]" in val:
                        try:
                            trange = eval(val.strip())
                            break
                        except Exception:
                            continue
        if trange is not None:
            break
    if trange is None:
        raise ValueError("❌ Could not extract 'peri-peak window' from info files.")
    print(f"⏱ Extracted peri-peak window: {trange}")

    # === Extract AUC window ===
    auc_peak_window = None
    for f in info_files:
        df = pd.read_feather(f)
        match = df[df["peak measurements"] == "AUC window (sec)"]
        if not match.empty:
            try:
                auc_peak_window = float(match.iloc[0, 1])
                break
            except Exception as e:
                print(f"⚠️ Could not parse AUC window value: {match.iloc[0, 1]} ({e})")
    if auc_peak_window is None:
        auc_peak_window = 2.0
    print(f"📐 Extracted AUC window: {auc_peak_window:.2f} sec")

    # === Combine and plot traces ===
    if baselined_peak_files:
        dfs = [pd.read_feather(f) for f in baselined_peak_files]
        combined_baselined_df = pd.concat(dfs, axis=1)
        
          # ---- Fix duplicate column names here ----
        def dedupe_columns(cols):
            seen = {}
            new_cols = []
            for c in cols:
                if c not in seen:
                    seen[c] = 0
                    new_cols.append(c)
                else:
                    seen[c] += 1
                    new_cols.append(f"{c}__{seen[c]}")
            return new_cols

        combined_baselined_df.columns = dedupe_columns(combined_baselined_df.columns)
        
        mean_trace = combined_baselined_df.mean(axis=1)
        sem_trace = stats.sem(combined_baselined_df, axis=1, nan_policy="omit")
        time = trange[0] + np.arange(combined_baselined_df.shape[0]) / fs

        half_window = auc_peak_window / 2
        auc_mask = (time >= -half_window) & (time <= half_window)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(time, combined_baselined_df, color='gray', alpha=0.4, linewidth=0.5)
        ax.plot(time, mean_trace, color='yellow', linewidth=3, label='Mean Response')
        ax.fill_between(time, mean_trace + sem_trace, mean_trace - sem_trace,
                        color='yellow', alpha=0.4, label='SEM')

        if auc_mask.any():
            ax.fill_between(time[auc_mask], mean_trace[auc_mask], 0,
                            color='magenta', alpha=0.6, label='AUC (Mean Trace)', zorder=5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Baseline-corrected Fluorescence")
        ax.set_title(f"Combined Baselined Peri-Peak Traces ({rat})")
        ax.legend()
        plt.tight_layout()

        base_path = os.path.join(folder, f"{filename_prefix}{rat}_combined_baselined_peak_df.feather")
        if os.path.exists(base_path):
            print(f"⚠️ Baselined trace file exists, skipping save: {base_path}")
        else:
            combined_baselined_df.to_feather(base_path)
            print(f"💾 Saved combined baseline-corrected traces to:\n  {base_path}")
    else:
        fig = None
        combined_baselined_df = pd.DataFrame()
        print("⚠️ No baselined peak trace files found. Skipping trace plot.")

    # === Save outputs ===
    def safe_save(df, path, desc):
        if os.path.exists(path):
            print(f"⚠️ File exists, skipping save: {path}")
        else:
            df.to_feather(path)
            print(f"💾 Saved {desc} to:\n  {path}")

    safe_save(combined_amps_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_amplitudes.feather"), "combined amplitudes")
    safe_save(combined_aucs_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peak_aucs.feather"), "combined AUCs")
    safe_save(summary_df, os.path.join(folder, f"{filename_prefix}{rat}_combined_peaks_summary.feather"), "summary (feather)")

    txt_path = os.path.join(folder, f"{filename_prefix}{rat}_combined_peaks_summary.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(summary_df.to_string(index=False))
        print(f"💾 Saved summary (text) to:\n  {txt_path}")

    return summary_df, combined_amps_df, combined_aucs_df, combined_baselined_df, fig


  

# comparison plot of the various preprocessing methods --------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_streams_preproc(streams_df, ts, tzoom, padding=0.2):
    """
    Plot highpass-filtered GCaMP dF/F and z-scored signals in separate subplots,
    with y-limits based on data range.

    Parameters:
        streams_df (pd.DataFrame): The dataframe containing preprocessed streams.
        ts (array-like): The time vector.
        tzoom (tuple): Time window for zoomed-in plot.
        padding (float): Fraction of data range to pad y-limits by.
    """

    # Identify relevant columns
    dff_cols = [col for col in streams_df.columns 
                if 'GCaMP' in col and 'dF/F' in col]
    
    zscore_cols = [col for col in streams_df.columns 
                   if 'GCaMP' in col and 'z-score' in col]
    
    isos_cols = [col for col in streams_df.columns 
                 if 'isos' in col and 'z-score' in col]

    # For comparison plot: specifically look for airPLS versions
    gcamp_airpls_col = next((col for col in zscore_cols if 'airPLS' in col), None)
    isos_airpls_col = next((col for col in isos_cols if 'airPLS' in col), None)

    def get_ylim(columns):
        """Compute global min/max with padding."""
        data = np.concatenate([streams_df[col].values for col in columns])
        y_min, y_max = np.min(data), np.max(data)
        y_range = y_max - y_min
        return y_min - y_range * padding, y_max + y_range * padding

    dff_ylim = get_ylim(dff_cols) if dff_cols else (-1, 1)
    zscore_ylim = get_ylim(zscore_cols) if zscore_cols else (-1, 1)

    def get_color(col):
        """Assign color based on column name."""
        if ('GCaMP' in col) and ('z-score' in col):
            return [0.1, 0.7, 0.2]  # green

        elif 'isos' in col:
            return [0.7, 0.7, 0.7]  # gray

        elif ('GCaMP' in col) and ('z-score' not in col):
            return [0.1, 0.6, 0.7]  # blue

    # === Full-Length Plot ===
    fig_full, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for col in dff_cols:
        ax1.plot(ts, streams_df[col], label=col, color=get_color(col))
    ax1.set_title('GCaMP dF/F Signals')
    ax1.set_ylabel('dF/F')
    ax1.set_ylim(dff_ylim)
    ax1.legend(loc='upper right')

    for col in zscore_cols:
        ax2.plot(ts, streams_df[col], label=col, color=get_color(col))
    ax2.set_title('GCaMP z-scored Signals')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('z-score')
    ax2.set_ylim(zscore_ylim)
    ax2.legend(loc='upper right')

    plt.suptitle('Full-Length Filtered GCaMP Signals')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # === Zoomed-In Plot ===
    fig_zoom, (ax1z, ax2z) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for col in dff_cols:
        ax1z.plot(ts, streams_df[col], label=col, color=get_color(col))
    ax1z.set_title(f'dF/F Signals (Zoomed: {tzoom[0]}–{tzoom[1]} sec)')
    ax1z.set_ylabel('dF/F')
    ax1z.set_xlim(tzoom)
    ax1z.set_ylim(dff_ylim)
    ax1z.legend(loc='upper right')

    for col in zscore_cols:
        ax2z.plot(ts, streams_df[col], label=col, color=get_color(col))
    ax2z.set_title(f'z-scored Signals (Zoomed: {tzoom[0]}–{tzoom[1]} sec)')
    ax2z.set_xlabel('Time (seconds)')
    ax2z.set_ylabel('z-score')
    ax2z.set_xlim(tzoom)
    ax2z.set_ylim(zscore_ylim)
    ax2z.legend(loc='upper right')

    plt.suptitle('Zoomed-In Filtered GCaMP Signals')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # === Additional Plot for isosbestic z-scored signals ===
    isos_cols_present = [col for col in isos_cols if col in streams_df.columns]
    isos_ylim = get_ylim(isos_cols_present) if isos_cols_present else (-1, 1)

    fig_isos, ax_isos = plt.subplots(figsize=(12, 5))
    for col in isos_cols_present:
        ax_isos.plot(ts, streams_df[col], label=col, color=get_color(col))
    ax_isos.set_title('Isosbestic 405nm z-scored Signals')
    ax_isos.set_xlabel('Time (seconds)')
    ax_isos.set_ylabel('z-score')
    ax_isos.set_ylim(isos_ylim)
    ax_isos.set_xlim(tzoom)
    ax_isos.legend(loc='upper right')

    plt.tight_layout()

    # === Comparison Plot: airPLS GCaMP vs airPLS isos ===
    if gcamp_airpls_col and isos_airpls_col:
        compare_ylim = get_ylim([gcamp_airpls_col, isos_airpls_col])

        fig_compare, ax_compare = plt.subplots(figsize=(12, 5))
        ax_compare.plot(ts, streams_df[gcamp_airpls_col], label='GCaMP airPLS', color=[0.1, 0.7, 0.2])
        ax_compare.plot(ts, streams_df[isos_airpls_col], label='Isos airPLS', color=[0.7,0.7,0.7])
        ax_compare.set_title('Comparison: airPLS GCaMP vs airPLS Isosbestic (Zoomed)')
        ax_compare.set_xlabel('Time (seconds)')
        ax_compare.set_ylabel('z-score')
        ax_compare.set_xlim(tzoom)
        #ax_compare.set_ylim(compare_ylim)
        ax_compare.set_ylim([-10,10])
        ax_compare.legend(loc='upper right')
        ax_compare.set_facecolor('black')  # optional for white line visibility

        plt.tight_layout()
    else:
        fig_compare = None  # in case either column is missing

    return fig_full, fig_zoom, fig_isos, fig_compare







       


    
# load preprocessed epoc streams data for stats and plotting ---------------------------------------------------------------------    


def load_compute_plot(
    feather_folder, 
    file_pattern, 
    new_fs, 
    ts, 
    trange,
    baseline_trange=None,        # tuple: (start, end) in seconds
    cue_window=(0, 2.8),        # cue light window
    approach_window=None         # optional approach/AUC window
):
    from glob import glob
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # Find all matching feather files
    feather_files = glob(os.path.join(feather_folder, file_pattern))
    
    if not feather_files:
        raise FileNotFoundError(f"No files matched pattern '{file_pattern}' in folder '{feather_folder}'")

    print(f"Found {len(feather_files)} GCaMP_465_epocs files:\n")
    for i, file in enumerate(feather_files, start=1):
        print(f"{i}. {os.path.basename(file)}")
    print()

    # Store outputs
    file_dfs = {}
    file_stats = {}
    file_figs = {}

    # Process each file individually
    for file in feather_files:
        file_name = os.path.basename(file)
        print(f"Processing file: {file_name}")

        df = pd.read_feather(file)

        if df.shape[1] <= 2:
            print(f"⚠️  {file_name}: Not enough events for reliable stats (only {df.shape[1]} columns). Skipping stats.\n")
            stats_df = None
            df_baselined = df.copy()
        else:
            # ---------------------------
            # Baseline subtraction
            # ---------------------------
            if baseline_trange is not None:
                baseline_start_idx = int((baseline_trange[0] - trange[0]) * new_fs)
                baseline_end_idx   = int((baseline_trange[1] - trange[0]) * new_fs)
                baseline_start_idx = max(0, baseline_start_idx)
                baseline_end_idx   = min(df.shape[0], baseline_end_idx)
                baselines = df.iloc[baseline_start_idx:baseline_end_idx].mean()
                df_baselined = df.subtract(baselines, axis=1)
            else:
                df_baselined = df.copy()

            # ---------------------------
            # Compute mean, std, SEM
            # ---------------------------
            mean_epoc_stream = np.mean(df_baselined, axis=1)
            std_epoc_stream = np.nanstd(df_baselined, axis=1, ddof=1)
            sem_epoc_stream = stats.sem(df_baselined, axis=1, nan_policy="omit")

            stats_df = pd.DataFrame({
                'mean_epoc_stream': mean_epoc_stream,
                'std_epoc_stream': std_epoc_stream,
                'sem_epoc_stream': sem_epoc_stream
            })

        # Time vector for x-axis
        epoc_ts = trange[0] + np.arange(0, len(df)) / new_fs

        # ---------------------------
        # Plotting
        # ---------------------------
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot individual peri-event traces
        for col in df_baselined.columns:
            ax.plot(epoc_ts, df_baselined[col], color=(0.4, 0.4, 0.4), linewidth=0.5)
        ax.plot([], [], color=(0.4, 0.4, 0.4), linewidth=0.5, label='Peri-event traces')

        # Event onset line
        ax.plot([0, 0], [np.min(df_baselined.values), np.max(df_baselined.values)], 'r', linewidth=3, label='Event Onset')
        ax.axhline(0, color='white', linestyle=':', linewidth=1)

        # Cue light shading
        ax.fill_between([cue_window[0], cue_window[1]], 
                        np.max(stats_df['mean_epoc_stream'])*0.9, 
                        np.max(stats_df['mean_epoc_stream'])*1.0, 
                        color='yellow', alpha=0.3, linewidth=0,
                        edgecolor='none')
        ax.text(np.mean(cue_window), np.max(stats_df['mean_epoc_stream'])*1.03, 'cue light', 
                color='yellow', ha='center', va='bottom', fontsize=14)

        # Approach window shading (like cue light)
        if approach_window is not None:
            ax.fill_between([approach_window[0], approach_window[1]], 
                            np.max(stats_df['mean_epoc_stream'])*0.9, 
                            np.max(stats_df['mean_epoc_stream'])*1.0, 
                            color='cyan', alpha=0.3, linewidth=0,
                            edgecolor='none')
            ax.text(np.mean(approach_window), np.max(stats_df['mean_epoc_stream'])*1.03, 'approach', 
                    color='cyan', ha='center', va='bottom', fontsize=14)


        # Plot mean ± SEM
        if stats_df is not None:
            ax.fill_between(
                epoc_ts,
                stats_df['mean_epoc_stream'] + stats_df['sem_epoc_stream'],
                stats_df['mean_epoc_stream'] - stats_df['sem_epoc_stream'],
                facecolor='y', alpha=0.4, label='SEM'
            )
            ax.plot(epoc_ts, stats_df['mean_epoc_stream'], color=[1,1,0], linewidth=3, label='Mean Response')

        # Set y-limits
        max_val = np.max(np.abs(stats_df['mean_epoc_stream']))
        ax.set_ylim([-max_val*2, max_val*2])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence (dF/F)')
        ax.legend(loc='upper right')
        ax.set_title(f'Peri-event Traces: {file_name}', pad=30)

        # Store outputs
        file_dfs[file_name] = df_baselined
        file_stats[file_name] = stats_df
        file_figs[file_name] = fig
        
        print(f'There are {df.shape[1]} events in this data file.\n')

    return file_dfs, file_stats, file_figs






# save concatenated multiple recording segments for stats and plotting, across days or within long sessions ---------------------------------------------------------------------  



def save_combined_perievent_results(
    feather_folder,
    processed_phase,
    rat,
    eventname,
    combined_epocs,
    combined_stats,
    combined_events_info,
    per_epoc_stats=None,   # NEW (optional)
):
    """
    Saves combined peri-event analysis outputs to disk using structured filenames and safety checks.

    Parameters:
        feather_folder (str or list of str): One or more base directories for saving files.
        processed_phase (str): Identifier for the current session or phase.
        rat (str): Subject identifier or combined identifier (e.g. ratA-ratB).
        eventname (str): Name of the behavioral event (e.g., 'active_lever').
        combined_epocs (DataFrame): All aligned epoc fluorescence traces.
        combined_stats (DataFrame): Time-resolved mean, std, SEM DataFrame.
        combined_events_info (DataFrame): Summary info to save as both Feather and .txt.
        per_epoc_stats (DataFrame, optional): Baselined per-epoc window stats and AUCs.
    """

    import os

    def safe_feather_save(df, filepath):
        if os.path.exists(filepath):
            print(f"\nError: File already exists at {filepath}. Skipping save.\n")
            return
        if df is None:
            print(f"Warning: No DataFrame provided for {filepath}. Skipping save.")
            return
        if df.empty:
            print(f"Warning: The DataFrame at {filepath} is empty. Skipping save.")
            return

        df = df.reset_index(drop=True)
        df.columns = [str(col) for col in df.columns]
        df.to_feather(filepath)
        print(f"Saved to: {filepath}")

    # Handle both single and multi-folder input
    if isinstance(feather_folder, list):
        folder_name = '+'.join([os.path.basename(f) for f in feather_folder])
        save_dir = feather_folder[0]
    else:
        folder_name = os.path.basename(feather_folder)
        save_dir = feather_folder

    # Avoid duplication of processed_phase in folder_name
    if processed_phase not in folder_name:
        tag = f"{folder_name}_{processed_phase}_{rat}"
    else:
        tag = f"{folder_name}_{rat}"

    base_path = os.path.join(save_dir, tag)

    print(f"\nSaving results for: {rat}")
    print(f"Save base path: {base_path}\n")

    epocs_fp = base_path + f'_{eventname}_combined_GCaMP_465_epocs.feather'
    stats_fp = base_path + f'_{eventname}_combined_GCaMP_465_stats.feather'
    info_fp_txt  = base_path + f'_{eventname}_combined_events_info.txt'
    info_fp_feat = base_path + f'_{eventname}_combined_events_info.feather'
    per_epoc_fp  = base_path + f'_{eventname}_combined_GCaMP_465_per_epoc_stats.feather'

    safe_feather_save(combined_epocs, epocs_fp)
    safe_feather_save(combined_stats, stats_fp)
    safe_feather_save(combined_events_info, info_fp_feat)

    if per_epoc_stats is not None:
        safe_feather_save(per_epoc_stats, per_epoc_fp)

    # Save event info as text
    if os.path.exists(info_fp_txt):
        print(f"\nError: File already exists at {info_fp_txt}. Skipping save.\n")
    else:
        with open(info_fp_txt, 'w') as f:
            f.write(combined_events_info.to_string(index=False, col_space=25, justify='left'))
        print(f"Saved text info to: {info_fp_txt}")


# save concatenated multiple recording segments for stats and plotting, across days or within long sessions ---------------------------------------------------------------------    



def save_combined_perievent_results_012226(
    feather_folder,
    processed_phase,
    rat,
    eventname,
    combined_epocs,
    combined_stats,
    combined_events_info,
):
    """
    Saves combined peri-event analysis outputs to disk using structured filenames and safety checks.

    Parameters:
        feather_folder (str or list of str): One or more base directories for saving files.
        processed_phase (str): Identifier for the current session or phase.mean
        rat (str): Subject identifier.
        eventname (str): Name of the behavioral event (e.g., 'active_lever').
        combined_epocs (DataFrame): All aligned epoc fluorescence traces.
        combined_stats (DataFrame): Mean, std, sem DataFrame.
        combined_auc_df (DataFrame): AUC values (events as columns).
        combined_events_info (DataFrame): Summary info to save as both Feather and .txt.
    """

    def safe_feather_save(df, filepath):
        # Check if file already exists
        if os.path.exists(filepath):
            print()
            print(f"Error: File already exists at {filepath}. Skipping save to avoid overwriting.")
            print()
            return

        # Check if the DataFrame is None
        if df is None:
            print(f"Warning: No DataFrame provided for {filepath}. Skipping save.") 
            return

        # Check if the dataframe is empty
        if df.empty:
            print(f"Warning: The DataFrame at {filepath} is empty. Skipping save.")
            return

        df = df.reset_index(drop=True)                     # Reset index
        df.columns = [str(col) for col in df.columns]      # Ensure column names are strings
        df.to_feather(filepath)                             # Save to Feather
        print(f"Saved to: {filepath}")

    # Handle both single and multi-folder input
    if isinstance(feather_folder, list):
        # Combine folder names for output tag
        folder_name = '+'.join([os.path.basename(f) for f in feather_folder])
        save_dir = feather_folder[0]
    else:
        folder_name = os.path.basename(feather_folder)
        save_dir = feather_folder

    # Avoid duplication of processed_phase in folder_name
    if processed_phase not in folder_name:
        tag = f"{folder_name}_{processed_phase}_{rat}"
    else:
        tag = f"{folder_name}_{rat}"

    # Final save path
    base_path = os.path.join(save_dir, tag)

    print(f"\nSaving results for rat: {rat}")
    print(f"Save base path: {base_path}\n")


    # Construct filepaths
    epocs_fp     = base_path + f'_{eventname}_combined_GCaMP_465_epocs.feather'
    stats_fp     = base_path + f'_{eventname}_combined_GCaMP_465_stats.feather'
    info_fp_txt  = base_path + f'_{eventname}_combined_events_info.txt'
    info_fp_feat = base_path + f'_{eventname}_combined_events_info.feather'

    # Save dataframes safely
    safe_feather_save(combined_epocs, epocs_fp)
    safe_feather_save(combined_stats, stats_fp)
    safe_feather_save(combined_events_info, info_fp_feat)

    # Save event info as a text summary
    if os.path.exists(info_fp_txt):
        print(f"\nError: File already exists at {info_fp_txt}. Skipping save.\n")
    else:
        with open(info_fp_txt, 'w') as f:
            f.write(combined_events_info.to_string(index=False, col_space=25, justify='left'))
        print(f"Saved text info to: {info_fp_txt}")




# concatenate multiple recording segments for stats and plotting ---------------------------------------------------------------------    

def chunks_concat_compute_plot_030126(
        *,
        feather_folder=None,
        file_pattern=None,
        feather_files=None,
        new_fs=20.0,
        ts=None,
        trange=None,
        baseline_trange=None,
        cue_window=None,
        auc_pre_window=None,
        auc_post_window=None,
        approach_window=None,
        plot_auc_region=True,
        plot=True,
        per_animal_traces=None,
        per_animal_labels=None,
        subset = None,
        # --- NEW ONSET/SLOPE PARAMETERS ---
        onset_search_window=(0.05, 3),
        deriv_threshold=0.1,
        deriv_smooth_window=7,
        deriv_smooth_poly=6,
        consecutive_points=2,
        peak_fraction=0.5,
        min_peak_amplitude=-1,
        amp_onset_fraction=0.02,
        plot_onset_detection = False,
        master_xlim = None,
        master_ylim = None
):
    """
    Loads and concatenates GCaMP_465 epocs feather files, computes statistics,
    plots peri-event responses, AND computes per-epoc baselined window stats.

    Returns
    -------
    (
        combined_epocs,                  # raw concatenated data
        combined_epocs_baselined,        # baseline-subtracted data
        epoc_stats,                      # mean/SEM/std for plotting
        mean_epoc_stream,
        std_epoc_stream,
        sem_epoc_stream,
        fig_across,
        fig_within,
        None,
        (cue_mean, cue_sem),
        (auc_pre_mean, auc_pre_sem),
        (auc_post_mean, auc_post_sem),
        (approach_mean, approach_sem),
        per_epoc_stats_df               # per-column stats
    )
    """
    import os, glob, re
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from statsmodels.api import RLM, add_constant
    from scipy.stats import linregress

    # ---------------------
    # Gather files
    # ---------------------
    if feather_files is not None:
        feather_files = list(feather_files)
    else:
        if feather_folder is None:
            raise ValueError("Either 'feather_files' or 'feather_folder' must be provided")
        if isinstance(feather_folder, str):
            feather_folder = [feather_folder]
        feather_files = []
        for folder in feather_folder:
            if file_pattern is None:
                raise ValueError("'file_pattern' must be provided if feather_files is None")
            feather_files.extend(glob.glob(os.path.join(folder, file_pattern)))

    if not feather_files:
        raise FileNotFoundError("No feather files found")

    # ---------------------
    # Load and concatenate
    # ---------------------
    epoc_dfs = []
    epoc_rats = []
    epoc_sources_per_df = []

    for f in feather_files:
        df = pd.read_feather(f)
        
        # ---------------------
        # Optional subsetting
        # ---------------------
        if subset is not None:
            n_cols = df.shape[1]

            if isinstance(subset, int):
                if subset > 0:
                    # First N
                    df = df.iloc[:, :min(subset, n_cols)]
                elif subset < 0:
                    # Last N
                    df = df.iloc[:, max(0, n_cols + subset):]

            elif isinstance(subset, (list, tuple)) and len(subset) == 2:
                start, end = subset

                # Convert 1-indexed inclusive to 0-indexed slice
                start_idx = max(start - 1, 0)
                end_idx   = min(end, n_cols)

                if start_idx < end_idx:
                    df = df.iloc[:, start_idx:end_idx]
                else:
                    df = df.iloc[:, 0:0]  # empty safely

            else:
                raise ValueError("subset must be int, negative int, or [start, end]")

        if df.shape[1] == 0:
            continue

        fname = Path(f).stem
        rat_match = re.search(r'ACW_coh5_IT_[fm]\d+', fname)
        rat_id = rat_match.group(0) if rat_match else "unknown_rat"

        df.columns = [f"{rat_id}_{col}" for col in df.columns]
        epoc_dfs.append(df)
        epoc_rats.append(rat_id)

        # Track source file for each column
        epoc_sources_per_df.append([Path(f).name] * df.shape[1])
        
    if len(epoc_dfs) == 0:
        raise ValueError("No epocs remaining after subsetting/baseline filtering.")

    combined_epocs = pd.concat(epoc_dfs, axis=1).reset_index(drop=True)

    # Deduplicate columns if needed
    def dedupe_columns(cols):
        seen = {}
        out = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                out.append(c)
            else:
                seen[c] += 1
                out.append(f"{c}__{seen[c]}")
        return out

    combined_epocs.columns = dedupe_columns(combined_epocs.columns)

    # ---------------------
    # Baseline correction
    # ---------------------
    if baseline_trange is not None:
        baseline_start_idx = int((baseline_trange[0] - trange[0]) * new_fs)
        baseline_end_idx   = int((baseline_trange[1] - trange[0]) * new_fs)

        filtered_epoc_dfs = []
        filtered_sources = []

        for df, src_list in zip(epoc_dfs, epoc_sources_per_df):

            baseline_slice = df.iloc[baseline_start_idx:baseline_end_idx]
            valid_fraction = baseline_slice.notna().sum(axis=0) / len(baseline_slice)
            keep_cols = valid_fraction[valid_fraction >= 0.1].index

            filtered_df = df[keep_cols]
            filtered_epoc_dfs.append(filtered_df)

            # Filter matching sources correctly
            keep_mask = df.columns.isin(keep_cols)
            filtered_sources.extend(np.array(src_list)[keep_mask].tolist())

        if len(filtered_epoc_dfs) == 0 or all(df.shape[1] == 0 for df in filtered_epoc_dfs):
            raise ValueError("No epocs remaining after baseline filtering.")

        combined_epocs = pd.concat(filtered_epoc_dfs, axis=1).reset_index(drop=True)
        combined_epocs.columns = dedupe_columns(combined_epocs.columns)

        baseline_slice = combined_epocs.iloc[baseline_start_idx:baseline_end_idx]
        baselines = baseline_slice.mean()
        combined_epocs_baselined = combined_epocs.subtract(baselines, axis=1)
        data_to_use = combined_epocs_baselined

        epoc_sources = filtered_sources  # <- update the sources list
    else:
        combined_epocs_baselined = combined_epocs.copy()
        data_to_use = combined_epocs_baselined


    # After baseline correction
    if trange is not None:
        start_idx = max(int((trange[0] - trange[0]) * new_fs), 0)  # 0 offset
        end_idx   = min(int((trange[1] - trange[0]) * new_fs), data_to_use.shape[0])
        data_to_use = data_to_use.iloc[start_idx:end_idx]

    # ---------------------
    # Time vector
    # ---------------------
    epoc_ts = trange[0] + np.arange(len(data_to_use)) / new_fs

    #epoc_ts = np.linspace(trange[0], trange[1], data_to_use.shape[0])
    
    ###################################

    dt = epoc_ts[1] - epoc_ts[0]

    # ============================================================
    # 🔬 ONSET + SLOPE DETECTION
    # ============================================================
    
    
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.tools import add_constant
    from scipy.signal import savgol_filter
    import numpy as np

    
    
    def detect_onset_and_slope(
            y, epoc_ts, dt,
            onset_search_window,
            deriv_threshold,
            deriv_smooth_window,
            deriv_smooth_poly,
            consecutive_points,
            peak_fraction,
            min_peak_amplitude,
            amp_onset_fraction
    ):
        """
        Detect onset, rising slope, time-to-peak, and peak amplitude
        using derivative + amplitude logic with full numerical safeguards.

        Returns
        -------
        onset_time : float
        slope : float
        time_to_peak : float
        peak_val : float
        """

        import numpy as np
        from scipy.signal import savgol_filter
        from statsmodels.robust.robust_linear_model import RLM
        from statsmodels.tools import add_constant

        # -----------------------------
        # Restrict to onset search window
        # -----------------------------
        mask = (epoc_ts >= onset_search_window[0]) & (epoc_ts <= onset_search_window[1])
        if mask.sum() < 3:
            return np.nan, np.nan, np.nan, np.nan

        y_win = y[mask]
        ts_win = epoc_ts[mask]

        # -----------------------------
        # Remove NaNs early
        # -----------------------------
        if np.all(np.isnan(y_win)):
            return np.nan, np.nan, np.nan, np.nan

        if np.any(np.isnan(y_win)):
            return np.nan, np.nan, np.nan, np.nan


        # -----------------------------
        # Smooth signal for peak detection
        # -----------------------------
        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y_win, sigma=2)

        # -----------------------------
        # Peak detection (on smoothed signal)
        # -----------------------------
        peak_idx = np.nanargmax(y_smooth)
        peak_val = y_smooth[peak_idx]
        peak_time = ts_win[peak_idx]
        
        
        if peak_val < min_peak_amplitude:
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Validate Savitzky–Golay window
        # -----------------------------
        max_window = len(y_win)
        if max_window % 2 == 0:
            max_window -= 1  # must be odd

        window_length = min(deriv_smooth_window, max_window)

        if window_length <= deriv_smooth_poly:
            return np.nan, np.nan, np.nan, np.nan

        if window_length < 3:
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Compute derivative safely
        # -----------------------------
        try:
            dydt = savgol_filter(
                y_win,
                window_length=window_length,
                polyorder=deriv_smooth_poly,
                deriv=1,
                delta=dt
            )
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

        if np.all(np.isnan(dydt)):
            return np.nan, np.nan, np.nan, np.nan

        # -----------------------------
        # Dynamic derivative threshold
        # -----------------------------
        max_dydt = np.nanmax(dydt)
        dynamic_threshold = max(deriv_threshold, 0.1 * max_dydt)
        above = dydt > dynamic_threshold

        # -----------------------------
        # Derivative-based onset
        # -----------------------------
        onset_idx_deriv = None
        for i in range(len(above) - consecutive_points + 1):
            if np.all(above[i:i + consecutive_points]):
                onset_idx_deriv = i
                break

        # -----------------------------
        # Amplitude-based onset
        # -----------------------------
        amp_threshold = amp_onset_fraction * peak_val
        onset_candidates_amp = np.where(y_win >= amp_threshold)[0]
        onset_idx_amp = onset_candidates_amp[0] if len(onset_candidates_amp) > 0 else None

        # -----------------------------
        # Combine logic
        # -----------------------------
        if onset_idx_deriv is None and onset_idx_amp is None:
            return np.nan, np.nan, np.nan, np.nan

        if onset_idx_deriv is None:
            onset_idx = onset_idx_amp
        elif onset_idx_amp is None:
            onset_idx = onset_idx_deriv
        else:
            onset_idx = min(onset_idx_deriv, onset_idx_amp)

        onset_time = ts_win[onset_idx]

        # -----------------------------
        # Time-to-peak
        # -----------------------------
        time_to_peak = peak_time - onset_time

        if time_to_peak <= 0:
            return onset_time, np.nan, np.nan, peak_val

        # -----------------------------
        # Rising slope fit window
        # -----------------------------
        threshold_val = peak_fraction * peak_val
        above_thresh = np.where(y_win >= threshold_val)[0]
        valid_end = above_thresh[above_thresh > onset_idx]

        end_idx = valid_end[0] if len(valid_end) > 0 else peak_idx

        if end_idx <= onset_idx:
            return onset_time, np.nan, time_to_peak, peak_val

        x_fit = ts_win[onset_idx:end_idx + 1]
        y_fit = y_win[onset_idx:end_idx + 1]

        if len(x_fit) < 3:
            return onset_time, np.nan, time_to_peak, peak_val

        # -----------------------------
        # Robust linear fit
        # -----------------------------
        try:
            X = add_constant(x_fit)
            rlm_model = RLM(y_fit, X)
            slope = rlm_model.fit().params[1]
        except Exception:
            slope = np.nan

        return onset_time, slope, time_to_peak, peak_val

    onset_dict = {}
    slope_dict = {}
    time_to_peak_dict = {}
    peak_amp_dict = {}
    
    dt = epoc_ts[1] - epoc_ts[0]

    for col in data_to_use.columns:
        y = data_to_use[col].values
        onset, slope, ttp, peak_amp = detect_onset_and_slope(
            y, epoc_ts, dt,
            onset_search_window=onset_search_window,
            deriv_threshold=deriv_threshold,
            deriv_smooth_window=deriv_smooth_window,
            deriv_smooth_poly=deriv_smooth_poly,
            consecutive_points=consecutive_points,
            peak_fraction=peak_fraction,
            min_peak_amplitude=min_peak_amplitude,
            amp_onset_fraction=amp_onset_fraction
        )
        onset_dict[col] = onset
        slope_dict[col] = slope
        time_to_peak_dict[col] = ttp
        peak_amp_dict[col] = peak_amp


    
    # ---------------------
    # Per-epoc stats (fixed for proper alignment)
    # ---------------------
    per_epoc_stats_df = pd.DataFrame(index=data_to_use.columns)
    per_epoc_stats_df.index.name = "epoc_id"

    # Map onset/slope
    per_epoc_stats_df["response_latency_sec"] = [onset_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["rising_slope"] = [slope_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["time_to_peak_sec"] = [time_to_peak_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["peak_amplitude"] = [peak_amp_dict.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]


    # ---------------------
    
    # Per-rat mean + SEM
    # ---------------------
    latency_vals = per_epoc_stats_df["response_latency_sec"].values
    slope_vals = per_epoc_stats_df["rising_slope"].values
    time_to_peak_vals = per_epoc_stats_df["time_to_peak_sec"].values
    peak_amp_vals = per_epoc_stats_df["peak_amplitude"].values
    

    onset_mean = np.nanmean(latency_vals)
    onset_sem = stats.sem(latency_vals, nan_policy="omit")

    slope_mean = np.nanmean(slope_vals)
    slope_sem = stats.sem(slope_vals, nan_policy="omit")
    
    time_to_peak_mean = np.nanmean(time_to_peak_vals)
    time_to_peak_sem = stats.sem(time_to_peak_vals, nan_policy="omit")

    peak_amp_mean = np.nanmean(peak_amp_vals)
    peak_amp_sem = stats.sem(peak_amp_vals, nan_policy="omit")


    per_rat_onset_stats = {
        "latency_mean": onset_mean,
        "latency_sem": onset_sem,
        "slope_mean": slope_mean,
        "slope_sem": slope_sem,
        "time_to_peak_mean": time_to_peak_mean,
        "time_to_peak_sem":  time_to_peak_sem,
        "peak_amp_mean": peak_amp_mean,
        "peak_amp_sem": peak_amp_sem
    }

    # ============================================================
    # OPTIONAL VISUALIZATION
    # ============================================================

    fig_onset = None

    if plot_onset_detection and per_animal_traces is not None and len(per_animal_traces) > 0:
        fig_onset, ax = plt.subplots(figsize=(8,6))
        for col in data_to_use.columns:
            y = data_to_use[col].values
            ax.plot(epoc_ts, y, alpha=0.3)

            onset = onset_dict[col]
            slope = slope_dict[col]

            if not np.isnan(onset) and not np.isnan(slope):
                ax.axvline(onset, linestyle='--', alpha=0.5)
                ax.axvline(0, linestyle="-", color="red", label="event")

                y0 = np.interp(onset, epoc_ts, y)
                x_line = np.linspace(onset, onset + 1.0, 50)
                y_line = y0 + slope * (x_line - onset)
                ax.plot(x_line, y_line, linewidth=2)

        ax.set_title("Response onset and slope detection")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Fluorescence (z-score)")
        ax.set_xlim(-0.5,3)
        
    ###################################
    
    
    # ---------------------
    # Mean / std / SEM
    # ---------------------
    n_trials = data_to_use.shape[1]
    
    mean_epoc_stream = np.nanmean(data_to_use, axis=1)
    if n_trials > 1:
        std_epoc_stream = np.nanstd(data_to_use, axis=1, ddof=1)
        sem_epoc_stream = stats.sem(data_to_use, axis=1, nan_policy='omit')
    else:
        std_epoc_stream = np.zeros_like(mean_epoc_stream)
        sem_epoc_stream = np.zeros_like(mean_epoc_stream)

    # ---------------------
    # Window-level stats
    # ---------------------
    def window_stats(window):
        if window is None:
            return np.nan, np.nan
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        wdata = data_to_use.iloc[start_idx:end_idx]
        if wdata.empty:
            return np.nan, np.nan
        mean_val = wdata.mean().mean()
        sem_val  = stats.sem(wdata.mean(), nan_policy='omit')
        return mean_val, sem_val

    cue_mean, cue_sem = window_stats(cue_window)
    approach_mean, approach_sem = window_stats(approach_window)

    # ---------------------
    # AUC
    # ---------------------
    def compute_auc(window):
        if window is None:
            return np.nan, np.nan
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        auc_vals = []
        for col in data_to_use.columns:
            y = data_to_use[col].iloc[start_idx:end_idx].values
            x = epoc_ts[start_idx:end_idx]
            finite_mask = np.isfinite(y)
            if finite_mask.sum() < 2:
                continue
            auc = np.trapz(y[finite_mask], x[finite_mask])
            if np.isfinite(auc):
                auc_vals.append(auc)
        if len(auc_vals) == 0:
            return np.nan, np.nan
        auc_vals = np.array(auc_vals)
        mean_auc = np.mean(auc_vals)
        sem_auc  = stats.sem(auc_vals) if len(auc_vals) > 1 else np.nan
        return mean_auc, sem_auc

    auc_pre_mean, auc_pre_sem   = compute_auc(auc_pre_window)
    auc_post_mean, auc_post_sem = compute_auc(auc_post_window)

    # ---------------------
    # Per-epoc stats
    # ---------------------
    def per_epoc_window_mean(window):
        if window is None:
            return {}
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        return {col: data_to_use[col].iloc[start_idx:end_idx].mean() for col in data_to_use.columns}

    def per_epoc_auc(window):
        if window is None:
            return {}
        start_idx = np.searchsorted(epoc_ts, window[0])
        end_idx   = np.searchsorted(epoc_ts, window[1])
        out = {}
        for col in data_to_use.columns:
            y = data_to_use[col].iloc[start_idx:end_idx].values
            x = epoc_ts[start_idx:end_idx]
            if len(x) == len(y) and np.any(np.isfinite(y)):
                out[col] = float(np.trapz(y, x))
        return out

    
    # Map window means
    cue_vals = per_epoc_window_mean(cue_window)
    approach_vals = per_epoc_window_mean(approach_window)
    auc_pre_vals = per_epoc_auc(auc_pre_window)
    auc_post_vals = per_epoc_auc(auc_post_window)

    per_epoc_stats_df["cue_mean"] = [cue_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["approach_mean"] = [approach_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["auc_pre"] = [auc_pre_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]
    per_epoc_stats_df["auc_post"] = [auc_post_vals.get(epoc, np.nan) for epoc in per_epoc_stats_df.index]

    # Add source file
    per_epoc_stats_df["source_file"] = epoc_sources

    # Reset index if you want epoc_id as a column
    per_epoc_stats_df = per_epoc_stats_df.reset_index()
    
    # ---------------------
    # Stats DataFrame (time-resolved)
    # ---------------------
    epoc_stats = pd.DataFrame({
        "mean_epoc_stream": mean_epoc_stream,
        "std_epoc_stream": std_epoc_stream,
        "sem_epoc_stream": sem_epoc_stream
    })

    # ---------------------
    # Plotting (optional)
    # ---------------------
    fig_trial = None
    fig_across = None
    fig_within = None

    mean_across_animal = None
    sem_across_animal  = None

    if per_animal_traces is not None and len(per_animal_traces) > 0:
        all_means = np.array([trace["mean"] for trace in per_animal_traces])
        mean_across_animal = np.nanmean(all_means, axis=0)
        sem_across_animal  = np.nanstd(all_means, axis=0, ddof=1) / np.sqrt(all_means.shape[0])

    if plot:
        import matplotlib.cm as cm

        # ------------------------------------------------------------
        # FIGURE 1 — ACROSS-TRIAL
        # ------------------------------------------------------------
        fig_trial, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axvline(0, color='r', linewidth=1)
        ax.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8)

        ax.plot(epoc_ts, mean_epoc_stream, color='yellow', linewidth=3, label='Mean response', zorder=11)
        ax.fill_between(epoc_ts,
                        mean_epoc_stream - sem_epoc_stream,
                        mean_epoc_stream + sem_epoc_stream,
                        color='yellow', alpha=0.4, label='SEM', zorder=10)

        # Y-limits
        k = 8
        y_min = np.nanmin(mean_epoc_stream - k * sem_epoc_stream)
        y_max = np.nanmax(mean_epoc_stream + k * sem_epoc_stream)
        ax.set_ylim(y_min, y_max)
        top_bar = y_max * 0.95

        # Cue and approach bars
        for window, color, label in zip([cue_window, approach_window],
                                        ['yellow', 'magenta'],
                                        ['cue', 'approach']):
            if window is not None:
                start, end = window
                x = epoc_ts[(epoc_ts >= start) & (epoc_ts <= end)]
                if x.size:
                    ax.fill_between(x, top_bar*0.98, top_bar, color=color, alpha=0.7)
                    ax.text(np.mean(x), top_bar*1.01, label, color=color,
                            ha='center', va='bottom', fontsize=14)

        # AUC shading
        if plot_auc_region:
            def safe_fill(window, color):
                start, end = window
                i0 = np.searchsorted(epoc_ts, start)
                i1 = np.searchsorted(epoc_ts, end)
                if i0 >= i1:
                    return
                y = mean_epoc_stream[i0:i1]
                if np.all(np.isnan(y)):
                    return
                y = pd.Series(y).interpolate(limit_direction='both').to_numpy()
                ax.fill_between(epoc_ts[i0:i1], 0, y, color=color, alpha=0.5)

            if auc_pre_window is not None:
                safe_fill(auc_pre_window, 'green')
            if auc_post_window is not None:
                safe_fill(auc_post_window, 'blue')

        ax.set_title('Across-trial mean ± SEM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence (z-score)')
        ax.set_xlim(master_xlim)
        ax.set_ylim (master_ylim)
        ax.legend()


        # ------------------------------------------------------------
        # FIGURE 2 — ACROSS-ANIMAL
        # ------------------------------------------------------------
        fig_across, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axvline(0, color='r', linewidth=1)
        ax.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8)

        if mean_across_animal is not None:
            ax.plot(epoc_ts, mean_across_animal, color='yellow', linewidth=3, label='Mean across animals')
            ax.fill_between(epoc_ts,
                            mean_across_animal - sem_across_animal,
                            mean_across_animal + sem_across_animal,
                            color='yellow', alpha=0.4, label='SEM')
        else:
            ax.plot(epoc_ts, mean_epoc_stream, color='yellow', linewidth=3, label='Mean response')
            ax.fill_between(epoc_ts,
                            mean_epoc_stream - sem_epoc_stream,
                            mean_epoc_stream + sem_epoc_stream,
                            color='yellow', alpha=0.4, label='SEM')

        # Cue and approach bars
        for window, color, label in zip([cue_window, approach_window],
                                        ['yellow', 'magenta'],
                                        ['cue', 'approach']):
            if window is not None:
                start, end = window
                x = epoc_ts[(epoc_ts >= start) & (epoc_ts <= end)]
                if x.size:
                    ax.fill_between(x, top_bar*0.98, top_bar, color=color, alpha=0.7)
                    ax.text(np.mean(x), top_bar*1.01, label, color=color,
                            ha='center', va='bottom', fontsize=14)

        # AUC shading
        # AUC shading
        if plot_auc_region:

            def safe_fill(window, color):
                start, end = window
                i0 = np.searchsorted(epoc_ts, start)
                i1 = np.searchsorted(epoc_ts, end)
                if i0 >= i1:
                    return

                # Use across-animal mean if available
                if mean_across_animal is not None:
                    y_source = mean_across_animal
                else:
                    y_source = mean_epoc_stream

                y = y_source[i0:i1]

                if np.all(np.isnan(y)):
                    return

                y = pd.Series(y).interpolate(limit_direction='both').to_numpy()
                ax.fill_between(epoc_ts[i0:i1], 0, y, color=color, alpha=0.5)

            if auc_pre_window is not None:
                safe_fill(auc_pre_window, 'green')
            if auc_post_window is not None:
                safe_fill(auc_post_window, 'blue')


        ax.set_ylim(y_min, y_max)
        ax.set_title('Across-animal mean ± SEM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence (z-score)')
        ax.set_xlim(master_xlim)
        ax.set_ylim(master_ylim)
        ax.legend()


        # ------------------------------------------------------------
        # FIGURE 3 — WITHIN-ANIMAL
        # ------------------------------------------------------------
        fig_within = None
        if per_animal_traces is not None and len(per_animal_traces) > 0:
            fig_within, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.axvline(0, color='r', linewidth=1)
            ax.axhline(0, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.8)

            # Plot per-animal traces
       
            cmap_name = "viridis"   # <- easily change this
            cmap = cm.get_cmap(cmap_name)
            colors = cmap(np.linspace(0.1, 0.9, len(per_animal_traces)))
            
            for trace, c in zip(per_animal_traces, colors):
                mean = trace["mean"]
                sem  = trace["sem"]
                rat  = trace["rat"]
                ax.plot(epoc_ts, mean, color=c, linewidth=2.5, alpha=0.9, label=rat)
                ax.fill_between(epoc_ts, mean - sem, mean + sem, color=c, alpha=0.25, linewidth=0,
                edgecolor='none')

            # -----------------------------
            # Add cue/approach bars
            # -----------------------------
            top_bar = np.nanmax([trace["mean"] + trace["sem"] for trace in per_animal_traces])
            for window, color, label in zip([cue_window, approach_window],
                                            ['yellow', 'magenta'],
                                            ['cue', 'approach']):
                if window is not None:
                    start, end = window
                    x = epoc_ts[(epoc_ts >= start) & (epoc_ts <= end)]
                    if x.size:
                        ax.fill_between(x, top_bar*0.98, top_bar, color=color, alpha=0.7)
                        ax.text(np.mean(x), top_bar*1.01, label, color=color,
                                ha='center', va='bottom', fontsize=14)

            ax.set_title('Within-animal mean ± SEM')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Fluorescence (z-score)')
            ax.set_xlim(master_xlim)
            ax.legend()

    return (
        combined_epocs,
        combined_epocs_baselined,
        epoc_stats,
        mean_epoc_stream,
        std_epoc_stream,
        sem_epoc_stream,
        fig_trial,
        fig_across,
        fig_within,
        fig_onset,
        (cue_mean, cue_sem),
        (auc_pre_mean, auc_pre_sem),
        (auc_post_mean, auc_post_sem),
        (approach_mean, approach_sem),
        (onset_mean, onset_sem),
        (slope_mean, slope_sem),
        (time_to_peak_mean, time_to_peak_sem),
        (peak_amp_mean, peak_amp_sem),
        per_epoc_stats_df
    )





# downsampling the preprocessed data --------------------------------------------------------------------------------------------


def downsamp (streams_preproc, fs, new_fs):
    
    N = int(fs/new_fs)
    print(f'the new sampling rate is {new_fs} Hz, so the window size rounds to {N} data points')

    streams_preproc_downsamp = pd.DataFrame()

    for column in streams_preproc.columns: 
    
        col_downsamp = []
        col_data = streams_preproc[column]
    
        for i in range(0, len(col_data), N):
            col_downsamp.append(np.mean(col_data[i:i+N]))
        
        streams_preproc_downsamp[column]=col_downsamp

    print("streams_preproc shape:", streams_preproc.shape)
    print("streams_preproc_downsamp shape:", streams_preproc_downsamp.shape)
    
    
    fig_8 = plt.plot(streams_preproc['ts'], streams_preproc['GCaMP 465nm dF/F'],color='b',linewidth=10, alpha = 0.3)
    plt.plot(streams_preproc_downsamp['ts'], streams_preproc_downsamp['GCaMP 465nm dF/F'],color='r')
    plt.xlim([0,3])
    plt.ylim([-5,5])
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('GCaMP_465 Signal (dF/F)', color='r')
    plt.title('downsampled GCaMP signals, dF/F')
    
    
    return streams_preproc_downsamp, fig_8






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    

def plot_streams_with_events(isos_405_downsamp, GCaMP_465_downsamp, ts_downsamp, epoc_data, box):
    """
    Plots downsampled GCaMP and isosbestic signals with selected event ticks from epoc_data.

    Parameters:
    - isos_405_downsamp: np.array, downsampled isosbestic signal
    - GCaMP_465_downsamp: np.array, downsampled GCaMP signal
    - ts_downsamp: np.array, time vector corresponding to signals
    - epoc_data: pd.DataFrame, event data with columns: blockname, start_date, name, onset, offset, data
    - box: str, value in 'blockname' to filter which events to show

    Returns:
    - fig: matplotlib Figure of the full signal with event annotations
    """

    plt.style.use('dark_background')

    # Event code map (restricted to displayable codes)
    event_labels = {
        2: 'drug available onset',
        3: 'trial begin + drug available onset',
        4: 'unavailable onset',
        8: 'active lever',
        16: 'inactive lever',
        32: 'infusion',
        40: 'active lever + infusion',
        128: 'active lever drug unavailable'
    }

    event_colors = {
        2: 'deepskyblue',
        3: 'steelblue',
        4: 'orange',
        8: 'lime',
        16: 'white',
        32: 'magenta',
        40: 'red',
        128: 'yellow'
    }

    # Filter by 'box' and desired event codes
    df_filtered = epoc_data[
        (epoc_data['name'] == box) &
        (epoc_data['data'].isin(event_labels.keys()))
    ]

    # Set y-limits
    GCaMP_ylim = [np.min(GCaMP_465_downsamp) * 0.9, np.max(GCaMP_465_downsamp) * 1.1]
    isos_ylim = [np.min(isos_405_downsamp) * 0.9, np.max(isos_405_downsamp) * 1.1]

    # Create main plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot GCaMP signal
    ax1.plot(ts_downsamp, GCaMP_465_downsamp, color=[0.1, 0.7, 0.2], label='GCaMP_465')
    ax1.set_ylim(GCaMP_ylim)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GCaMP_465 Signal (dF/F)', color='g')

    # Plot isos signal on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(ts_downsamp, isos_405_downsamp, color=[0.7, 0.7, 0.7], label='isos_405')
    ax2.set_ylim(isos_ylim)
    ax2.set_ylabel('isos_405 Signal (dF/F)', color='gray')

    # Overlay event ticks
    for code, label in event_labels.items():
        onset_times = df_filtered[df_filtered['data'] == code]['onset'].values
        for t in onset_times:
            ax1.axvline(t, color=event_colors[code], linestyle='-', ymin=0.9, ymax=1.0, lw=2)
        if len(onset_times) > 0:
            ax1.plot([], [], color=event_colors[code], label=label)  # dummy plot for legend

    # Legend and styling
    ax1.legend(loc='best', fontsize=8)
    plt.title(f'Downsampled Streams with Behavioral Events — {box}')
    plt.tight_layout()
    return fig
    
    
    
    

    
# fitting and normalization ------------------------------------------------------------------------------------------------------------
    
def controlFit(control, signal):
    
    # GuPPY version
    # function to fit control channel to signal channel

    p = np.polyfit(control, signal, 1)
    arr = (p[0]*control)+p[1]
    return arr


def deltaFF_old(signal, control):
    
    # function to compute deltaF/F using fitted control channel and filtered signal channel

    res = np.subtract(signal, control) # numerator of F(t)-f0
    normData = np.divide(res, control) # (F(t)-f0)/F0
    normData = normData*100

    return normData




def norm_plot(isos_405, GCaMP_465, fitted_control, corrected_data, ts, tzoom):
    
    plt.figure(figsize=(10,7))
    fig_6,ax1=plt.subplots()  # create a plot to allow for dual y-axes plotting
    plot1=ax1.plot(ts, GCaMP_465, color=[0,0.43,0.59]) 
    plot1=ax1.plot(ts, fitted_control, color=[1,0.40,0])
    ax2=plt.twinx()# create a right y-axis, sharing x-axis on the same plot
    plot2=ax2.plot(ts, isos_405, color=[0.35,0.10,0.50]) 

    #plt.plot(ts, corrected_data,color='r')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('465 fluor.')
    ax1.set_title('lls fit')
    ax2.set_ylabel('405 fluor.')

    GCaMP_ylim = [(np.min(GCaMP_465)*0.9),(1.1*np.max(GCaMP_465))]
    isos_ylim = [(np.min(isos_405)*0.9),(1.1*np.max(isos_405))]


    ax1.set_ylim(GCaMP_ylim)
    ax2.set_ylim(isos_ylim)
    plt.xlim(tzoom)
  

    legend = fig_6.legend(['GCaMP_465_filt','fitted_control','isos_405_filt'],loc='upper right', bbox_to_anchor=(0.9, 0.88))
    #legend = fig.legend(['GCaMP_465_filt','isos_405_filt'],loc='upper right', bbox_to_anchor=(0.9, 0.88))
    
    # create a plot that zooms in on the normalized signal

    plt.figure(figsize=(10,7))
    fig_7 = plt.plot(ts, corrected_data,color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence (dF/F)')
    plt.xlim(tzoom) # we can zoom in to see how motion artifacts can be corrected for
    plt.ylim([-10, 10]) # we can zoom in to see how motion artifacts can be corrected for

    plt.legend(['GCaMP_465_dFF']);
    

    
    return fig_6, fig_7






# motion correction using a linear fit between isosbestic and GCaMP streams ------------------------------------------------------------------------------------------------------------

def motcorr_polyfit(ts, isos_405, GCaMP_465, tzoom):
    
    # GuPPY version
    # function to fit control channel to signal channel

    p = np.polyfit(isos_405, GCaMP_465, 1)
    arr = (p[0]*isos_405)+p[1]
        
    GCaMP_465_motcorr = np.subtract(GCaMP_465, arr) # numerator of F(t)-f0

    
    # function to compute deltaF/F using fitted control channel and filtered signal channel


    fig_12 = plt.scatter(isos_405[::5], GCaMP_465[::5],alpha=0.8, marker='.')
    x = np.array(plt.xlim())
    plt.xlabel('isos_405_expcorr')
    plt.ylabel('GCaMP_465_expcorr')
    plt.title('isos_405 - GCaMP_465 correlation.')




    fig_13,ax1=plt.subplots()  
    plot1=ax1.plot(ts, GCaMP_465, color=[1,0,1] , label='GCaMP - pre motion correction', alpha=0.8)
    plot2=ax1.plot(ts, isos_405, color=[1,1,1], label='isosbestic')
    plot3=ax1.plot(ts, GCaMP_465_motcorr, color=[0.15,0.6,0.2], label='GCaMP - motion corrected')
    plot4=ax1.plot(ts, arr, color=[1,1,0], label='estimated motion')

                   
                   
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GCaMP Signal (mV)', color='g')
    ax1.set_title('Motion Correction')

    lines = plot1+plot2+plot3+plot4
    labels = [l.get_label() for l in lines]  
    legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.95, 0.98))

    #GCaMP_ylim = [(np.min(GCaMP_465_expcorr_motcorr)*4),(4*np.max(GCaMP_465_expcorr_motcorr))]

    
    ax1.set_xlim(tzoom) 
    #ax1.set_ylim(GCaMP_ylim);


    return GCaMP_465_motcorr, arr, fig_12, fig_13

# motion correction using a linear fit between isosbestic and GCaMP streams ------------------------------------------------------------------------------------------------------------

def motcorr_linreg(ts, isos_405_expcorr, GCaMP_465_expcorr, tzoom):

    slope, intercept, r_value, p_value, std_err = linregress(x=isos_405_expcorr, y=GCaMP_465_expcorr)

    fig_12 = plt.scatter(isos_405_expcorr[::5], GCaMP_465_expcorr[::5],alpha=0.8, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercept+slope*x)
    plt.xlabel('isos_405_expcorr')
    plt.ylabel('GCaMP_465_expcorr')
    plt.title('isos_405 - GCaMP_465 correlation.')

    print('Slope    : {:.3f}'.format(slope))
    print('R-squared: {:.3f}'.format(r_value**2))

    
    GCaMP_465_est_motion = intercept + slope * isos_405_expcorr
    GCaMP_465_expcorr_motcorr = GCaMP_465_expcorr - GCaMP_465_est_motion   

    fig_13,ax1=plt.subplots()  
    plot1=ax1.plot(ts, GCaMP_465_expcorr, color=[1,0,1] , label='GCaMP - pre motion correction', alpha=0.8)
    plot2=ax1.plot(ts, isos_405_expcorr, color=[1,1,1], label='isosbestic')
    plot3=ax1.plot(ts, GCaMP_465_expcorr_motcorr, color=[0.15,0.6,0.2], label='GCaMP - motion corrected')
    plot4=ax1.plot(ts, GCaMP_465_est_motion, color=[1,1,0], label='estimated motion')

                   
                   
                   
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GCaMP Signal (mV)', color='g')
    ax1.set_title('Motion Correction')

    lines = plot1+plot2+plot3+plot4
    labels = [l.get_label() for l in lines]  
    legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.95, 0.98))

    #GCaMP_ylim = [(np.min(GCaMP_465_expcorr_motcorr)*4),(4*np.max(GCaMP_465_expcorr_motcorr))]

    
    ax1.set_xlim(tzoom) 
    #ax1.set_ylim(GCaMP_ylim);


    return GCaMP_465_expcorr_motcorr, GCaMP_465_est_motion, fig_12, fig_13
    
    
    
    



