import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.channels import read_custom_montage
import os 



def import_file(filepath, montage = 'standard_1020'): 

  '''function to import raw data :  returns  one Raw file with 3 electrodes + EOG electrode + ECG electrode

  Parameters
  ----------
  filepath 
  montage : electrode montage (default : standard_1020)

  Return
  ------
  Raw file (mne)
  '''
  #creating the montage object
  montage = make_standard_montage(montage)

  #reading the file with an EOG channel 
  Raw = read_raw_edf(filepath, stim_channel = False, verbose=False)

  Raw.rename_channels({'EOG': 'Fp2'})


  #loading the data into memory
  Raw.load_data()


  Raw.set_channel_types({'ECG':'ecg'})

  Raw.set_montage(montage)

  import_file.filename = filepath
  print(Raw.info)

  return Raw



def plot_eeg2(raw_eeg, channels = ['C4'], tmin = 0, tmax = 60): 
  ''' 
  Function to visualize the unprocessed data 

  Paramters
  ---------
  Raw_eeg : Raw file
  Channels : List for the channels to plot (default : ['C4'])
  tmin : minimum time to plot (default 0)
  tmax : maximum time to plot (default 60)

  Returns
  -------
  NA
  '''

  plt.figure(figsize = (30,20))
  samp_freq = int(raw_eeg.info['sfreq'])
  time_indices = np.arange(tmin*samp_freq, tmax*samp_freq)
  n_channel = len(channels)

  for name_channel in channels:

    eeg_array = raw_eeg.copy().pick_channels([name_channel]).get_data()
    
    x = raw_eeg.times[time_indices]
    y = eeg_array[0, time_indices]

    plt.subplot(n_channel, 1,channels.index(name_channel)+1)
    plt.plot(x,y)
    title = 'EEG for electrode'+name_channel
    plt.title(title)
    plt.xlabel('times')
    plt.ylabel('amplitude')
  plt.suptitle(raw_eeg.info['meas_date'], size = 'xx-large')



