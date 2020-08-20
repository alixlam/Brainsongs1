import numpy as np
import matplotlib.pyplot as plt 
from mne import make_fixed_length_epochs


def stand(x):
  '''
  Standardize array
  '''
  return (x-np.mean(x))/np.std(x)


def plot_eeg2(raw_eeg, channels = ['Fpz'], tmin = 0, tmax = 60, color = 'blue'): 
    '''
    Plot EEG 

    Parameters
    ----------
    raw_eeg : Raw file
    channels : Channels to plot (default Fpz)
    tmin, tmax: minimum and max time to plot (default 0 and 60)

    Return 
    ------
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
        plt.plot(x,y, color = color)
        title = 'EEG for electrode {}'.format(name_channel)
        plt.title(title)
        plt.xlabel('times')
        plt.ylabel('amplitude')
    
    plt.suptitle(raw_eeg.info['meas_date'], size = 'xx-large')



def creating_epochs(raw, tmin = 0, verbose = True, duration = 0.5):
    '''
    Plot EEG 

    Parameters
    ----------
    raw : Raw file
    tmin: minimum time to consider
    duration : length of epochs

    Return 
    ------
    Epochs 

    '''
    epochs = make_fixed_length_epochs(raw.crop(tmin = tmin),duration = duration, verbose = verbose)
    return epochs

