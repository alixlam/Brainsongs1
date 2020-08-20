import numpy as np
import matplotlib.pyplot as plt
import os 

from mne.io import read_raw_edf
import matplotlib.pyplot as plt 
from mne.channels import make_standard_montage
from mne.channels import read_custom_montage
from mne.preprocessing import create_eog_epochs, ICA

from mne import make_fixed_length_epochs, Epochs, Annotations
from mne import Epochs
from mne import Annotations 
import pandas as pd
from mne import make_fixed_length_events
from mne.preprocessing import create_eog_epochs
from mne.io import RawArray
from math import floor


def import_file(filepath, montage = 'standard_1020'): 
  '''
  Import edf file and returns raw mne object

  Parameters
  ----------
  filepath : path to edf file
  montage : Name of electrode montage
  
  Return 
  ------
  Raw file

  '''

  #creating the montage object
  montage = make_standard_montage(montage)

  #reading the file with an EOG channel 
  Raw = read_raw_edf(filepath, stim_channel = False, verbose=False)


  #loading the data into memory
  Raw.load_data()

  #Raw.set_channel_types({'ECG':'ecg'})

  Raw.set_montage(montage)

  import_file.filename = filepath

  return Raw


def run_preprocessing_eog(raw_eeg, tmin = 0,remove_artefact = True, show = False):
  '''
  Remove EOG artefacts

  Parameters
  ----------
  raw_eeg : mne raw file
  tmin : minimum time to consider (default = 0)
  remove_artefact : whether to remove artefacts once found  (default = True)
  show : Whether to show plots (components, signal before and after ... )

  Return 
  ------
  Clean raw file if remove artefact is true, original raw file otherwise

  '''

  raw_inter = raw_eeg.copy().crop(tmin = tmin)
  filtered_raw = raw_inter.filter(l_freq=3., h_freq=40.)
  print(filtered_raw.annotations)
  ica = ICA(n_components=raw_eeg.info.get('nchan'), random_state=9)
  
  ica.fit(filtered_raw, verbose = False)
  ica.exclude = []
  # find which ICs match the EOG pattern
  eog_indices, eog_scores = ica.find_bads_eog(raw_inter, threshold = 1.6 , ch_name = 'Fpz', verbose = False)
  eog_indices.append(3)
  #eog_indices.append(4)
  ica.exclude = eog_indices
  


  if show == True:
    ica.plot_sources(filtered_raw)
    ica.plot_components()

    # barplot of ICA component "EOG match" scores
    #ica.plot_scores(eog_scores)

    ica.plot_overlay(raw_inter)
    #ica.plot_properties(raw_inter)

  if eog_indices != []: 
    print('matching ICA component found')
    # plot diagnostics
    #ica.plot_properties(filtered_raw, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    if show == True:
      ica.plot_sources(raw_inter)

    if remove_artefact == True: 
      reconst_raw = filtered_raw.copy()
      ica.apply(reconst_raw)
      return reconst_raw.filter(3,40)
    else : 
      return filtered_raw
    
  else : 
    print('No matching ICA component')
    return filtered_raw

def get_clean_idx(raw_file, duration = 1, reject = 60):
  '''
  Get idx of bad segments of data. Devide the data into epochs of length (duration) and reject segments 
  with peak to peak amplitude that exceeds a predefined threshold. Returns the index of good epochs. 

  Parameters
  ----------
  raw_file : Raw file
  duration : length of epochs (default 1s)
  reject : Threshold to reject (default 60)
  
  Return 
  ------
  list of good index
  '''
  events = make_fixed_length_events(raw_file.copy().crop(tmax= 1467), id = 1,duration = duration)
  epochs = Epochs(raw_file.copy().crop(tmax= 1467), events, tmin = 0, tmax = duration , reject = dict(eeg = reject), baseline = None)
  epochs.drop_bad()
  return epochs.selection  



