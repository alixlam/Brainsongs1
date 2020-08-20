
import numpy as np
from mne.preprocessing import create_eog_epochs
from mne.preprocessing import ICA
from mne import make_fixed_length_epochs
from mne import Epochs
from mne import Annotations 
import pandas as pd
from mne import make_fixed_length_epochs
from mne import make_fixed_length_events
from mne.preprocessing import create_eog_epochs

from plotting import import_file


'''Function to transform dataset into epochs 
Arguments : raw : raw file to transform 
            tmin : minimum time to take into account (default = 0)
            tmax: maximum time to take into account (default = 60)
            duration : duration of each epoch (default = 0.5)

Return: Epoch object (mne)
'''

def creating_epochs(raw, tmin = 0, tmax = 60, verbose = True, duration = 0.5):
  '''Function to transform dataset into epochs 

  Parameters
  ----------
  raw : raw file to transform 
  tmin : minimum time to take into account (default = 0)
  tmax: maximum time to take into account (default = 60)
  duration : duration of each epoch (default = 0.5)

  Return
  ------
  Epoch object (mne)
'''

  epochs = make_fixed_length_epochs(raw.crop(tmin = tmin, tmax = tmax),duration = duration, verbose = verbose)
  return epochs



'''Function to locate bad segments 
Arguments : raw : raw file to transform 
            tmin : minimum time to take into account (default = 0)
            tmax: maximum time to take into account (default = 60)
            threshold: peak peak amplitude threshold (default = 1500)
            duration : duration of each epoch (default = 0.5)

Returns : Raw object annotated with bad segments
          Channels with bad segments
          Duration of bad segments
'''

def calc_variance(raw, tmin = 0, tmax = 60, threshold= 1500, verbose = True, duration = 0.5):
  '''
  Function to locate bad segments 
  
  Parameters
  ----------
  raw : raw file to transform 
  tmin : minimum time to take into account (default = 0)
  tmax: maximum time to take into account (default = 60)
  threshold: peak peak amplitude threshold (default = 1500)
  duration : duration of each epoch (default = 0.5)

Return
------
Raw object annotated with bad segments
Channels with bad segments
Duration of bad segments
'''

  epochs = creating_epochs(raw, tmin = tmin, tmax = tmax, verbose = verbose, duration = duration)
  var_to_drop = []
  bad_channels = []
  durations = []
  channel_names = {'0' : 'C4', '1' : 'P4', '2' : 'O2', '3' : 'EOG'}

  inter = 0
  i = 0
  n = 0
  for epoch in epochs:
    for index in range(int(raw.info['nchan'])-1): 
      var = np.var(epoch[index])
      minmax = np.max(epoch[index]) - np.min(epoch[index])

        
      if minmax >=threshold:
          var_to_drop.append(i*duration)
          bad_channels.append(index)
    if (i+1)%(20*1/duration) ==0 and i >0:
      durations.append(len(var_to_drop)*duration-inter)
      inter += durations[n]
      n +=1


      #variances[i, index] = var
    i+= 1
  bad_channels = list(set(bad_channels))
  for index in range(len(bad_channels)): 

    bad_channels[index] = channel_names[str(bad_channels[index])]
  my_annot =  Annotations(onset = var_to_drop, duration = [duration]*len(var_to_drop), description=['bad']*len(var_to_drop))
  raw.set_annotations(my_annot)
  return raw, bad_channels, durations


'''Function to remove occular artefact by removing the ICA component(s) that are closest to a reference channel (here Fp2)
Arguments : raw_eeg : Raw file 
            tmin : default 0
            tmax : default 60
            remove_artefact : wether to remove the ICA component from raw (default True)
            show : show scores (default : true)

Return : Return cleaned raw or original raw if remove_artefact = True or no matching components were found.
'''
def run_preprocessing_eog(raw_eeg, tmin = 0, tmax = 60, remove_artefact = True, show = True):


  '''Function to remove occular artefact by removing the ICA component(s) that are closest to a reference channel (here Fp2)

  Parameters
  ----------
  raw_eeg : Raw file 
  tmin : default 0
  tmax : default 60
  remove_artefact : wether to remove the ICA component from raw (default True)
  show : show scores (default : true)

  Return
  ------
  Return cleaned raw or original raw if remove_artefact = True or no matching components were found.
  
  '''
  raw_inter = raw_eeg.copy().crop(tmin = tmin, tmax = tmax)
  filtered_raw = raw_inter.filter(l_freq=1., h_freq=40.)
  

  ica = ICA(n_components=raw_eeg.info.get('nchan')-2, random_state=9)

  print(raw_eeg.info)
  
  ica.fit(filtered_raw, verbose = False)
  ica.exclude = []
  # find which ICs match the EOG pattern
  eog_indices, eog_scores = ica.find_bads_eog(raw_inter, ch_name = 'Fp2', threshold = 1.4, verbose = False)
  ica.exclude = eog_indices
  

  if show == True:
    ica.plot_sources(filtered_raw)
    ica.plot_components()

    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)

    #ica.plot_overlay(raw_inter)
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
      return reconst_raw
    else : 
      return filtered_raw
    
  else : 
    print('No matching ICA component')
    return filtered_raw

