import pandas as pd 
import numpy as np
import os
from plotting import import_file
from Preprocess import calc_variance
from plotting import plot_eeg2
from Preprocess import run_preprocessing_eog
from sklearn.metrics import auc
from math import sqrt



def create_data_frame(path = 'Data'):
  ''' 
  function to create a dataframe of the files and informations about quality of different files

  Parameters
  ----------
  path : path to data 

  Return 
  ------
  df pandas dataframe

  '''

  fichiers = os.listdir(path)
  fichiers.remove('.DS_Store')
  #fichiers.remove('.ipynb_checkpoints') 
  #fichiers.remove('sample_data') 
  d = {'filenames' : fichiers}
  

  df = pd.DataFrame(data = d)
  df = df.sort_values('filenames')

  liste_events = ['resting1', 'baseline1', 'meditate1', 'training1', 'bloc1', 'bloc2', 'bloc3', 'bloc4', 'bloc5','resting2', 'baseline2', 'meditate2', 'training2', 'bloc6', 'bloc7', 'bloc8', 'bloc9', 'bloc10', 'training3', 'training4', 'bloc11', 'bloc12', 'bloc13', 'bloc14', 'bloc15', 'bloc16']
  detailed_events = np.array([['resting1']*3,['baseline']*3,['meditate']*3,['lent', 'rapide', 'libre'], ['libre', 'rapide', 'lent'], ['rapide', 'lent', 'lent'], ['libre', 'libre', 'rapide'], ['lent', 'rapide', 'lent'], ['libre', 'rapide', 'libre'], ['resting2']*3, ['baseline']*3, ['meditate']*3, ['lent', 'rapide', 'libre'], ['rapide', 'lent','lent'], ['libre', 'rapide', 'lent'], ['libre', 'rapide', 'libre'], ['lent', 'rapide', 'lent'], ['libre', 'libre', 'rapide'], [],[],['Val']*3,['Val']*3,['Val']*3,['Val']*3,['Val']*3,['Val']*3])
  liste_date = []
  liste_channels = []
  duration = []
  pourcentage_keep = []
  pourcentage = []
  s_lent =0 
  s_rapide = 0
  s_libre = 0

  i = 0
  for filename in df['filenames']:
    raw = import_file(path+ '/'+filename)
    liste_date.append(raw.info['meas_date'])
    channels, n = calc_variance(raw, verbose = False)[1:]
    liste_channels.append(channels)
    duration.append(n)
    pourcentage = [round((time/ 20)*100, 1) for time in n]

    
    if len(detailed_events[i]) >2:
      pourcentage_keep.append([detailed_events[i][j]+' : '+str(pourcentage[j]) + '%' for j in range(len(pourcentage))])
      for j in range(len(n)):
        if detailed_events[i][j] == 'lent':
          s_lent += n[j]
        elif detailed_events[i][j] == 'rapide':
          s_rapide += n[j]
        else :
          s_libre += n[j]

    else: 
      pourcentage_keep.append([str(sum(pourcentage))+ '%'])
    i+=1

  df['Date'] = liste_date
  df['session'] = liste_events
  df['Description'] = detailed_events
  df['channels'] = liste_channels 
  df['Lost pourcentage'] = pourcentage_keep

  event_lent = 0
  event_rapide = 0 
  event_libre = 0
  for events in detailed_events :
    event_lent += events.count('lent')
    event_rapide += events.count('rapide')
    event_libre += events.count('libre')
  
  liste = ['Lent : '  +str(round((event_lent*20)/s_lent, 1))+ '%', 'Rapide : '  +str(round((event_rapide*20)/s_rapide, 1))+ '%','Libre : '  +str(round((event_libre*20)/s_libre, 1))+ '%']
  
  
  return df, liste


#df.to_csv('ficherEEG.csv', header = True)


def data_preparation(filepath, show = False):
  ''' 
  Function to prepare data for the dataset, ie run_preprocessing and save the raw file and the array containing the time serie

  Parameters
  ----------
  filepath : path to data 
  Show : wether to plot each new time serie after preprocessing 

  Return 
  ------
  the new_raw (raw object mne), a vector containing the time serie (careful it contains the possible artefact due to electrode (jumps))
  the raw files divided in blocs 
  '''

  from mne.io import RawArray
  raw = import_file('Data/'+filepath)
  raw.drop_channels(['ECG'])
  annot_raw, _, _ = calc_variance(raw)

  if show == True:  
      plot_eeg2(raw, channels = ['C4', 'P4', 'O2', 'EOG'], tmin = 0, tmax = 60)

  new_raw = run_preprocessing_eog(annot_raw, show =show)


  data_array = np.empty((3,4,5001))
  print(new_raw.info)
  for i in range(3): 
      inter = new_raw.copy()
      data_array[i] = inter.crop(tmin = i*20, tmax = 20 + i*20).get_data()
      
  raw_files = [new_raw.copy().crop(0, 20), new_raw.copy().crop(25, 40),new_raw.copy().crop(45, 60)]


  return new_raw, data_array, raw_files


'''
creating the dataframe 
'''
df, liste = create_data_frame()


'''
creating the first dataset :
Putting data in a dictionary for easy access :
eg : dataset['bloc5'] returns {'time serie' : np.array([....]), 
                               'Labels' : ['lent', 'rapide', 'libre'],
                               'raw files : 3 raw files 
                               'full raw' : 1 full raw}
'''
dataset = {}
for file in df['filenames']: 
  new_raw, X, raws = data_preparation(file, show = False)
  sfreq = new_raw.info['sfreq']
  labels = df[df['filenames'] == file]['Description'].values[0]
  session = str(df[df['filenames'] == file]['session'].values[0])
  dataset[session] = {'time serie': X, 'Labels' : labels, 'raw files': raws, 'full raw':new_raw}


'''Creating second dataset with different states grouped together
eg dict2['lent'] returns {'raw file': all raw files of state 'lent', 'index' : [0,1, ...]}
with the list of index corresponding to the date of the recording (0: 17th of june, 1: 19th of june)
'''

raw_lent = []
raw_rapide =[]
raw_libre = []
raw_meditate =[]
raw_baseline =[]
raw_resting = []
raw_resting2 = []
raw_val = []



list_index_lent = []
list_index_rapide = []
list_index_libre = []
list_index_meditate = []
list_index_baseline = []
list_index_resting = []


for k in ['resting1', 'resting2','meditate1', 'meditate2','baseline1', 'baseline2','training1', 'training2', 'bloc1', 'bloc2', 'bloc3', 'bloc4', 'bloc5', 'bloc6', 'bloc7', 'bloc8', 'bloc9', 'bloc10', 'bloc11', 'bloc12', 'bloc13', 'bloc14', 'bloc15','bloc16']:
  if k in ['resting1', 'meditate1','baseline1','training1', 'bloc1', 'bloc2', 'bloc3', 'bloc4', 'bloc5']:
    date_index = 0
  else :
    date_index = 1

  for i in range(3):
    if dataset[k]['Labels'][i] == 'lent':
      raw_lent.append(dataset[k]['raw files'][i])
      list_index_lent.append(date_index)
    elif dataset[k]['Labels'][i] == 'rapide':
      raw_rapide.append(dataset[k]['raw files'][i])
      list_index_rapide.append(date_index)
    elif dataset[k]['Labels'][i] == 'meditate':
      raw_meditate.append(dataset[k]['raw files'][i])
      list_index_meditate.append(date_index)
    elif dataset[k]['Labels'][i] == 'baseline':
      raw_baseline.append(dataset[k]['raw files'][i])
      list_index_baseline.append(date_index)
    elif dataset[k]['Labels'][i] == 'libre' :
      raw_libre.append(dataset[k]['raw files'][i])
      list_index_libre.append(date_index)
    elif dataset[k]['Labels'][i] == 'resting1':
      raw_resting.append(dataset[k]['raw files'][i])
      list_index_resting.append(date_index)
    
    elif dataset[k]['Labels'][i] == 'resting2': 
      print('here')
      raw_resting2.append(dataset[k]['raw files'][i])

    elif dataset[k]['Labels'][i] == 'Val':
      print('here') 
      raw_val.append(dataset[k]['raw files'][i])
    
    else:
      pass
      
dict2 ={} 
dict2['lent'] = { 'raw file': raw_lent, 'index': list_index_lent}
dict2['rapide'] = { 'raw file': raw_rapide, 'index': list_index_rapide}
dict2['libre'] = { 'raw file': raw_libre, 'index' : list_index_libre}
dict2['meditate']= { 'raw file': raw_meditate, 'index': list_index_meditate}
dict2['baseline']= { 'raw file': raw_baseline, 'index': list_index_baseline}
dict2['resting1']= { 'raw file': raw_resting, 'index' : list_index_resting}
dict2['resting2']= {'raw file': raw_resting2}
dict2['val'] = {'raw file': raw_val}

