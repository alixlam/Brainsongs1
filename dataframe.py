import numpy as np
import matplotlib.pyplot as plt 

import os 
import pandas as pd 

from preproc import import_file



def create_data_frame(path = 'data'):


  fichiers = os.listdir(path)
  fichiers.remove('caroleeg_perf1.npz')
  fichiers.remove('caroleeg_perf3.npz')
  fichiers.remove('.DS_Store')


  print(fichiers)
  d = {'filenames' : fichiers}
  

  df = pd.DataFrame(data = d)
  df = df.sort_values('filenames')
  electrodes = []
  liste_date = []
  liste_channels = []
  duration = []
  pourcentage_keep = []
  segments = []
  time = []
  length = 0.5


  i = 0
  for filename in df['filenames']:
    raw = import_file(path+'/'+filename)
    liste_date.append(raw.info['meas_date'])
    #raw,channels, n = calc_variance(raw, verbose = False, duration = length)[0:3]
    #segments.append(len(raw.annotations))
    #liste_channels.append(channels)
    #print(n)
    #duration.append(n)
    print(raw.times[-1])
    #pourcentage = round((len(raw.annotations)*length/raw.times[-1])*100,2)
    electrodes.append(raw.info['ch_names'])
    time.append(raw.times[-1])
    #pourcentage_keep.append([str(pourcentage)+ '%'])
    i+=1

  df['Date'] = liste_date
  df['Electrodes names'] = electrodes
  #df["'corrupted' channels"] = liste_channels 
  #df['Lost pourcentage'] = pourcentage_keep
  #df['Segments length'] = segments
  df['file length'] = time
  
  return df

df= create_data_frame()
#df.to_csv('ficherEEG.csv', header = True)
print(df)