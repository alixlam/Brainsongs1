import numpy as np
import matplotlib.pyplot as plt 

#from dataframe import df
from preproc import import_file, run_preprocessing_eog, get_clean_idx
from Useful_function import creating_epochs
from mne.time_frequency import tfr_morlet
from analysis import quantize
from math import floor, ceil
from scipy.stats import pearsonr
from Useful_function import stand 




def get_power(file = 'data/caroltrio_eeg_2019.06.20_23.57.27.edf', show = False):
  '''
  Get power according to time and frequency

  Parameters
  ----------
  file : EEG file
  
  Return 
  ------
  raw : raw cleaned of eog arteact
  power, times 
  '''



  ncycle_dict = {'data/caroltrio_eeg_2019.06.20_23.57.27.edf':50, 'data/caroltrio_eeg_2019.03.16_14.07.07.edf': 20}
  raw = import_file(file)
  raw = run_preprocessing_eog(raw, show = show)

  epochs = creating_epochs(raw, duration = raw.times[-1])
  freqs = np.arange(2,30, 4)
  n_cycles = freqs*ncycle_dict[file] # different number of cycle per frequency
  power, _ = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                          return_itc=True, decim=3, n_jobs=1)


  power.plot([4],dB = True,tmin = 60, title=power.ch_names[0])#, vmin = -50)
  if file == 'data/caroltrio_eeg_2019.06.20_23.57.27.edf':
      power.crop(tmax = 1467)
  
  else :
    power.crop(tmax = 583)
  
  times = np.linspace(0, power.times[-1], num = power.data[0].shape[1])

  return raw, power, times



def get_measures(curnpz = 'data/caroleeg_perf3.npz', power = None, res = 1,): 
  '''
  Get focus and subjective time from npz file

  Parameters
  ----------
  curnpz : npz file
  res : sampling rate of focus and subjective time
  
  Return 
  ------
  subjective time and focus
  
  '''

  assert power is not None, "you have to provide power in order to match the times"
  if curnpz == 'data/caroleeg_perf3.npz':
    tmax = 1467
  
  else : 
    tmax = 583
  
  focus = np.load(curnpz)['focus']
  subjtime = np.load(curnpz)['subjtime']
  #Quantization
  focus_q = quantize(focus,res=res,delaytime=0) 
  subt_q = quantize(subjtime,res=res,delaytime=1)
  maxtime = min(subt_q.shape[0], focus_q.shape[0])
  mintime = maxtime - ceil(power.times[-1]*res)

  focus_q[mintime:maxtime,1] += np.random.rand(maxtime-mintime)
  subt_q[mintime:maxtime,1] += np.random.rand(maxtime-mintime)

  focus_t = np.empty((tmax,2))
  focus_t[:,0] = focus_q[mintime:maxtime, 0] - mintime
  focus_t[:,1] = focus_q[mintime:maxtime,1]

  subt_t = np.empty((tmax,2))
  subt_t[:,0] = subt_q[mintime:maxtime, 0] - subt_q[mintime,0]
  subt_t[:,1] = subt_q[mintime:maxtime,1]

  return subt_t, focus_t



def plot_powers(power, times, subjt, focus):
  '''
  Plot the evolution of power in different frequency bands and the subjective time and focus 
  according to time.

  Parameters
  ----------
  powers : time frequency powers, result of function get_power 
  times : time for power
  subjt,focus : subjective time and focus result of function get_measures
  
  Return 
  ------
  Plot
  
  
  '''
  grid = plt.GridSpec(5, 3, wspace=0.4, hspace=0.3)
  plt.figure(figsize=(15,10))
  powers = [power.data.mean(axis = 0)[1], power.data.mean(axis = 0)[2],power.data.mean(axis = 0)[3]+power.data.mean(axis = 0)[4],(power.data.mean(axis = 0)[3]+power.data.mean(axis = 0)[4])/power.data.mean(axis = 0)[2]]
  titles = ['Theta', 'Alpha', 'Beta', 'Beta/Alpha']

  for i, pow in enumerate(powers):
    plt.subplot(grid[i,:3])
    plt.plot(times, 10. * np.log10(pow))
    plt.title(titles[i])

  plt.subplot(grid[4,:3])
  plt.plot(focus[:,0],focus[:,1],label="Focus")
  plt.plot(subjt[:,0],subjt[:,1],label="Subj. duration")
  plt.ylabel('Valeur')
  plt.xlabel('Temps (s)')
  plt.legend(loc=2, borderaxespad=0.)
  


def pearson_cor(power, raw = None, feedback = None, clean_eeg = True, reject = 60, show= True):
  '''
  Computes pearson correlation between power and subjective time or focus

  Parameters
  ----------
  raw : Preprocessed EEG file if you want to clean the EEG of peaks 
  power, times : Result of get_powers
  subjt, focus : subjective time or focus (result of function get_measures)
  clean_eeg : whether to clean the EEG of peaks (default =True)
  reject : threshold for Peak to peak amplitude necessary if clean eeg = True (default = 60)
  show : whether to show plots of y and power 


  Return 
  ------
  powers, title
  Plot
  
  
  '''
  powers = [power.data.mean(axis = 0)[1], power.data.mean(axis = 0)[2],power.data.mean(axis = 0)[3]+power.data.mean(axis = 0)[4],(power.data.mean(axis = 0)[3]+power.data.mean(axis = 0)[4])/power.data.mean(axis = 0)[2]]
  titles = ['Theta', 'Alpha', 'Beta', 'Beta/Alpha']
  
  time_indices = np.array(feedback[:,0]*1/(power.times[-1]/power.data.shape[-1]), dtype = int)
  
  
  if clean_eeg == True:
    assert raw is not None, "You have to provide the preprocessed EEG file"
    indices = get_clean_idx(raw_file = raw, duration = feedback.shape[0]/power.times[-1], reject = reject)
  else :
    indices = np.arange(0, power.data.shape[-1], 1)
  
  assert (feedback is not None), "You have to choose between Subjt and focus"
    

  y = stand(feedback[:,1][indices])

  plt.figure(figsize = (15,10))

  for i, title in enumerate(titles):
    plt.subplot(4,1, i+1)
    x = powers[i][time_indices][indices]
    x = 10*np.log10(x)
    x = stand(x)
    cor = pearsonr(y,x)[0]
    p_v = pearsonr(y,x)[1]
    print('Correlation for {} : {:f} (p_value : {:f}) '.format(title, cor, p_v))
    if show == True:
      plt.plot(y, label = 'subjective time')
      plt.plot(x, label = '{} (correlation {:f})'.format(title, cor))
      plt.title('Normalized Subjective time and {} power'.format(title))
      plt.legend()



