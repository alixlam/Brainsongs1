from mne.time_frequency import psd_multitaper
import numpy as np
import matplotlib.pyplot as plt
from mne import EpochsArray
from math import floor
from sklearn.metrics import auc

def variance(list):
  return np.std(np.array(list))

def mean(list):
  return np.mean(np.array(list))

def create_epochs(dict2, lengthepoch = 2):
    '''Function to create an epoch object from the data with events correspondong to each state
       
    Parameters
    ----------
    dataset: dict2 with files sorted by states 
    lengthepoch : length of the desired epoch (default 2)

    Return
    ------
    Epochs
    '''
    states = ['lent', 'rapide', 'libre','meditate', 'baseline', 'resting1','resting2', 'val']
    lengthepoch = lengthepoch
    sfreq = dict2['libre']['raw file'][1].info['sfreq']
    data = []
    info = dict2['libre']['raw file'][1].info
    indices = np.arange(len(dict2['libre']['raw file'])*3)
    events = []
    event_id =dict(lent = 1, rapide=2, libre=3, meditate = 4, baseline = 5, resting1=6, resting2=7, val=8) 

  
    k = 0
    tmin = 0

    for  state in states:
        id = event_id[state]
        for l, epochs in enumerate(dict2[state]['raw file']):
            for i in range(floor(15/lengthepoch)):
                if epochs.times[-1] == 20:
                    test = epochs.copy().crop(5+i*lengthepoch, 5+(i+1)*lengthepoch).get_data()
                else:
                    test = epochs.copy().crop(i*lengthepoch, (i+1)*lengthepoch).get_data()  
                
                events.append([k, 1, id])
                k +=1
                data.append(test)


    events = np.array(events)
    data = np.array(data) 
    epochs = EpochsArray(data, info, events, tmin, event_id, reject = dict(eeg=2000))
    return epochs


def frequency_analysis(epochs, states = ['meditate', 'baseline','lent', 'rapide', 'libre', 'resting1', 'val'], show = True, pick = 'C4'):
    '''
    Function to return psd of each state
    
    Parameters
    ---------- 
    epochs : Epoch object 
    states: list of strings (default all)
    show : wether do plot the psd (default = True)
    pick : if show = True, the channel to plot (string) 
    
    Return 
    ------ 
    list_alphas := list of alpha powers[{'C4': [.....], 'P4' : [....]},{'C4': ....} ] (len(list_alphas) = len(states))
    list_betas :=  list of beta powers
    list_thetas := list of theta powers
    list_baratio := list of Beta/alpha ratios 
                
    '''
    states = states
    nstate = [len(epochs[state]) for state in states]
    pick = pick

    alpha_band  = 9.3


    list_alphas, list_betas, list_thetas,list_ab = [], [], [],[]
    chan_alphas, chan_betas, chan_thetas, chan_ab = {'C4' : [], 'P4' : [], 'O2' : [], 'Fp': []},{'C4' : [], 'P4' : [], 'O2' : [], 'Fp': []},{'C4' : [], 'P4' : [], 'O2' : [], 'Fp': []},{'C4' : [], 'P4' : [], 'O2' : [], 'Fp': []}


    for index,state in enumerate(states):
        print(state)
        for channel in ['C4', 'P4', 'O2', 'Fp']:
            if channel == 'Fp':
                raw = epochs[state].copy().pick_channels(['Fp2'])
            else:
                raw = epochs[state].copy().pick_channels([channel])
            
            psds, freqs = psd_multitaper(raw,low_bias = True,
                                fmin = 2, fmax=40,
                                n_jobs=1)
            psds = 10 * np.log10(psds)
            psds = psds.mean(1)
            test = psds.mean(0)
            psds_std = psds.mean(0).std(0)
            
            if channel == pick and (state not in []):
                
                plt.plot(freqs, test)
                plt.legend(states)
                plt.title('PSD of electrode '+ pick)
                plt.xlabel('frequency (Hz)')
                plt.ylabel('Power (dB)')
                plt.fill_between(freqs, test - psds_std, test+ psds_std, alpha=.1)

            
            for t,epochspsds in enumerate(psds):

                xx = freqs[((freqs > alpha_band - 2) & (freqs < alpha_band+2))]
                yy =epochspsds[(freqs > alpha_band - 2) & (freqs < alpha_band+2)]
                alphach = (auc(xx,yy))
                chan_alphas[channel].append(alphach)

                xx = freqs[((freqs < alpha_band - 2) & (freqs > alpha_band-6))]
                yy = epochspsds[((freqs < alpha_band - 2) & (freqs > alpha_band-6))]

                thetach = (auc(xx,yy))

                chan_thetas[channel].append(thetach)

                xx = freqs[((freqs > alpha_band + 2) & (freqs < 40))]
                yy = epochspsds[((freqs > alpha_band + 2) & (freqs < 40))]

                betach = (auc(xx,yy))

                chan_betas[channel].append(betach)
                chan_ab[channel].append((betach/alphach))

        list_alphas.append(chan_alphas)
        list_betas.append(chan_betas)
        list_thetas.append(chan_thetas)
        list_ab.append(chan_ab)
        chan_alphas, chan_betas, chan_thetas, chan_ab = {'C4' : [], 'P4' : [], 'O2' : [], 'Fp' : []},{'C4' : [], 'P4' : [], 'O2' : [], 'Fp' : []},{'C4' : [], 'P4' : [], 'O2' : [], 'Fp' : []},{'C4' : [], 'P4' : [], 'O2' : [], 'Fp' : []}

    print('MAR : '+str(len(list_alphas)))

    alphas = [round(mean(list_alphas[i]['C4']+ list_alphas[i]['P4']+list_alphas[i]['O2']+list_alphas[i]['Fp']),2) for i in range(len(states)-1)]
    betas = [round(mean(list_betas[i]['C4']+ list_betas[i]['P4']+list_betas[i]['O2']+list_betas[i]['Fp']),2) for i in range(len(states)-1)]
    thetas = [round(mean(list_thetas[i]['C4']+ list_thetas[i]['P4']+list_thetas[i]['O2']+list_thetas[i]['Fp']),2) for i in range(len(states)-1)]
    ab =  [round(mean(list_ab[i]['C4']+ list_ab[i]['P4']+list_ab[i]['O2']+list_ab[i]['Fp']),2) for i in range(len(states)-1)]

    plt.show()
    return list_alphas, list_betas, list_thetas, list_ab, alphas, betas, thetas, ab

