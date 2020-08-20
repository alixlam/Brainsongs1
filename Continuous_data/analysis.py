import numpy as np 

import librosa

import os 

from tqdm import tqdm

def quantize(vector,res=0.3,endtime=2,delaytime=1):
    vectimes = vector[:,0]
    maxtime = max(vectimes)
    
    alltimes = np.arange(0,maxtime-endtime,res)
    qvec = []
    ## loop over all time segments between two instants of length "res"
    
    prev_val = vector[0,1]
    
    for i,curtime in enumerate(alltimes[:-2]):
        
        ind = np.argwhere((vectimes>curtime) & (vectimes<alltimes[i+1]))
        
        if len(ind)==0:
            qvec.append([curtime,prev_val])
        else:
            qvec.append([curtime,np.mean(vector[ind,1])])
            prev_val = np.mean(vector[ind,1])
    return np.stack(qvec) + np.repeat(np.array([[delaytime],[0]]),len(qvec),axis=1).T


def write_clustered_segments(inputfile,labels_pred,seglength,analysis_name,resultpath,max_n_examples=None,
                             shuffle_examples=True,concat_summary=True,save_indiv=False):

    for i in range(labels_pred.max()+1): # Loop through all cluster labels
        
        if save_indiv:
            # Create a directory for the current label
            os.makedirs(os.path.join(resultpath,'%s/clusters_%d' % (analysis_name,i)),exist_ok=True)

        # fetch all labels with the current value 

        ind_curlabel = np.argwhere(labels_pred==i)
        if len(ind_curlabel) == 0:
            continue

        # Calculate the offsets in the original wave file 
        offsets_wave = ind_curlabel * seglength
        
        if shuffle_examples:
            offsets_wave = np.random.permutation(offsets_wave)

        if max_n_examples is None: 
            curmax_nexamples = len(offsets_wave)
        else:
            curmax_nexamples = max_n_examples

        print("Writing %d examples for Cluster %d" % (curmax_nexamples,i))
        # loop on all examples (maximum of n_examples)
        
        if concat_summary:
            bigwave = []
        
        
        for curoffset in tqdm(range(min(len(offsets_wave),curmax_nexamples))):

            string_offset = '%0.2f_sec_' % offsets_wave[curoffset]

            # cut the original wave file and save the excerpt
            curwave,cursr = librosa.load(path=inputfile, sr = None, mono = False, offset=offsets_wave[curoffset],duration=seglength)
            
            if len(curwave) == 0:
                continue
            
            debug = False
            if debug:
                print(curwave.shape)
                print(cursr)
                print(os.path.join(resultpath,'%s/clusters_%d/%s.wav' % (analysis_name,i,string_offset)))
            
            if save_indiv:
                librosa.output.write_wav(path=os.path.join(resultpath,'%s/clusters_%d/%s.wav' % (analysis_name,i,string_offset)),y=curwave,sr=cursr)
            if concat_summary:
                bigwave.append(curwave)
    
        if concat_summary:
            bigwave = np.hstack(bigwave)
            librosa.output.write_wav(path=os.path.join(resultpath,'%s/summary_cluster_%d.wav' % (analysis_name,i)),y=bigwave,sr=cursr)