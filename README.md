# Brainsong1

This code was written for the Brain Songs project during an internship at IMT Atlantique supervized by Nicolas Farrugia and Giulia Lioi. 


## Block analysis 

![alt text](https://github.com/alixlam/Brainsongs1/blob/master/block_data/experimental_protocol.png)

### Method for preprocessing and analysis :
Preprocessing:
First, the signals were low pass filtered with a cutoff frequency of 40 hertz. 
To reduce the contamination of the EEG signal by eye movement artefacts, IC’s components were estimated on the high-passed filtered (1.0 Hz) continuous data. 
Using a prefrontal electrode as an Electro-Oculography channel, we ran an autodetection algorithm to find the IC’s components that best match the ‘EOG’ channel. The ICA components that correlate strongly with the EOG signal were then removed and the EEG signal was recomposed with the remaining IC’s components.
Finally, the new signal was segmented into consecutive epochs of 1 seconds, epochs where the signal’s amplitude of one or more channels exceeded a predefined threshold were removed. 

Data analysis:
Individual Alpha Frequency (IAF) [1] was determined by finding the individual dominant EEG frequency in the baseline signal. Based on that frequency we defined the Alpha, Theta and beta bands respectively (7.5-12.5, 4.5-7.5, 12.5-30).
To conduct our analysis, we estimated the power spectral density using multitapers and computed the powers in the different frequency bands. The 1s epochs were labelled with their corresponding state (free, low, and fast) and we ran ANOVA. 

## Continuous data

### Method for preprocessing : 
The preprocessing was the same as previously

Data analysis:
We conducted a time-frequency analysis using Morlet Wavelet to compute the correlation between the subjective time feedback and the evolution of power of different frequency bands (alpha, beta and theta) in time.
Then we tried the Spoc algorithm [2] to see if we could predict the variable subjective time with the EEG signal. 

## Reference

[1] Klimesch W. EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. Brain Res Brain Res Rev. 1999;29(2-3):169-195. doi:10.1016/s0165-0173(98)00056-3

[2] Dahne, S., et al (2014). SPoC: a novel framework for relating the amplitude of neuronal oscillations to behaviorally relevant parameters. NeuroImage, 86, 111-122.

