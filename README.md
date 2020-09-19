# Brainsong1

This code was written for the Brain Songs project during an internship at IMT Atlantique supervized by Nicolas Farrugia and Giulia Lioi. 


## Block analysis 

![alt text](https://github.com/alixlam/Brainsongs1/blob/master/block_data/experimental_protocol.png)

### Method for preprocessing and analysis :
Preprocessing:
The preprocessing was done using the MNE-python toolbox by Gramfort et al. [1]
First, the signals were bandpassed filtered with FIR (Finite Impulse Response) keeping the frequency between 1.0 and 40 Hz.
To reduce the contamination of the EEG signal by eye movement artefacts, IC’s components were estimated on the continuous data. 
Using a prefrontal electrode as an Electro-Oculography channel, we ran an autodetection algorithm to find the IC’s components that best match the ‘EOG’ channel. The ICA components that correlate strongly with the EOG signal were then removed and the EEG signal was recomposed with the remaining IC’s components.
Finally, the new signal was segmented into consecutive epochs of 2 seconds, epochs where the signal’s amplitude of one or more channels exceeded a predefined threshold were removed in order to keep 90% of data. 

Data analysis:
Individual Alpha Frequency (IAF) [2] was determined by finding the individual dominant EEG frequency in the baseline signal. Based on that frequency we defined the Alpha, Theta and beta bands respectively (7.5-12.5, 4.5-7.5, 12.5-25).
To conduct our analysis, we estimated the power spectral density using multitapers and computed the powers in the different frequency bands. The 2s epochs were labelled with their corresponding state (free, low, and fast) and we ran ANOVA. 

## Continuous data

### Method for preprocessing : 
The preprocessing was the same as previously

Data analysis:
We conducted a time-frequency analysis using Morlet Wavelet to compute the correlation between the subjective time feedback and the evolution of power of different frequency bands (alpha, beta and theta) in time.
Then we tried the Spoc algorithm [3] to see if we could predict the variable subjective time with the EEG signal. 

## References
[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X, [DOI]


[2] Klimesch W. EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. Brain Res Brain Res Rev. 1999;29(2-3):169-195. doi:10.1016/s0165-0173(98)00056-3

[3] Dahne, S., et al (2014). SPoC: a novel framework for relating the amplitude of neuronal oscillations to behaviorally relevant parameters. NeuroImage, 86, 111-122.

