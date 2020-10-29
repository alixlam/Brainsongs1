from make_class import fit_HMM, plotDistribution, plotTimeSeries,get_hidden_states
from coranalysis import get_power, get_measures, plot_powers, cor
#from dataframe import df
from preproc import import_file, run_preprocessing_eog
import matplotlib.pyplot as plt
from Useful_function import stand
import numpy as np
from mne import Annotations
from mne import Epochs
from scipy import stats
from math import floor 
from mne import EpochsArray
from mne.time_frequency import psd_multitaper
from sklearn.metrics import auc
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation


def create_epochs_c(perf = 1, show_destrib = False, show_states = False, length_epoch = 3):
	files = ["data/caroltrio_eeg_2019.03.16_14.07.07.edf",'data/caroltrio_eeg_2019.06.20_23.57.27.edf']

	if perf == 1:
		file = files[0]
	else :
		file = files[1]

	states = get_hidden_states(perf = perf, show_destrib=show_destrib, show_states= show_states)
	annotations = {0 : 'fast',1 :'slow'}

	onset = [0]
	duration = []
	previous = states[0]
	desc = [annotations[states[0]]]

	count = 0

	for i in range(len(states)):
		if states[i] == previous and i != len(states)- 1:
			count += 1
		
		elif states[i] != previous and i != len(states)-1:
			onset.append(i)
			desc.append(annotations[states[i]])
			previous = states[i]
			duration.append(count)
			count = 1
		elif states[i]== previous and i == len(states)-1: 
			duration.append(count)
			print('je suis la')

		else :
			print('je suis ici')
			print(i)
			duration.append(count)



	raw = import_file(file)

	new_raw = run_preprocessing_eog(raw, show= False)

	annotation_ ={'fast' : 1, 'slow' :2} 


	events =[]
	sfreq = raw.info['sfreq']
	info = raw.info
	data = []
	k = 0
	tmin = 0

	for i in range(len(onset)-1):
		print(i)
		id = annotation_[desc[i]]
		test = new_raw.copy().crop(onset[i], onset[i+1])
		for i in range(floor((onset[i+1]-onset[i])/length_epoch)):
			dat = test.copy().crop(i*length_epoch, (i+1)*length_epoch).get_data()
			events.append([k, 1,id])
			k+=1
			data.append(dat)

	events= np.array(events)
	data = np.array(data)
	epochs = EpochsArray(data, info, events, tmin, annotation_, reject = dict(eeg = 100))

	return epochs, desc


def get_frequency_c(epochs): 

	state = ['slow', 'fast']
	list_alphas = []
	list_betas = []
	list_thetas = []
	alphas = []
	betas = []
	thetas  = []
	alpha_band  = 9.3

	for index,state in enumerate(state):

		raw = epochs[state].copy()
		psds, freqs = psd_multitaper(raw,low_bias = True,fmin = 2, fmax=40, n_jobs=1)
		psds = 10 * np.log10(psds)
		psds = psds.mean(1)
		test = psds.mean(0)
		psds_std = psds.mean(0).std(0)
		plt.plot(freqs, test)

				
		for t,epochspsds in enumerate(psds):

			xx = freqs[((freqs > alpha_band - 2) & (freqs < alpha_band+2))]
			yy =epochspsds[(freqs > alpha_band - 2) & (freqs < alpha_band+2)]
			alphach = (auc(xx,yy))

			xx = freqs[((freqs < alpha_band - 2) & (freqs > alpha_band-6))]
			yy = epochspsds[((freqs < alpha_band - 2) & (freqs > alpha_band-6))]

			thetach = (auc(xx,yy))

			xx = freqs[((freqs > alpha_band + 2) & (freqs < 25))]
			yy = epochspsds[((freqs > alpha_band + 2) & (freqs < 25))]

			betach = (auc(xx,yy))
			print(betach)
			#plt.plot(xx,yy)
		

			list_alphas.append(alphach)
			list_betas.append(betach)
			list_thetas.append(thetach)
		alphas.append(list_alphas)
		betas.append(list_betas)
		thetas.append(list_thetas)
		list_alphas = []
		list_betas = []
		list_thetas = []
		
	plt.show()

	
	return alphas, betas, thetas


def barplots(alphas, betas, thetas): 
	sns.set(style = 'white', font_scale = 2)
	p_values = []
	list = [thetas, alphas, betas]

	for l,frequency in enumerate(['theta','alpha', 'beta']):
		p_values.append(stats.ttest_ind(list[l][0],list[l][1], equal_var = False)[1])

	p_values= np.array(p_values)*len(p_values)

	dataframe = pd.DataFrame()
	dataframe['States'] = ['slow']*len(alphas[0])+['fast']*len(alphas[1])
	dataframe['Alpha'] = alphas[0]+alphas[1]
	dataframe['Theta'] = thetas[0] + thetas[1]
	dataframe['Beta'] = betas[0] + betas[1]


	fig, ax = plt.subplots(1,3, figsize = (17,20), squeeze = False)
	place = {0: (0,0), 1: (0,1), 2: (0,2)}

	for l,frequency in enumerate(['Theta','Alpha', 'Beta']):
		sns.barplot(ax = ax[place[l]], y= frequency, x = 'States', data = dataframe, capsize = 0.1 ).set(xlabel = '', ylabel = frequency)
		add_stat_annotation(ax[place[l]], y= frequency, x = 'States', data = dataframe, box_pairs= [('slow', 'fast')], perform_stat_test=False, pvalues = [p_values[l]] , text_format='star', loc='outside', verbose=2, comparisons_correction = None, line_offset = 0.02, text_offset = 0.01)
	
	fig.subplots_adjust(wspace = 0.38, hspace = 0.62)
	fig.show()
	return p_values



'''epochs, desc = create_epochs_c(perf = 1, show_destrib = False, show_states = False, length_epoch = 3)
alphas, betas, thetas = get_frequency_c(epochs)
p_values1 = barplots(alphas,betas,thetas)'''


epochs3, desc3 = create_epochs_c(perf = 3, show_destrib = True, show_states = True, length_epoch = 3)
alphas3, betas3, thetas3 = get_frequency_c(epochs3)
p_values3 = barplots(alphas3,betas3,thetas3)

y = np.array(['slow']*len(alphas3[0])+['fast']*len(alphas3[1]))

X = np.array([alphas3[0]+alphas3[1], betas3[0]+betas3[1], thetas3[0]+thetas3[1]]).T

import numpy as np 
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold,permutation_test_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Normalizer


classifier = SVC()
#classifier = MLPClassifier()
#classifier =  DecisionTreeClassifier()
#classifier = KNeighborsClassifier()

resampler = SMOTE(random_state= 22)


estimator = Pipeline([
  #('Normalizer', Normalizer()),
  #('resample', resampler),
  #('feature_selection', SelectKBest(f_classif, k = 5)),
  ('classification', classifier)
])


kfoldcv = StratifiedKFold(n_splits=4)
mean_acc = cross_val_score(estimator=estimator,X=X,y=y,cv=kfoldcv)
#permutation_test_score(estimator=estimator,X=X,y=y,cv=kfoldcv)'''

results = cross_validate(estimator = estimator, X=X, y=y, cv=kfoldcv, return_estimator=True)
best_estimator = results['estimator'][-1]


#X_test = X = np.array([alphas[0]+alphas[1], betas[0]+betas[1], thetas[0]+thetas[1]]).T
#prediction = best_estimator.predict(X_test)