import numpy as np
import matplotlib.pyplot as plt
from plotting import import_file, plot_eeg2
from create_dataset import df, dict2, liste
import seaborn as sns
from epochs_analysis import variance, mean
import numpy as np
import matplotlib.pyplot as plt 
from epochs_analysis import create_epochs, frequency_analysis
from create_dataset import dict2
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statannot import add_stat_annotation
from scipy import stats




list_alphas, list_betas, list_thetas, list_ab= frequency_analysis(create_epochs(dict2), pick = 'all')[0:4]


import pandas as pd
dataframe = pd.DataFrame()

dataframe['State'] = ['Slow']*(len(list_alphas[2])*len(list_alphas[2]['C4']))+['Fast']*len(list_alphas[3])*(len(list_alphas[3]['P4']))+['Free']*(len(list_alphas[4]))*(len(list_alphas[4]['O2']))
dataframe['alpha'] = list_alphas[2]['C4'] + list_alphas[2]['P4'] + list_alphas[2]['O2']+ list_alphas[2]['Fp'] + list_alphas[2]['all'] +list_alphas[3]['C4'] + list_alphas[3]['P4'] + list_alphas[3]['O2']+ list_alphas[3]['Fp']+ list_alphas[3]['all']+ list_alphas[4]['C4'] + list_alphas[4]['P4'] + list_alphas[4]['O2'] + list_alphas[4]['Fp'] + list_alphas[4]['all']
dataframe['beta'] = list_betas[2]['C4'] + list_betas[2]['P4'] + list_betas[2]['O2'] + list_betas[2]['Fp']+list_betas[2]['all'] +list_betas[3]['C4'] + list_betas[3]['P4'] + list_betas[3]['O2']+ list_betas[3]['Fp']+ list_betas[3]['all']+ list_betas[4]['C4'] + list_betas[4]['P4'] + list_betas[4]['O2'] + list_betas[4]['Fp']+list_betas[4]['all']
dataframe['theta'] = list_thetas[2]['C4'] + list_thetas[2]['P4'] + list_thetas[2]['O2']+ list_thetas[2]['Fp']+list_thetas[2]['all'] + list_thetas[3]['C4'] + list_thetas[3]['P4'] + list_thetas[3]['O2']+ list_thetas[3]['Fp']+list_thetas[3]['all'] + list_thetas[4]['C4'] + list_thetas[4]['P4'] + list_thetas[4]['O2'] + list_thetas[4]['Fp']+list_thetas[4]['all']
#dataframe['baratio'] = list_ab[2]['C4'] + list_ab[2]['P4'] + list_ab[2]['O2']+ list_ab[2]['Fp']+list_ab[2]['all'] + list_ab[3]['C4'] + list_ab[3]['P4'] + list_ab[3]['O2']+ list_ab[3]['Fp'] +list_ab[3]['all']+ list_ab[4]['C4'] + list_ab[4]['P4'] + list_ab[4]['O2'] + list_ab[4]['Fp']+list_ab[4]['all']

electrodes = []
for i in range(2,5):
  electrodes += (['C4']*len(list_alphas[i]['C4']) + ['P4']*len(list_alphas[i]['P4']) + ['O2']*len(list_alphas[i]['O2'])  + ['Fp']*len(list_alphas[i]['Fp'])+['all']*len(list_alphas[i]['all']))

dataframe['electrode'] = electrodes

fig, ax = plt.subplots(2,2, figsize = (17,20), squeeze = False)
place = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}

test = 'beta'+'~ State'
sns.set(font_scale=2, style= 'white')  # crazy big

list = [list_alphas, list_betas, list_thetas]
p_values = []

for l,frequency in enumerate(['alpha', 'beta', 'theta']):
  for channel in ['all']:
    p_values.append(stats.ttest_ind(list[l][2][channel],list[l][3][channel], equal_var= False)[1])
    p_values.append(stats.ttest_ind(list[l][2][channel],list[l][4][channel], equal_var= False)[1])
    p_values.append(stats.ttest_ind(list[l][3][channel],list[l][4][channel], equal_var= False)[1])

df = pd.DataFrame()
df['frequency'] = ['alpha']*3 + ['beta']*3 + ['theta']*3
df['channel'] = ['all']*9
df['test'] = ['lent/rapide', 'lent/libre', 'rapide/libre']*3
df['pvalues'] = p_values

print(df)


'''for i, channel in enumerate(['Fp', 'C4', 'P4', 'O2']):
    mod = ols(test, data=dataframe[(dataframe['electrode'] == channel)]).fit()             
    pair_t = mod.t_test_pairwise('State')
    print(pair_t.result_frame)
    sns.barplot(ax = ax[place[i]],y= 'beta', x = 'State', data = dataframe[dataframe["electrode"] == channel], capsize = 0.1 ).set(xlabel = '', ylabel = '')
    ax[place[i]].set_title(channel, loc = 'right')
    add_stat_annotation(ax[place[i]], data=dataframe[dataframe["electrode"] == channel], x='State', y='beta', box_pairs= [("Libre", "Lent"), ("Rapide", "Lent"), ("Rapide", "Libre")], perform_stat_test=False, pvalues = pair_t.result_frame['pvalue-hs'] , text_format='star', loc='outside', verbose=2, comparisons_correction = None)
fig.subplots_adjust(
        left = 0.2,  # the left side of the subplots of the figure
        right = 1 ,  # the right side of the subplots of the figure
        bottom = 0.3,  # the bottom of the subplots of the figure
        top = 1 ,    # the top of the subplots of the figure
        wspace = 0.3,
        hspace = 0.5 )
'''


fig, ax = plt.subplots(1,3, squeeze = False)
place = {0: (0,0), 1: (0,1), 2: (0,2)}

p_values = np.array(p_values)*len(p_values)
let = ['T', 'A','B']

for i, frequency in enumerate(['theta','alpha', 'beta']):
  sns.barplot(ax = ax[place[i]], y = frequency, x = 'State', data = dataframe[dataframe['electrode'] == 'all'],  capsize = 0.1 ).set(xlabel = '', ylabel = let[i]+frequency[1:])
  #ax[place[i]].set_title(let[i]+frequency[1:],fontsize = 'small', loc = 'right')

  if frequency == 'beta':
    box_pairs = [('Slow', 'Fast'), ('Fast', 'Free')]
    pvalues = [p_values[3], p_values[5]]
    add_stat_annotation(ax[place[i]], data=dataframe[dataframe["electrode"] == 'all'], x='State', y=frequency, box_pairs= box_pairs, perform_stat_test=False, pvalues = pvalues , text_format='star', loc='outside', verbose=2, comparisons_correction = None, line_offset = 0.02, text_offset = 0.01)

  elif frequency == 'theta': 
    box_pairs = [('Fast', 'Free')]
    pvalues = [p_values[8]]
    add_stat_annotation(ax[place[i]], data=dataframe[dataframe["electrode"] == 'all'], x='State', y=frequency, box_pairs= box_pairs, perform_stat_test=False, pvalues = pvalues , text_format='star', loc='outside', verbose=2, comparisons_correction = None)

fig.subplots_adjust(wspace = 0.5, hspace = 2) # the left side of the subplots of the figure
fig.show()
