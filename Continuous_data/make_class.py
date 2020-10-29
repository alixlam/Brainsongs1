from coranalysis import get_power, get_measures
from hmmlearn.hmm import GMMHMM, GaussianHMM
from scipy import stats as ss
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Useful_function import stand 
from sklearn.preprocessing import normalize



def fit_HMM(feedback, n_components = 2, hmm = 'GaussianHMM'):
	assert hmm == 'GaussianHMM' or hmm =='GMMHMM', "You have to choose between GaussianHMM or GMMHMM"

	y = feedback[:,1]

	if hmm == 'GaussianHMM':
		model = GaussianHMM(n_components =n_components)
	else : 
		model = GMMHMM(n_components= n_components)
	
	model.fit(y.reshape(len(y),1))

	states = model.predict(y.reshape(len(y),1))
	mus = np.array(model.means_)
	sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
	P = np.array(model.transmat_)

	return y,states, mus, sigmas, P


 
def plotDistribution(Q, mus, sigmas, P):

    # calculate stationary distribution
	eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
	one_eigval = np.argmin(np.abs(eigenvals-1))
	pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])

	x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
	fx_0 = pi[0]*ss.norm.pdf(x_0,mus[0],sigmas[0])

	x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
	fx_1 = pi[1]*ss.norm.pdf(x_1,mus[1],sigmas[1])


	x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
	fx = pi[0]*ss.norm.pdf(x,mus[0],sigmas[0]) + pi[1]*ss.norm.pdf(x,mus[1],sigmas[1])

	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(Q, color='k', alpha=0.5, density=True, bins = 50)
	l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='Fast')
	l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='Slow')
	l3, = ax.plot(x, fx, c='k', linewidth=2, label='Combined State Distn')

	fig.subplots_adjust(bottom=0.15)
	handles, labels = plt.gca().get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
	fig.show()

	return None

def plotTimeSeries(Q, hidden_states, ylabel, perf = 3):
	if perf == 1:
		ma = [1,0]
	else:
		ma = [0,1]
	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot(111)
 
	xs = np.arange(len(Q))
	masks = hidden_states == ma[0]
	ax.scatter(xs[masks], Q[masks], c='r', label='Fast')
	masks = hidden_states == ma[1]
	ax.scatter(xs[masks], Q[masks], c='b', label='Slow')
	ax.plot(xs, Q, c='k')
		
	ax.set_xlabel('Time (s)')
	ax.set_ylabel(ylabel)
	fig.subplots_adjust(bottom=0.2)
	handles, labels = plt.gca().get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
	fig.show()

	return None

def get_hidden_states(perf = 1, show_destrib = False, show_states = False):
	files = ["data/caroltrio_eeg_2019.03.16_14.07.07.edf", 'data/caroltrio_eeg_2019.06.20_23.57.27.edf',"data/caroleeg_perf1.npz","data/caroleeg_perf3.npz"]
	
	if perf == 1: 
		rawfile = files[0]
		curnpz = files[2]
	else:
		rawfile = files[1]
		curnpz = files[3]

	_, power, _ = get_power(file = rawfile,show = False)
	subjt, _ = get_measures(curnpz = curnpz, power = power)

	y,states, mus, sigmas, P = fit_HMM(subjt)

	if show_destrib == True:
		plotDistribution(y.reshape(len(y), 1), mus, sigmas, P)
	
	if show_states == True:
		plotTimeSeries(y.reshape(len(y),1), states, 'subj time', perf = perf)

	return states
