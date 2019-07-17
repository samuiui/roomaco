# from http://aidiary.hatenablog.com/entry/20120225/1330179868
from scipy.signal import lfilter
import numpy as np
from scipy.fftpack.realtransforms import dct

def pre_emphasis(signal, p=0.97):
	"""
	pre-emphasis filtering

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		target signal 
	p: float
		pre-emphasis coefficient

	Returns
	----------------------------------------------------------------------------
	output: array like
		filtered signal
	"""
	output = lfilter([1.0, -p], 1, signal)
	return output

def amplitude_spectrum(signal, fs, nfft=2048):
	"""
	calculate amplitude spectrum 

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		target signal
	nfft: int, default 2048
		fft sample

	Returns
	----------------------------------------------------------------------------
	spec: array like
		amplitude spectrum of input signal
	freqlist: array like
		frequency scale
	"""
	hamming = np.hamming(len(signal))
	signalwin = signal*hamming
	spec = np.abs(np.fft.fft(signal, nfft))[:int(nfft/2)]
	freqlist = np.fft.fftfreq(nfft, d=1.0/fs)[:int(nfft/2)]
	return spec, freqlist

def hz2mel(f):
	"""
	transform Hz to mel

	Parameters
	----------------------------------------------------------------------------
	f: float
		frequency [Hz]

	Returns
	----------------------------------------------------------------------------
	output: float
		frequency on mel scale
	"""
	output = 1127.01048 * np.log(f / 700.0 + 1.0)
	return output

def mel2hz(m):
	"""
	transform mel to Hz

	Parameters
	----------------------------------------------------------------------------
	f: float
		frequency on mel scale
	
	Returns
	----------------------------------------------------------------------------
	output: float
		frequency [Hz]
	"""
	output = 700.0 * (np.exp(m / 1127.01048) - 1.0)
	return output

def melFilterBank(fs, nfft, numChannels):
	"""
	generate mel filter bank

	Parameters
	----------------------------------------------------------------------------
	fs: int
		sampling frequency [Hz]
	nfft: int
		fft sample
	numChannels: int
		number of filter
	
	Returns
	----------------------------------------------------------------------------
	filterbank: array like

	fcenters: array like
	"""
	nyquisthz = fs/2
	nyquistmel = hz2mel(nyquisthz)
	nmax = int(nfft/2)
	df = fs/nfft
	dmel = nyquistmel/(numChannels+1)
	melcenters = np.arange(1, numChannels+1)*dmel
	fcenters = mel2hz(melcenters)
	indexcenter = np.round(fcenters / df)
	indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
	indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

	filterbank = np.zeros((numChannels, nmax))
	for c in np.arange(0, numChannels):
		increment = 1.0/(indexcenter[c] - indexstart[c])
		for i in np.arange(indexstart[c], indexcenter[c]):
			filterbank[int(c), int(i)] = (i - indexstart[c])*increment
		decrement = 1.0 / (indexstop[c] - indexcenter[c])
		for i in np.arange(indexcenter[c], indexstop[c]):
			filterbank[int(c), int(i)] = 1.0 - ((i - indexcenter[c])*decrement)
	return filterbank, fcenters

def mfcc(signal, fs, numChannels, nceps=12, p=0.97, nfft=2048):
	"""
	calculate mfcc from signal

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		target signal
	nceps: int
		dimention of mfcc

	Returns
	----------------------------------------------------------------------------
	output: array like
		mfcc
	"""
	signal = pre_emphasis(signal, p)
	spec = amplitude_spectrum(signal, fs)
	melfil = melFilterBank(fs, nfft, numChannels)[0]
	melspec = np.log10(np.dot(spec[0], melfil.T))
	ceps = dct(melspec, type=2, norm="ortho", axis=-1)
	output = ceps[:nceps]
	return output