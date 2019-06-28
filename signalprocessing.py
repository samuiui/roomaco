import numpy as np
import scipy
from scipy.signal import sosfilt, butter, sosfiltfilt

def order2fc(order, radix=10):
	"""
	determine center frequency from order
	for band pass filter

	Parameters
	----------------------------------------------------------------------------
	order : int
	    order to decide center frequency
	radix : int
	    can use only 2 or 10. 10 is recommended by JIS & ISO.

	Returns
	----------------------------------------------------------------------------
	fc : float
	    center frequency [Hz]
	"""
	if radix == 10:
	    fc = (radix ** (order/10))*1000
	elif radix == 2:
	    fc = (radix ** (order/3))*1000
	else :
	    raise Exception('use 2 or 10 for exp=.')
	return fc

def fc2cutoff(fc, oct_width = 3):
	"""
	calculate cutoff frequencies from center frequency and octave band width
	for generate band pass filter

	Parameters
	----------------------------------------------------------------------------
	fc: float(or int)
	    center frequency [Hz]
	oct_width: int, default 3
	    1/oct_width octave band width
	    
	Returns
	----------------------------------------------------------------------------
	fl: float
	    lower cutoff frequency [Hz]
	fh: float
	    higher cutoff frequency [Hz]
	"""    
	fl = fc / (2 ** (1 / 2 * oct_width))
	fh = fc * (2 ** (1 / 2 * oct_width))
	return fl, fh

def preferred_frequency_list(oct_width=3):
	"""
	provide preffered frequencies

	Parameters
	----------------------------------------------------------------------------
	oct_width: int, default 3
	    1/oct_width octave
	    now we can use only 1 or 3
	    
	Returns
	----------------------------------------------------------------------------
	preferred_freqencies: array like
	    preferred frequencies defined in ISO 266
	"""

	if oct_width == 3:
	    preferred_frequencies = np.array([0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5,
	    								6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40,
	    								50, 63, 80, 100, 125, 160, 200, 250,
	    								315, 400, 500, 630, 800, 1000, 1250,
	    								1600, 2000, 2500, 3150, 4000, 5000,
	    								6300, 8000, 10000, 12500, 16000, 20000])
	    return preferred_frequencies
	elif oct_width == 1:
	    preferred_frequencies = np.array([1, 2, 4, 8, 16, 31.5, 63, 125, 250,
	    								500, 1000, 2000, 4000, 8000, 16000])
	    return preferred_frequencies
	else:
	    raise Exception('use only 1 or 3.')

def nearest_preferred_frequency(frequency, oct_width=3):
	"""
	pick up the nearest nominal frequency

	Parameters
	----------------------------------------------------------------------------
	frequency: int(or float)
	    target frequency [Hz]
	oct_width: int, default 3
	    1/oct_width octave
	    use only 1 or 3
	    
	Returns
	----------------------------------------------------------------------------
	preferred_frequency: int(or float)
	    the nearest nominal frequency from input frequency
	"""
	frequency_list = preferred_frequency_list(oct_width)
	nearest_frequency_index = np.abs(frequency_list - frequency).argmin()
	preferred_frequency = frequency_list[nearest_frequency_index]
	return preferred_frequency

def generate_bpf_coef(fc, fs = 44100, oct_width = 3, order = 8):
	"""
	generate band pass filter

	Parameters
	----------------------------------------------------------------------------
	fc: float
	    center frequency [Hz]
	fs: int, default 44100
	    sampling frequency [Hz]
	oct_width: int, default 3
	    1/oct_width octave band width
	order: int
		filter's order
	    
	Return
	----------------------------------------------------------------------------
	bpf_coef: array like
	    second-order filter coefficients for sosfilt
	    np.shape(bpf_coef) == (n_sections, 6)
	"""
	nyquist = 0.5*fs
	low = fc2cutoff(fc, oct_width)[0] / nyquist
	high = fc2cutoff(fc, oct_width)[1] / nyquist
	bpf_coef = butter(order/2, [low, high], btype='bandpass', output='sos')
	return bpf_coef

def band_pass_filter(signal, fc, fs = 44100, oct_width = 3, order = 8):
	"""
	filter signal band pass filter

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		target signal
	fc: float
	    center frequency [Hz]
	fs: int, default 44100
	    sampling frequency [Hz]
	oct_width: int, default 3
	    1/oct_width octave band width
	order: int
		filter's order
	    
	Return
	----------------------------------------------------------------------------
	signal_filtered: array like
		filtered signal
	"""
	"""signal_filtered = sosfiltfilt(generate_bpf_coef(fc, fs, oct_width, order)
								, signal, padtype='even')
	"""
	signal_filtered = sosfilt(generate_bpf_coef(fc, fs, oct_width, order)
								, signal)
	return signal_filtered

def convolution(signal1, signal2):
	"""
	convolve two signals

	Parameters
	--------------------------------------------------------------------------------------------
	signal1, signal2: array like
	    target signals of convolution 
	    
	Return
	--------------------------------------------------------------------------------------------
	convolved_signal: array like
	    convolved signal        
	"""
	# adjust length of signal
	if len(signal1) > len(signal2):
	    zero_padding = np.zeros(len(signal1)-len(signal2))
	    signal2 = np.hstack((signal2, zero_padding))
	elif len(signal1) < len(signal2):
	    zero_padding = np.zeros(len(signal2)-len(signal1))
	    signal1 = np.hstack((signal1, zero_padding))

	signal1_fft = fftpack.fft(signal1)
	signal2_fft = fftpack.fft(signal2)
	convolved_signal_fft = signal1_fft*signal2_fft
	convolved_signal_complex = fftpack.ifft(convolved_signal_fft)
	convolved_signal = np.real(convolved_signal_complex)
	return convolved_signal