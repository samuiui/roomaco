import numpy as np
from scipy.fftpack import fft, ifft
from roomaco.roomparameters import detect_arrival_sample

def logss(A = 0.8, N_exp=16):
	"""
	generate Log-Swept Sine signal

	Parameters
	----------------------------------------------------------------------------
	A: float, default 0.8
		amplifier (must be 0 < A <= 1)
	N_exp: int
		length [sample] = 2 ** N_exp

	Return
	----------------------------------------------------------------------------
	output: array like
		Log-Swept Sine signal
	"""
	N = 2 ** N_exp
	J = N // 2
	k = np.arange(1, N/2, 1)
	shift = (N-J) // 2
	# design logss in freqency domain
	logss_freq = np.zeros(N+1, dtype=np.complex)
	logss_freq[2:int(N/2 + 1)] = (1/np.sqrt(k)*
		np.exp(-1j*J*np.pi/((N/2)*np.log(N/2))*k*np.log(k)))
	half_logss_freq = logss_freq[2:int(N/2 + 1)]
	logss_freq[int(N/2 + 2):int(N + 1)] = half_logss_freq[::-1].conj()
	logss_freq[1] = 1
	logss_freq = logss_freq[1:]
	# adjust amplitude
	Q = A * np.sqrt(J/2) / np.sqrt(np.sum(np.abs(logss_freq) ** 2) / N)
	logss_freq = Q * logss_freq
	
	logss = np.real(ifft(logss_freq, n = N))
	output = np.roll(logss, shift)
	return output

def inverse_logss_freq(N_exp=16):
	"""
	generate inverce log-swept sine frequency

	Parameters
	----------------------------------------------------------------------------
	N_exp: int, default 16
		signal length[sample] is 2**N_exp

	Return
	----------------------------------------------------------------------------
	output: array like (complex)
		inverse log-ss signal in frequency domain
	"""
	N = 2**N_exp
	J = N//2
	k = np.arange(1, N/2, 1)

	invlogss_freq = np.zeros(N+1, dtype=np.complex)
	invlogss_freq[2:int(N/2 + 1)] = np.sqrt(k)*np.exp(1j*J
		*np.pi/((N/2)*np.log(N/2))*k*np.log(k))
	half_invlogss_freq = invlogss_freq[2:int(N/2 + 1)]
	invlogss_freq[int(N/2+2):int(N+1)] = half_invlogss_freq[::-1].conj()
	invlogss_freq[1] = 1
	invlogss_freq = invlogss_freq[1:]
	return invlogss_freq

def ir_from_logss(signal, N_exp=16):
	"""
	calculate impulse response from measured signal using Log-SS

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		measured signal using Log-SS
		Note: preprocessed signal is recommended for input
	N_exp: int, default 16
		length of one cycle of measurement Log-SS signal = 2 ** N_exp

	Returns
	----------------------------------------------------------------------------
	ir: array like
		impulse response
	"""
	N = 2**N_exp
	signal_fft = fft(signal)
	ir_freq = signal_fft * inverse_logss_freq(N_exp)
	ir_freq_ifft = ifft(ir_freq)
	ir = np.real(ir_freq_ifft)
	return ir

def cross_spectral(N, input_signal, output_signal, start_sample=0):
	"""
	calclate impulse response from two signal(x, y)
	by using cross spectral technique

	Parameters
	----------------------------------------------------------------------------
	N: int
		length of analysis target
	input: array like
		input signal of the system
	output: array like
		output signal of the system
	start_sample: int
		determine start sample of analysis

	Returns
	----------------------------------------------------------------------------
	h: array like
		estimated impulse response of the target system
		imuples response = h[:int(len(h)/4)]
	"""
	average_times = int(((len(output_signal)-N)/(N/2))+1)
	loop_times = min(average_times, 
	                min(int((len(output_signal))/(N/2))-1, 
	                    int((len(input_signal))/(N/2))-1))
	zero_pad = np.zeros(N)
	for i in range(loop_times):
		x = input_signal[int(N/2*i + 1):int(N/2*i + N)]
		x = np.concatenate([x, zero_pad])
		y = output_signal[int(N/2*i + 1):int(N/2*i + N)]
		y = np.concatenate([y, zero_pad])

		X = fft(x)
		Y = fft(y)

		if i == 0:
			cross_spectral = np.conj(X)*Y
			power_spectral = np.conj(X)*X
		else :
			cross_spectral = cross_spectral + np.conj(X)*Y
			power_spectral = power_spectral + np.conj(X)*X
	H = cross_spectral/power_spectral
	h = np.real(ifft(H))
	return h

def sync_add(signal, N, times, start_sample=0):
	"""
	synchronous addition

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		target signal
	N: int
		one cycle length
	times: int
		number of synchronous addition
	start_sample: int, default 0
		start point of synchronous addition

	Returns
	----------------------------------------------------------------------------
	output: array like
		synchronous added signal
		Note: len(output)==N
	"""
	target_signal = signal[start_sample:start_sample+times*N+times]
	added_signal = np.zeros(N)
	len(added_signal)
	for i in range(times):
		added_signal += target_signal[i*N+i:(i+1)*N+i]
	output = added_signal/times
	return output
