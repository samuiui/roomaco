import numpy as np
from roomaco.signalprocessing import band_pass_filter
from math import log10
import sys

def detect_arrival_sample(ir, rate = 20):
	"""
	detect direct sound arrival time with sample

	Parameters
    ----------------------------------------------------------------------------
	ir: array like
        impulse response
    rate: int, default 20
        threshold [dB]

    Returns
    ----------------------------------------------------------------------------
    arrival_sample: int
        detected direct sound arrival time[sample]
        note calc "arrival_sample*sampling frequency" to obtain arrival time[s]
	"""
	ir_power = ir ** 2.0
	ir_power_max = max(ir_power)
	i = 0
	arrival_sample = 0
	power_rate = ir_power[i] / ir_power_max
	for i in range(len(ir)):
		if ir_power[i] != 0.0:
			arrival_sample = i
			power_rate = ir_power[i] / ir_power_max
			if 10.0*log10(power_rate) > -1.0*rate:
				break
		else:
			continue
	return arrival_sample

def clarity(ir, fc, bound_ms = 50, fs = 44100, rate=20):
	"""
	calculate clarity of impulse response

	Parameters
	----------------------------------------------------------------------------
	ir: array like
		target impulse respose
	fc: float
		center frequency of target octave band
	bound_ms: int, default 50
		boundary time[ms] for comparison
		(e.g. bound_ms = 50 means C_50)
	frequency: int, default 44100
		sampling frequency
	rate: int, default 20
		threshold rate to detect direct sound arrival

	Returns
	----------------------------------------------------------------------------
	clarity: float
		clarity value[dB] of a target impulse response
	"""
	# split signal
	bound_sample = int(bound_ms*0.001*fs)
	sound_arrival_sample = detect_arrival_sample(ir, rate=rate)
	bound = bound_sample + sound_arrival_sample
	ir_first = ir[sound_arrival_sample:bound]
	ir_latter = ir[bound:]
	# zero padding
	analysis_length = len(ir_first)+len(ir_latter)
	first_pad = np.zeros(int(analysis_length - len(ir_first)))
	np.array(ir_first)
	ir_first_pad = np.hstack((ir_first, first_pad))
	latter_pad = np.zeros(int(analysis_length - len(ir_latter)))
	np.array(ir_latter)
	ir_latter_pad = np.hstack((latter_pad, ir_latter))
	# band pass filter
	ir_first_bpf = band_pass_filter(ir_first_pad,
		fc, fs=fs, oct_width=1, order=8)
	ir_latter_bpf = band_pass_filter(ir_latter_pad,
		fc, fs=fs, oct_width=1, order=8)
	# to power
	ir_first_bpf_power = ir_first_bpf**2
	ir_latter_bpf_power = ir_latter_bpf**2
	# calculate Clarity 
	c_mole = np.sum(ir_first_bpf_power)
	c_deno = np.sum(ir_latter_bpf_power)
	clarity = 10 * np.log10(c_mole/c_deno)
	return clarity

def definition(ir, fc, bound_ms = 50, fs = 44100, rate=20):
	"""
	calculate definition of impulse response

	Parameters
	----------------------------------------------------------------------------
	ir: array like
		target impulse respose
	fc: float
		center frequency of target octave band
	bound_ms: int, default 50
		boundary time[ms] for comparison
		(e.g. bound_ms = 50 means C_50)
	frequency: int, default 44100
		sampling frequency
	rate: int, default 20
		threshold rate to detect direct sound arrival

	Returns
	----------------------------------------------------------------------------
	definition: float
		definition value[dB] of a target impulse response
	"""
	# split signal
	bound_sample = int(bound_ms*0.001*fs)
	ir_bpf = band_pass_filter(ir, fc, fs=fs, oct_width=1)
	sound_arrival_sample = detect_arrival_sample(ir_bpf, rate=rate)
	bound = bound_sample + sound_arrival_sample
	ir_first = ir[sound_arrival_sample:bound]
	ir_all = ir[sound_arrival_sample:]
	# zero padding
	first_pad = np.zeros(int(len(ir_all)-len(ir_first)))
	np.array(ir_first)
	ir_first_pad = np.hstack((ir_first, first_pad))
	# band pass filter
	ir_first_bpf = band_pass_filter(ir_first_pad,
		fc, fs=fs, oct_width=1, order=8)
	ir_all_bpf = band_pass_filter(ir_all,
		fc, fs=fs, oct_width=1, order=8)
	# to power
	ir_first_bpf_power = ir_first_bpf**2
	ir_all_bpf_power = ir_all_bpf**2
	# calculate Clarity 
	d_mole = np.sum(ir_first_bpf_power)
	d_deno = np.sum(ir_all_bpf_power)
	definition = 10 * np.log10(d_mole/d_deno)
	return definition

def calc_decaycurve(ir):
	"""
	calcurate schroeder decay curve from impulse response

	Parameters
	----------------------------------------------------------------------------
	ir: array like
	    target impulse response

	Returns
	----------------------------------------------------------------------------
	decay_curve: array like
	    schroeder decay curve
	"""
	ir_power = ir**2.0
	ir_power_sum = np.sum(ir_power)
	temp = ir_power_sum
	curve=[]
	for i in range(len(ir)):
		temp = temp - ir_power[i]
		curve.append(temp)
	curve_dB = 10 * np.log10(curve)
	curve_offset = max(curve_dB)
	decay_curve = curve_dB - curve_offset
	return decay_curve

def calc_earlydecay_sec(ir, fs = 44100):
	"""
	calcurate early decay time from impulse response

	Parameters
	----------------------------------------------------------------------------
	ir: array like
	    target impulse response
	fs: int
	    sampling frequency [Hz]

	Returns
	----------------------------------------------------------------------------
	earlydecay_sec: float
	    early decay time [s]
	"""
	decay = calc_decaycurve(ir)
	i = 0
	while decay[i] > -0.1:
	    start = i
	    i += 1
	while decay[i] > -10.1:
	    end = i
	    i += 1
	# fit 
	target = decay[start:end]
	x = np.linspace(start, end, end - start)
	a, b = np.polyfit(x, target, 1)
	earlydecay_time = (-10/a) * 6 / fs
	return earlydecay_time

def calc_rt_sec(ir, decay_level_dB = 60, fs = 44100):
	"""
	calcrate reverberation time from impulse response

	Parameters
	----------------------------------------------------------------------------
	ir: array like
	    target impulse response
	decay_level_dB: int
	    calc time to decay to this level
	    if decay_level_dB = 10, rt_sec equals early decay time.
	fs: int
	    sampling frequency [Hz]

	Returns
	----------------------------------------------------------------------------
	rt_sec: float
	    reverberation time [s]
	"""
	decay = calc_decaycurve(ir)
	start_sample = detect_arrival_sample(ir)
	times = 60 / decay_level_dB

	while decay[start_sample] > -5.0:
		start_sample +=1

	i = start_sample
	while decay[i] > -1*decay_level_dB-5.0:
	    end_sample = i
	    i += 1
	target = decay[start_sample:end_sample]
	x = np.linspace(start_sample, end_sample, end_sample - start_sample)
	a, b = np.polyfit(x, target, 1)
	rt_sec = (-1*decay_level_dB/a)*times / fs
	return rt_sec

def center_time(ir, fs=44100):
	"""
	calculate centre time Ts(Kurer, 1969) of impulse response

	Parameters
	----------------------------------------------------------------------------
	ir: array like
		target impulse response
	fs: int
		sampling frequency

	Returns
	----------------------------------------------------------------------------
	ts: float
		center time[sec]		
	"""
	ir_power = ir**2
	ts_mole = 0
	for t in range(len(ir_power)):
		ts_mole += t*ir_power[t]
	ts_deno = np.sum(ir_power)
	ts = ts_mole/ts_deno/fs
	return ts

def dB_curve(singal):
	"""
	convert scale of input singal to dB scale

	Parameters
	----------------------------------------------------------------------------
	signal: array like
		target signal

	Returns
	----------------------------------------------------------------------------
	output: array like
		signal in dB scale
	"""
	signal_square = signal ** 2
	output = 10*log10(signal_square/max(signal_square))
	return output