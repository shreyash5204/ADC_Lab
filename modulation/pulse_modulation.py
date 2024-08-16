import numpy as np
import matplotlib.pyplot as plt
from .util import *
from scipy import signal

def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)


def PPM(inputs):

    def generate_ppm(input_signal, ppm_ratio, pulse_frequency, time_duration):
        t = np.linspace(0, time_duration, len(input_signal), endpoint=False)
    
        normalized_signal = (input_signal - np.min(input_signal)) / (np.max(input_signal) - np.min(input_signal))
        pulse_positions = np.floor(ppm_ratio * normalized_signal * len(t)).astype(int)
    
        ppm_waveform = np.zeros(len(t))
        ppm_waveform[pulse_positions] = 1
    
        return t, ppm_waveform

    [fm, Am, message_type, fs,ppm_ratio] = inputs
    fm = round_to_nearest_multiple(fm)
    x = np.linspace(-1/500, 1/500, 1000000)

    sampling_rate = 1000000
    duration = 1
    duty_cycle_range = (20, 80)
    position_range = (0.1, 0.9)

    if message_type == "sin":
        message = Am * np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am * np.cos(2 * np.pi * fm * x)
    elif message_type == 'tri':
        message = triangular(fm, Am, x)

    pulse = 1+signal.square(2 * np.pi * fs * x)
    time_duration = len(message) / fs

    t, ppm_waveform = generate_ppm(message, ppm_ratio, fs, time_duration)

    

    a = plot_graph(x, message, title="Message", condition="plot", color="red")
    b = plot_graph(t, ppm_waveform, title="PPM Signal", condition="plot", color="green")

    return [a, b]
    

def PCM(inputs):
    sampling_rate = 1000
    [fm,Am,message_type,ql,nb] = inputs
    fm = round_to_nearest_multiple(fm)

    duration = 1
    x = np.linspace(-1/500, 1/500, 1000)

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)

    num_quantization_levels = ql  # Number of quantization levels
    amplitude_range = (-Am, Am)

    # Generate a continuous message signal
    t = np.linspace(0, 1, sampling_rate)
    message_signal = np.sin(2 * np.pi * fm * x)  # Example message signal

    # Calculate the step size between quantization levels
    step_size = (amplitude_range[1] - amplitude_range[0]) / (num_quantization_levels - 1)


    # Quantize the analog signal
    quantized_signal = np.round((message - amplitude_range[0]) / step_size) * step_size + amplitude_range[0]
    quantized_value = ''

    for value in quantized_signal:
        quantized_value += "," + str((value - amplitude_range[0]) / step_size)

    # Encoding: Convert quantized values to binary
    encoded_signal = np.array([format(int((value - amplitude_range[0]) / step_size), '0{0}b'.format(3)) for value in quantized_signal])
    encoded_str = ''.join(encoded_signal)

    def generate_pulse_from_encoded(encoded_str, pulse_width, sampling_rate, start_index, num_bits):
        pulse_signal = np.zeros(int(num_bits * sampling_rate * pulse_width))
        encoded_bit=''
        for i in range(num_bits):
            bit = encoded_str[start_index + i]
            encoded_bit = encoded_bit + bit
            if bit == '1':
                pulse_signal[i * int(sampling_rate * pulse_width):(i + 1) * int(sampling_rate * pulse_width)] = 1
            
        return encoded_bit,pulse_signal


    pulse_width = 0.01  # Pulse width in seconds
    start_index = 0  # Start index in the encoded string
    num_bits = nb # Number of bits to plot

    encoded_bit,pulse_signal = generate_pulse_from_encoded(encoded_str, pulse_width, sampling_rate, start_index, num_bits)


    a = plot_graph(x, message,color="red", title="Message signal")
    b = plot_graph(t, quantized_signal,color="green", title="Quantized wave")
    c = plot_graph(np.linspace(0, num_bits, len(pulse_signal)), pulse_signal,color="pink", title="pulse")
    d = encoded_bit
    #e = quantized_value

    return [a,b,c,d] 


def PWM(inputs):
    [fm,Am,message_type,fs] = inputs
    fm = round_to_nearest_multiple(fm)
    #N  = 1000
    x = np.linspace(-1/500, 1/500, 1000000)

    fm = round_to_nearest_multiple(fm)
    fs = round_to_nearest_multiple(fs)

    sampling_rate = 1000000  # Number of samples per second
    duration = 1  # Duration of the signal in seconds
    duty_cycle_range = (20, 80)

 
   

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)
    elif message_type =='tri':
        message = triangular(fm, Am, x)

    # Generate time values
    t = np.linspace(-10, 10, int(sampling_rate * duration))
    normalized_message = (message - message.min()) / (message.max() - message.min())

    # Generate duty cycle based on the normalized message signal
    duty_cycle = np.interp(normalized_message, (0, 1), duty_cycle_range) / 100.0

    # Generate the PWM signal
    pwm_signal = np.where(np.mod(t, 1/fs) < duty_cycle / fs, 1, 0)
    modulated_wave = message * pwm_signal

    a = plot_graph(x, message, title="Message signal",condition="plot",color="red")
    b = plot_graph(t, pwm_signal, title="PWM Signal",condition="plot",color="green")
    c = plot_graph(x, modulated_wave, title="Modulated wave",condition="plot",color="blue")
    #d = plot_graph(x, demodulated_wave, title="Demodulated wave",condition="plot",color="blue")

    return [a,b,c]


def PAM(inputs):
    # [Am,Ac,fm,fc,message_type,fs] = inputs
    [fm, Am, message_type, fs] = inputs
    # N  = 1000
    x = np.linspace(-1/500, 1/500, 1000000)

    fm = round_to_nearest_multiple(fm)
    fs = round_to_nearest_multiple(fs)

    pulse_width = 0.01  # Pulse width in seconds
    pulse_period = 0.1  # Pulse period in seconds
    duration = 1.0  # Duration of the signal in seconds
    sampling_rate = 1000000  # Sampling rate in Hz

    if fs <= 40000:
        t = np.linspace(-10, 10, 1000000)
    else:
        t = np.linspace(-1, 1, 1000000)

    pulse = 1 + signal.square(2 * np.pi * fs * t)

    if message_type == "sin":
        message = Am * np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am * np.cos(2 * np.pi * fm * x)
    elif message_signal == "tri":
        message = triangular(fm, Am, x_message)

    modulated_wave = message * pulse
    b, a = signal.butter(10, 0.8, "low")
    demodulated_wave = signal.filtfilt(b, a,modulated_wave)

    a = plot_graph(x, message, title="Message", condition="plot", color="red")
    b = plot_graph(x, pulse, title="Pulse", condition="plot", color="green")
    c = plot_graph(
        x,
        modulated_wave,
        title="Natural Pulse Modulated wave",
        condition="plot",
        color="blue",
    )
    # d = plot_graph(
    #     x,
    #     demodulated_wave ,
    #     title="Demodulated Wave (Envelope Detection)",
    #     condition="plot",
    #     color="purple",
    # )

    return [a, b, c]

def circular_shift(arr, angle):
    length = len(arr)
    shift = int((angle % (2 * np.pi)) / (2 * np.pi) * length)
    return np.roll(arr, shift)

def QUANTIZATION(inputs):
    [fm,Am,message_type,ql] = inputs
    #N  = 1000
    x = np.linspace(-1/500, 1/500, 1000000)
    fm = round_to_nearest_multiple(fm)


    amplitude_range = (-Am, Am)  # Range of the amplitude values
    num_quantization_levels = ql  # Number of quantization levels
    x_message = create_domain_AM()

    # Generate a continuous message signal
    message_signal = np.sin(2 * np.pi * fm * x)  # Example message signal

    # Calculate the step size between quantization levels
    step_size = (amplitude_range[1] - amplitude_range[0]) / (num_quantization_levels - 1)


    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x_message)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x_message)
    # elif message_type== "tri":
    #     message = triangular(fm, Am, x_message)    

    # Quantize the message signal
    quantized_wave = np.round((message - amplitude_range[0]) / step_size) * step_size + amplitude_range[0]
    #Demodulation by cumulative summation with manual initial condition
    reconstructed_message = np.cumsum(quantized_wave-np.mean(quantized_wave))
    shift_angle = -fm

    # Perform circular shift based on angle
    reconstructed_message = circular_shift(reconstructed_message, shift_angle)
    # Normalize the reconstructed message to the original amplitude
    reconstructed_message = Am * (reconstructed_message / np.max(np.abs(reconstructed_message)))
    phase_shift = np.pi / 2  # Shift by pi/4 radians

    # Apply phase shift to the reconstructed signal (assuming it's a sinusoidal signal)



    a = plot_graph(x, message, title="Message",condition="plot",color="red")
    b = plot_graph(x, quantized_wave, title="Quantized wave",condition="plot",color="blue")
    c = plot_graph(x, reconstructed_message_shifted, title="Demodulated Wave",condition="plot",color="blue")

    return [a,b]

def SAMPLING(inputs):
    [fm,Am,message_type,fs] = inputs
    x = np.linspace(-1/500, 1/500, 1000000)
    if fs>=4600 and fs<=5400:
        fs=5500
    fm = round_to_nearest_multiple(fm)
    fs = round_to_nearest_multiple(fs)


    if fs <= 40000:
        t = np.linspace(-10, 10, 1000000)
    else:
        t = np.linspace(-1, 1, 1000000)

    pulse = 1+signal.square(2 * np.pi * fs * t)

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)
    elif message_type=='tri':
        message = triangular(fm, Am, x)

    modulated_wave = message * pulse


    a = plot_graph(x, message, title="Message",condition="plot",color="red")
    b = plot_graph(x, pulse, title="Pulse",condition="plot",color="green")
    c = plot_graph(x, modulated_wave, title="Modulated wave",condition="plot",color="blue")


    return [a,b,c]