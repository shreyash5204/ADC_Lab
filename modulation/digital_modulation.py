# ------- BASK - Binary Amplitude Shift Keying ---------
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from .util import *

def countNoOfDigits(f):
    count = 0
    while int(f) != 0: 
        count = count+1
        f=f/10
    return count    

    
def countSpace(noOfDigits):
    space = 0
    while noOfDigits != 1: 
        space = space * 10 + 1
        noOfDigits = noOfDigits-1  
    return space   

def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)


def BASK(Tb, fc,Ac1,Ac2, inputBinarySeq):

    fc = round_to_nearest_multiple(fc)
    condition = 'line'
    m = inputBinarySeq.reshape(-1, 1)
    N = len(m) # length of binary sequence
    
    x_carrier = create_domain_AM()

    t = np.arange(0, N * Tb, Tb / 100) 
    A = np.sqrt(2 / Tb) 
    t1 = 0
    t2 = Tb

    bit = np.array([])
    
    for n in range(N):
        if m[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate((bit, se)) 
   
    fDigits = countNoOfDigits(fc)
    space = countSpace(fDigits) * 9    

    t2 = np.arange(Tb / 99, Tb + Tb / 99, Tb / space)

    message = np.array([])
    for i in range(N):
        if m[i] == 1:
            y = Ac1 * np.cos(2 * np.pi * fc * t2)
        else:
            y = Ac2 * np.cos(2 * np.pi * fc * t2)
        message = np.concatenate((message, y))


    #plotting message signal

    plt.subplot(3, 1, 1)
    plt.plot(t, bit, "b", linewidth=2.5)
    plt.grid(True)
    plt.axis([0, Tb * N, -1, 2])
    plt.ylabel("Amplitude (V)")
    plt.xlabel("Time (ms)")
    plt.title("Message signal")
    plt.grid(True)

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msg = data.getvalue().hex()
    plt.figure()


    c1 = Ac1 * np.cos(2 * np.pi * fc * x_carrier)
    c2 = Ac2 * np.cos(2 * np.pi * fc * x_carrier)

    carrier1 = plot_graph(condition = condition, x = x_carrier, y = c1, title = "Carrier Signal 1",color='g')
    carrier2 = plot_graph(condition = condition, x = x_carrier, y = c2, title = "Carrier Signal 2",color='g')

    t3 = np.arange(Tb / 99, Tb * N + Tb / 99,Tb / space)
    plt.subplot(3, 1, 2)
    if Ac1 > Ac2 :
        plt.axis([0, Tb * N, -Ac1 - 5, Ac1 + 5])
    else:  
        plt.axis([0, Tb * N, -Ac2 - 5, Ac2 + 5])  
    plt.plot(t3, message, "r")
    plt.grid(True)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (V)")
    plt.title("Modulated Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    mod = data.getvalue().hex()
    plt.figure()

    plt.close('all')  # Close all plots

    # return [msg_mod, carrier]
    return [msg,carrier1,carrier2,mod]



# ------- BFSK - Binary Frequency Shift Keying ----------


def BFSK(Tb,Ac, fc1, fc2, inputBinarySeq):
    # Binary Information
    x = inputBinarySeq.reshape(-1, 1)
  
    bp = Tb #but period
    condition = "line"
    fc1 = round_to_nearest_multiple(fc1)
    fc2 = round_to_nearest_multiple(fc2)
    bit = np.array([])
    
    for n in range(len(x)):
        if x[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate((bit, se))

    t1 = np.arange(0, len(x) * bp, bp / 100)

 

    # Binary-FSK modulation
    # A = np.sqrt(2 / Tb)  # Amplitude of carrier signal
    br = 1 / bp  # bit rate
    # f1 = br * fc1  # carrier frequency for information as 1
    # f2 = br * fc2  # carrier frequency for information as 0

    if fc2>fc1:
        fDigits = countNoOfDigits(fc2)
    else:
        fDigits = countNoOfDigits(fc1)    

    space = countSpace(fDigits) * 9    

    t2 = np.arange(bp / 99, bp + bp / 99, bp / space)
    m = np.array([])
    for i in range(len(x)):
        if x[i] == 1:
            y = Ac * np.cos(2 * np.pi * fc1 * t2)
        else:
            y = Ac * np.cos(2 * np.pi * fc2 * t2)
        m = np.concatenate((m, y))


    #ploting message signal

    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, "b", linewidth=2.5)
    plt.grid(True)
    plt.axis([0, bp * len(x), -1, 2])
    plt.ylabel("Amplitude (V)")
    plt.xlabel("Time (ms)")
    plt.title("Message signal")
    plt.grid(True)

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msg = data.getvalue().hex()
    plt.figure()



    x_carrier = create_domain_AM()
    c1 = Ac * np.cos(2 * np.pi * fc1 * x_carrier)
    c2 = Ac * np.cos(2 * np.pi * fc2 * x_carrier)

    carrier1 = plot_graph(condition = condition, x = x_carrier, y = c1, title = "Carrier Signal 1",color='g')
    carrier2 = plot_graph(condition = condition, x = x_carrier, y = c2, title = "Carrier Signal 2",color='g')

    # Modulated Signal
    t3 = np.arange(bp / 99, bp * len(x) + bp / 99, bp / space)
    plt.subplot(3, 1, 2)
    plt.axis([0, bp * len(x), -Ac - 5, Ac + 5])
    plt.plot(t3, m, "r")
    plt.grid(True)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (V)")
    plt.title("Modulated Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    mod = data.getvalue().hex()
    plt.figure()

    return [msg, carrier1, carrier2, mod]


# ------------- BPSK - Binary Phase Shift Keying ---------
def BPSK(Tb,Ac, fc, inputBinarySeq):
    # x = np.array([1, 0, 0, 1, 1, 0, 1])  # Binary Information
    x = inputBinarySeq.reshape(-1, 1)
    condition = 'line'
    # bp = 0.000001  # bit period
    bp = Tb
    fc = round_to_nearest_multiple(fc)
    # Transmitting binary information as digital signal
    bit = np.array([])
    for n in range(len(x)):
        if x[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate([bit, se])

    t1 = np.arange(bp / 100, 100 * len(x) * (bp / 100) + bp / 100, bp / 100)
    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, linewidth=2.5)
    plt.grid(True)
    plt.axis([0, bp * len(x), -1, 2])
    plt.ylabel("Amplitude(Volt)")
    plt.xlabel("Time(ms)")
    plt.title("Message Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msgSignal = data.getvalue().hex()
    plt.figure()

    # Binary-PSK modulation

    br = 1 / bp  # bit rate
    f = br * 2  # carrier frequency

    fDigits = countNoOfDigits(fc)
    space = countSpace(fDigits) * 9   

    t2 = np.arange(bp / 99, bp + bp / 99, bp / space)
    ss = len(t2)
    m = np.array([])
    for i in range(len(x)):
        if x[i] == 1:
            y = Ac * np.cos(2 * np.pi * fc * f * t2)
        else:
            y = Ac * np.cos(2 * np.pi * fc * f * t2 + np.pi)
        m = np.concatenate([m, y])

    x_carrier = create_domain_AM()
    c = Ac * np.cos(2 * np.pi * fc * x_carrier)

    carrier = plot_graph(condition = condition, x = x_carrier, y = c, title = "Carrier Signal",color='g')


    # Modulated
    t3 = np.arange(bp / 99, bp * len(x) + bp / 99, bp / space)
    plt.subplot(3, 1, 2)
    plt.plot(t3, m, "r")
    plt.axis([0, bp * len(x), -Ac - 5, Ac + 5])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude(V)")
    plt.title("Modulated Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    modulatedSignal = data.getvalue().hex()
    plt.figure()

    return [msgSignal, carrier, modulatedSignal]


# ------- QPSK ---------------
def QPSK(Tb,Ac, fc, inputBinarySeq):

    x = inputBinarySeq.reshape(-1, 1)
    condition = 'line'
    bp = Tb
    fc = round_to_nearest_multiple(fc)
    # Transmitting binary information as digital signal
    bit = np.array([])
    for n in range(len(x)):
        if x[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate([bit, se])

    t1 = np.arange(bp / 100, 100 * len(x) * (bp / 100) + bp / 100, bp / 100)
    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, linewidth=2.5)
    plt.grid(True)
    plt.axis([0, bp * len(x), -1, 2])
    plt.ylabel("Amplitude(V)")
    plt.xlabel("Time(ms)")
    plt.title("Message Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msgSignal = data.getvalue().hex()
    plt.figure()

    br = 1 / bp  # bit rate
    f = br * 2  # carrier frequency

    fDigits = countNoOfDigits(fc)
    space = countSpace(fDigits) * 9   

    t2 = np.arange(bp / 99, bp + bp / 99, bp / space)
    ss = len(t2)
    s = np.array([])

    for i in range(0,len(x), 2):
        if x[i] == 0 and x[i+1] == 0:
            y = Ac * np.cos(2 * np.pi * fc * t2 + np.pi/4)
        elif x[i] == 0 and x[i+1] == 1:
            y = Ac * np.cos(2 * np.pi * fc * t2 + 3*(np.pi/4))
        elif x[i] == 1 and x[i+1] == 0:
            y = Ac * np.cos(2 * np.pi * fc * t2 + 5*(np.pi/4))
        elif x[i] == 1 and x[i+1] == 1:
            y = Ac * np.cos(2 * np.pi * fc * t2 + 7*(np.pi/4))
        s = np.concatenate([s, y])
        s = np.concatenate([s, y])

    x_carrier = create_domain_AM()
    c = Ac * np.cos(2 * np.pi * fc * x_carrier)

    carrier = plot_graph(condition = condition, x = x_carrier, y = c, title = "Carrier Signal",color='g')


    # Modulated
    t3 = np.arange(bp / 99, bp * len(x) + bp / 99, bp / space)
    plt.subplot(3, 1, 2)
    plt.plot(t3, s, "r")
    plt.axis([0, bp * len(x), -Ac - 5, Ac + 5])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude(V)")
    plt.title("Modulated Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    modulatedSignal = data.getvalue().hex()
    plt.figure()

    return [msgSignal, carrier, modulatedSignal]

# -------DPSK ----------
def DPSK(Tb,Ac, fc, inputBinarySeq):

    x = inputBinarySeq.reshape(-1, 1)
    condition = 'line'
    bp = Tb
    fc = round_to_nearest_multiple(fc)
    # Transmitting binary information as digital signal
    bit = np.array([])
    for n in range(len(x)):
        if x[n] == 1:
            se = np.ones(100)
        else:
            se = -np.ones(100)
        bit = np.concatenate([bit, se])

    t1 = np.arange(bp / 100, 100 * len(x) * (bp / 100) + bp / 100, bp / 100)
    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, linewidth=2.5)
    plt.grid(True)
    plt.axis([0, bp * len(x), -2, 2])
    plt.ylabel("Amplitude(V)")
    plt.xlabel("Time(ms)")
    plt.title("Message Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msgSignal = data.getvalue().hex()
    plt.figure()

    br = 1 / bp  # bit rate
    f = br * 2  # carrier frequency

    fDigits = countNoOfDigits(fc)
    space = countSpace(fDigits) * 9   

    t2 = np.arange(bp / 99, bp + bp / 99, bp / space)
    ss = len(t2)
    s = np.array([])
    prev_phase = 0

    for i in range(len(x)):
        if x[i] == 0:
            phase_change = np.pi
        else:
            phase_change = 0
            
        phase = prev_phase + phase_change

        y = Ac * np.cos(2 * np.pi * fc * t2 + phase)
        s = np.concatenate([s, y])

        # Update the previous phase for the next bit
        prev_phase = phase


    x_carrier = create_domain_AM()
    c = Ac * np.cos(2 * np.pi * fc * x_carrier)

    carrier = plot_graph(condition = condition, x = x_carrier, y = c, title = "Carrier Signal",color='g')

    # Modulated
    t3 = np.arange(bp / 99, bp * len(x) + bp / 99, bp / space)
    plt.subplot(3, 1, 2)
    plt.plot(t3, s, "r")
    plt.axis([0, bp * len(x), -Ac - 5, Ac + 5])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude(V)")
    plt.title("Modulated Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    modulatedSignal = data.getvalue().hex()
    plt.figure()

    return [msgSignal, carrier, modulatedSignal]

    