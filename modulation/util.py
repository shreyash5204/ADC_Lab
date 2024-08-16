import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def triangular(f,A,x):
    # return np.absolute(np.fmod(np.absolute(x),2*A)-A)
    return A * np.abs((2 * x * f) % 2 - 1)
    

def plot_graph(x,y,title,xlabel="Time",ylabel="Amplitude",color="b",condition="line",text=""): # changed to line from scattter
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y_smooth = np.interp(x_smooth, x, y) # interpolates the points

    plt.style.use('seaborn') # use seaborn style for ploting
    fig, ax = plt.subplots() #v create new figure and set of subplots
    fig = plt.figure(figsize=(20,3)) # mentioning figure sixe(20 inches, 3 inches)
    plot_axis(fig,ax) # calls plot_axis function
    s = [3 for i in x] # for each point in signal give the marker point size 3
    plt.title(title) # set plot tiltle
    plt.xlabel(xlabel,loc="left") # set plot xlabel
    plt.ylabel(ylabel,loc="top") # set plot ylabel


    if text!="":
        plt.text(-210, 5, text, fontsize=14, ha='center', va='center',rotation=90)

    plt.plot(x_smooth,y_smooth,c=color)
    fig.tight_layout() # adjust layout of figure

    data = BytesIO() # stores raw image data
    fig.savefig(data,format="png") # saves figure in png format
    data.seek(0)
    encoded_image = data.getvalue().hex() #retrieves binary data and converts it to hexadecimal string
    plt.close() # close the figure
    return encoded_image

def plot_axis(fig,ax):
    ax = fig.add_subplot(1, 1, 1) # plot with one row , one colum and sublplot index 1
    ax.spines['left'].set_position('center') # moving y axis to center
    ax.spines['bottom'].set_position('zero') # moving x axis to center
    ax.spines['right'].set_color('black') # when colour is set to black they will be removed from the plot area
    ax.spines['top'].set_color('black')
    ax.xaxis.set_ticks_position('bottom') # positioning the tick marks
    ax.yaxis.set_ticks_position('left')

def create_domain_AM():
    x= np.linspace(-1/1000,1/1000,1000000) # creates an array of points between -200 and 200 with 10000 points in between
    return x

def destructure_dict(d, *keys):
    return (d[k] for k in keys)
