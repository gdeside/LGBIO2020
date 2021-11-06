#####################################################################################################
# LGBIO2050 - TP2 : FFT & WAVELETS
# Helper Functions to plot signals
#####################################################################################################

import matplotlib.pyplot as plt

import numpy as np 

import os
root = os.getcwd()
from pathlib import Path

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rcParams.update({'font.size': 16}) 

"""--------------------------------------------------------------------------------------------------
PLOT DATA SIGNALS IN A 2-DIMENSIONAL SPACE AND THE LOCAL MAXIMA
INPUTS: 
    - x : list of coordinates along the first axis
    - y : list of coordinates along the second axis
    - max_idx : indexes of maxima
    - x_label : label of the x axis
    - y_label : label of the y axis 
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - time : None or list of 2 indexes to plot subpart of time signal [start,end]
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def max_plot(x, y, max_idx, x_label, y_label, show_fig, time = None, file_path=None): 
    if time is not None: 
        x, y    = x[time[0]:time[1]], y[time[0]:time[1]]
        max_idx = [i-time[0] for i in max_idx if (i >= time[0] and i <= time[1])]
    plt.plot(x, y)
    plt.scatter(x[max_idx], y[max_idx], marker='.', color='red', s=100)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen 
    if show_fig:
        plt.show()
    plt.close()


"""--------------------------------------------------------------------------------------------------
PLOT ECG SIGNALS DEPENDING ON THE TIME
INPUTS: 
    - ecg_signals : a matrix of [(n+1)xm] dimensions where n (nb of channels) << m 
                    first row must be the time vector! 
    - label : list of n strings with channel names (do not consider time)
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - time : None or list of 2 indexes to plot subpart of time signal [start,end]
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def ecg_plot(ecg_signals, label, show_fig, time = None, file_path=None): 
    if time is not None: 
        ecg_signals = ecg_signals[:, time[0]:time[1]]
    # Same y scale for all channels
    bottom = np.amin(ecg_signals[1:ecg_signals.shape[0]])
    top    = np.amax(ecg_signals[1:ecg_signals.shape[0]])

    # One big figure to frame the whole
    fig = plt.figure(figsize=(12,4))
    ax0 = fig.add_subplot(111) 
    plt.subplots_adjust(hspace=-0.2)
    ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # Plot each channel
    for idx in range(1, ecg_signals.shape[0]):
        if idx == 1 :
            _ax = fig.add_subplot(ecg_signals.shape[0]-1, 1, idx)
            ax  = _ax
        else:   
            ax = fig.add_subplot(ecg_signals.shape[0]-1, 1, idx, sharex=_ax)
        if idx == ecg_signals.shape[0]-1:
            ax.tick_params(labelcolor='black', top=False, bottom=True, left=False, right=False)
            ax.patch.set_alpha(0)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (sec)')
        else:
            ax.axis('off')          
        ax.plot(ecg_signals[0], ecg_signals[idx],  linewidth=1)
        ax.set_ylim(bottom, top)
        plt.text(-0.45, 0, label[idx-1])

    ax0.set_ylabel('Amplitude (mv)')

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()


"""--------------------------------------------------------------------------------------------------
PLOT ECG SIGNALS DEPENDING ON THE TIME WITH FFT
INPUTS: 
    - ecg_signals : a matrix of [(n+1)xm] dimensions where n (nb of channels) << m 
                    first row must be the time vector! 
    - fft : a matrix of [(n+1)xm] dimensions where n (nb of channels) << m 
                    first row must be the frequency vector!
    - label : list of n strings with channel names (do not consider time)
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - time : None or list of 2 indexes to plot subpart of time signal [start,end]
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def ecg_plot_withfft(ecg_signals, fft, label, show_fig, time = None, file_path=None): 
    if time is not None: 
        ecg_signals = ecg_signals[:, time[0]:time[1]]
    # Same y scale for all channels
    bottom1 = np.amin(ecg_signals[1:ecg_signals.shape[0]])
    top1    = np.amax(ecg_signals[1:ecg_signals.shape[0]])
    bottom2 = np.amin(fft[1:ecg_signals.shape[0]])
    top2    = np.amax(fft[1:ecg_signals.shape[0]])


    # One big figure to frame the whole
    fig = plt.figure(figsize=(12,8))
    ax0 = fig.add_subplot(121) 
    ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # Plot each time channel
    for idx in range(1, ecg_signals.shape[0]):
        if idx == 1 :
            _ax = fig.add_subplot(ecg_signals.shape[0]-1, 2, 2*(idx-1)+1)
            ax  = _ax
        else:   
            ax = fig.add_subplot(ecg_signals.shape[0]-1, 2, 2*(idx-1)+1, sharex=_ax)
        if idx == ecg_signals.shape[0]-1:
            ax.tick_params(labelcolor='black', top=False, bottom=False, left=False, right=False)
            ax.set_xlabel('Time (sec)')
        else:
            ax.get_xaxis().set_visible(False)
        ax.set_ylim(bottom1, top1)  
        ax.plot(ecg_signals[0], ecg_signals[idx],  linewidth=1)
        
    # Plot each frequency channel
    ax1 = fig.add_subplot(122) 
    ax1.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for idx in range(1, fft.shape[0]):
        if idx == 1 :
            _ax2 = fig.add_subplot(fft.shape[0]-1, 2, 2*idx)
            ax2  = _ax2
        else:   
            ax2 = fig.add_subplot(fft.shape[0]-1, 2, 2*idx, sharex=_ax2)
        if idx == fft.shape[0]-1:
            ax2.tick_params(labelcolor='black', top=False, bottom=True, left=False, right=False)
            ax2.patch.set_alpha(0)
            ax2.set_xlabel('Frequency (Hz)')
        else:
            ax2.get_xaxis().set_visible(False)
          
        ax2.plot(fft[0], fft[idx],  linewidth=1)
        ax2.set_ylim(bottom2, top2)
        plt.text(fft[0][-1]*0.65, top2*0.85, label[idx-1])        

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()


"""--------------------------------------------------------------------------------------------------
PLOT DWT OF A SIGNAL
INPUTS: 
    - signal : original sigal vector (without time)
    - approx_coeffs : list (1xn) of n lists (1x|n|) with approximation coefficients in decreasing size |n|
    - detail_coeffs : list (1xn) of n lists (1x|n|) with detail coefficients in decreasing size |n|
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""

def dwt_plot(signal, approx_coeffs, detail_coeffs, show_fig, file_path=None): 

    fig = plt.figure(figsize=(20,10))
    ax_main = fig.add_subplot(len(approx_coeffs) + 1, 1, 1)
    ax_main.plot(signal, linewidth=1)
    ax_main.set_xlim(0, len(signal) - 1)
    ax_main.get_xaxis().set_visible(False)
    ax_main.get_yaxis().set_visible(False)


    for i, y in enumerate(approx_coeffs):
        if i == len(approx_coeffs)-1:
            ax = fig.add_subplot(len(approx_coeffs) + 1, 2, 3 + i * 2)
            ax.plot(y, 'teal', linewidth=1)
            ax.set_xlim(0, len(y) - 1)
            ax.patch.set_alpha(0)
            ax.get_yaxis().set_visible(False)
            plt.text(-(len(y) - 1)*0.15, np.mean(y), "A%d" % (i + 1))
        
    for i, y in enumerate(detail_coeffs):
        ax = fig.add_subplot(len(detail_coeffs) + 1, 2, 4 + i * 2)
        ax.plot(y, 'crimson', linewidth=1)
        ax.set_xlim(0, len(y) - 1)
        ax.get_yaxis().set_visible(False)
        if i != len(approx_coeffs)-1:
            ax.get_xaxis().set_visible(False)
        plt.text(-(len(y) - 1)*0.15, np.mean(y), "D%d" % (i + 1))

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()

"""--------------------------------------------------------------------------------------------------
PLOT NEUROGRAM SIGNAL DEPENDING ON THE TIME WITH FFT
INPUTS: 
    - signals : a matrix of [(n+1)xm] dimensions where n (nb of channels) << m 
                    first row must be the time vector! 
    - fft : a matrix of [(n+1)xm] dimensions where n (nb of channels) << m 
                    first row must be the frequency vector!
    - label : list of n strings with channel names (do not consider time)
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - time : None or list of 2 indexes to plot subpart of time signal [start,end]
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def neurogram_plot_withfft(ecg_signals, fft, label, show_fig, time = None, file_path=None): 
    if time is not None: 
        ecg_signals = ecg_signals[:, time[0]:time[1]]
    # One big figure to frame the whole
    fig = plt.figure(figsize=(14,10))
    ax0 = fig.add_subplot(121) 
    ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # Plot each time channel
    for idx in range(1, ecg_signals.shape[0]):
        if idx == 1 :
            _ax = fig.add_subplot(ecg_signals.shape[0]-1, 2, 2*(idx-1)+1)
            ax  = _ax
        else:   
            ax = fig.add_subplot(ecg_signals.shape[0]-1, 2, 2*(idx-1)+1, sharex=_ax)
        if idx == ecg_signals.shape[0]-1:
            ax.tick_params(labelcolor='black', top=False, bottom=False, left=False, right=False)
            ax.set_xlabel('Time (sec)')
        else:
            ax.get_xaxis().set_visible(False)
        ax.plot(ecg_signals[0], ecg_signals[idx],  linewidth=1)
        
    # Plot each frequency channel
    ax1 = fig.add_subplot(122) 
    ax1.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for idx in range(1, fft.shape[0]):
        if idx == 1 :
            _ax2 = fig.add_subplot(fft.shape[0]-1, 2, 2*idx)
            ax2  = _ax2
        else:   
            ax2 = fig.add_subplot(fft.shape[0]-1, 2, 2*idx, sharex=_ax2)
        if idx == fft.shape[0]-1:
            ax2.tick_params(labelcolor='black', top=False, bottom=True, left=False, right=False)
            ax2.patch.set_alpha(0)
            ax2.set_xlabel('Frequency (Hz)')
        else:
            ax2.get_xaxis().set_visible(False)
          
        ax2.plot(fft[0], fft[idx],  linewidth=1)
        plt.text(fft[0][-1]*0.6, np.amax(fft[idx])*0.9, label[idx-1])        

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()
