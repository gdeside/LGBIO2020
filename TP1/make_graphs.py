#####################################################################################################
# LGBIO2050 - TP1 : PCA & ICA
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
PLOT DATA POINTS IN A 2-DIMENSIONAL SPACE
INPUTS: 
    - x : list of coordinates along the first axis
    - y : list of coordinates along the second axis
    - title : title of the graph
    - x_label : label of the x axis
    - y_label : label of the y axis 
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def scatter_plot(x, y, title, x_label, y_label, show_fig, file_path=None): 
    plt.scatter(x, y, marker='.')
    plt.title(title)
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
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def ecg_plot(ecg_signals, show_fig, file_path=None): 
    # Same y scale for all channels
    bottom = np.amin(ecg_signals[1:ecg_signals.shape[0]])
    top    = np.amax(ecg_signals[1:ecg_signals.shape[0]])

    # One big figure to frame the whole
    fig = plt.figure(figsize=(8,8))
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

    ax0.set_ylabel('Amplitude (mv)')

    #fig.tight_layout() 
    

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
PLOT EEG SIGNALS DEPENDING ON THE TIME
INPUTS: 
    - eeg_signals : a matrix of [(n+1)xm] dimensions where n (nb of channels) << m 
                    first row must be the time vector! 
    - label : list of n strings with channel names (do not consider time)
    - show_fig : True if the plot must be displayed on screen, False otherwise
    - (file_path) : path where the graph must be saved (if needed)
--------------------------------------------------------------------------------------------------"""
def eeg_plot(eeg_signals, label, show_fig, file_path=None): 
    # Same y scale for all channels
    bottom = np.amin(eeg_signals[1:eeg_signals.shape[0]])
    top    = np.amax(eeg_signals[1:eeg_signals.shape[0]])

    # One big figure to frame the whole
    fig = plt.figure(figsize=(12,8))
    ax0 = fig.add_subplot(111) 
    plt.subplots_adjust(hspace=-0.5)
    ax0.tick_params(labelcolor='black', top=False, bottom=False, left=False, right=False)

    # Plot each channel
    for idx in range(1, eeg_signals.shape[0]):
        if idx == 1 :
            _ax = fig.add_subplot(eeg_signals.shape[0]-1, 1, idx)
            ax  = _ax
        else:   
            ax = fig.add_subplot(eeg_signals.shape[0]-1, 1, idx, sharex=_ax)
        if idx == eeg_signals.shape[0]-1:
            ax.tick_params(labelcolor='black', top=False, bottom=True, left=False, right=False)
            ax.patch.set_alpha(0)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (sec)')
        else:
            ax.axis('off')          
        ax.plot(eeg_signals[0], eeg_signals[idx],  linewidth=0.5)
        ax.set_ylim(bottom, top)
        plt.text(-0.45, 0, label[idx-1])

    ax0.get_yaxis().set_visible(False)
    ax0.get_xaxis().set_visible(False)

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
        
    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()