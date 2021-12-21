import matplotlib.pyplot as plt
import numpy as np
import os
root = os.getcwd()
from pathlib import Path

# Some functions are inspired from LGBIO2020 - TP1 ICA & PCA

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


"""--------------------------------------------------------------------------------------------------
PLOT EEG SIGNALS DEPENDING ON THE TIME WITH THE TARGET
INPUTS: 
    - eeg_signals : a matrix of [nxm] dimensions where n (nb of channels) << m 
    - target : vector of [m] length
    - label : list of n strings with channel names (do not consider time)
    - time : vector of [m] length
--------------------------------------------------------------------------------------------------"""
def eeg_target_plot(eeg_signals,target,label,time):
    for i in range(np.shape(eeg_signals)[0]):
        fig, axs = plt.subplots(2,1,figsize=(16,10))
        fig.tight_layout()
        axs = axs.flatten()
        axs[0].plot(time[100000:150000],eeg_signals[i][100000:150000])
        axs[1].plot(time[100000:150000],target[100000:150000])
        name = "figures/target/signal_target_{}.png".format(label[i])
        fig.savefig(name)
        plt.close()
    return


"""--------------------------------------------------------------------------------------------------
PLOT FORIER TRANSFORM OF EEG SIGNALS (32 CHANNELS)
INPUTS: 
    - eeg_signals : a matrix of [nxm] dimensions where n s(nb of channels) << m 
    - label : list of n strings with channel names (do not consider time)
    - freq_acquisition : frequence of acquisition
--------------------------------------------------------------------------------------------------"""

def eeg32_freq_plot(eeg_signals,label,freq_acquisition):
    for i in range(4):
        fig, axs = plt.subplots(4, 2,figsize=(16,10))
        fig.tight_layout()
        axs =axs.flatten()
        for j in range(8):
            signal_freq = np.fft.fftshift(np.fft.fft(eeg_signals[i*8 + j]))
            signal_freq_abs = np.fft.fftshift(np.fft.fftfreq(signal_freq.size,d=1/freq_acquisition))
            axs[j].plot(signal_freq_abs[(len(signal_freq_abs)//2):],
                                                            (np.abs(signal_freq[(len(signal_freq_abs)//2):])))
            axs[j].set_title(label[i*8 +j])
            axs[j].set_ylim(0,150e4)
        name = "figures/frequential/signal_freq_{}.png".format(i)
        fig.savefig(name)
        plt.close()  



"""--------------------------------------------------------------------------------------------------
PLOT FORIER TRANSFORM OF EEG SIGNALS AFTER AND BEFORE FILTERING
INPUTS: 
    - eeg_signals : a matrix of [nxm] dimensions where n s(nb of channels) << m 
    - eeg_filtering_signals : a matrix of [nxm] dimensions where n s(nb of channels) << m 
    - label : list of n strings with channel names (do not consider time)
    - freq_acquisition : frequence of acquisition
--------------------------------------------------------------------------------------------------"""

def comparison_filtering_plot(eeg_signals,eeg_filtering_signals,freq_acquisition,label):
    for i in range(len(label)):
        fig, axs = plt.subplots(1, 2,figsize=(18,8))

        signal_freq = np.fft.fftshift(np.fft.fft(eeg_signals[i]))
        signal_freq_abs = np.fft.fftshift(np.fft.fftfreq(signal_freq.size,d=1/freq_acquisition))
        axs[0].plot(signal_freq_abs[(len(signal_freq_abs)//2):],(np.abs(signal_freq[(len(signal_freq_abs)//2):])))
        axs[0].set_ylim(0,150e4)
        axs[0].set_xlabel('freqence [Hz]')
        axs[0].set_ylabel('amplitude')
        axs[0].set_title("Fourier transform of {} before filtering".format(label[i]))

        signal_freq_1 = np.fft.fftshift(np.fft.fft(eeg_filtering_signals[i]))
        signal_freq_abs_1 = np.fft.fftshift(np.fft.fftfreq(signal_freq_1.size,d=1/freq_acquisition))

        axs[1].plot(signal_freq_abs_1[(len(signal_freq_abs_1)//2):],(np.abs(signal_freq_1[(len(signal_freq_abs_1)//2):])))
        axs[1].set_ylim(0,1e6)
        axs[1].set_xlim(0,50)
        axs[1].set_xlabel('freqence [Hz]')
        axs[1].set_ylabel('amplitude')
        axs[1].set_title("Fourier transform of {} after filtering".format(label[i]))

        fig.savefig("figures/frequential/signal_freq_w_{}_comparison.png".format(label[i]))
        plt.close()



"""--------------------------------------------------------------------------------------------------
PLOT DELAY BETWEEN TARGET AND EEG
INPUTS: 
    - eeg_signals : a vector of [m] length
    - target :  a vector of [m] length
    - time :  a vector of [m] length
--------------------------------------------------------------------------------------------------"""
def delay_plot(eeg_signal,target,time):
    for i in range(len(eeg_signal)-1):
        if not np.isnan(target[i]) and not np.isnan(target[i+1]):
            if target[i] != target[i+1]:
                fig, axs = plt.subplots(2, 1,figsize=(16,10))
                axs[0].plot(time[i-6000:i+6000],eeg_signal[i-6000:i+6000])
                axs[0].set_xlim(time[i-6000],time[i+6000])
                axs[1].plot(time[i-6000:i+6000],target[i-6000:i+6000])
                axs[1].set_xlim(time[i-6000],time[i+6000])
                axs[1].set_xlabel("time[ms]")
                name = "figures/delay/delay_nofilter_{}.png".format(i)
                fig.savefig(name)
                plt.close()
