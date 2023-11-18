"""
نمودار تولید شده در مبحث فوریه و توان در EEG متقارن است به دلیل وجود رابطه‌های منظم و ترازشده بین موج‌های مختلف در فرایند تولید سیگنال‌های EEG است.
 سیگنال‌های EEG از انبوهی از فعالیت‌های نورونی در مغز تولید می‌شوند که هر یک در فرکانس ورودی خود فعالیت دارند. این فعالیت‌ها می‌توانند مثبت یا منفی باشند و با فرکانس‌های مختلف تولید می‌شوند.

نمودار توان در فوریه نشان می‌دهد که سیگنال EEG چه در فرکانس‌های مختلف قدرت دارد.
 این نمودار معمولاً سیمتری است زیرا سیگنال EEG تقریباً هم‌ساز باشد.
  به عبارت دیگر، فعالیت نورونی در هر دو نیمکره مغز تقریباً یکسان است و هنگامی که یک تغییر یا تحریک نورونی در یک نیمکره رخ می‌دهد، تقریباً همان تغییر یا تحریک در نیمکره دیگری نیز رخ می‌دهد.

علاوه بر این، نمودار توان در فوریه معمولاً شامل انعکاسی از فعالیت‌های هم‌زمان در هر دو نیمکره مغز است.
 به عبارت دیگر، اگر سیگنال EEG تحت تأثیر فعالیت نورونی از هر دو نیمکره مغز باشد،
  آنگاه نمودار تولید شده از آن نیز شامل فعالیت‌های مشابه در هر دو نیمکره خواهد بود. این باعث می‌شود که نمودار توان در فوریه متقارن به نظر آید.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.fft import fft
from scipy.signal import spectrogram

# Load the data from the mat file
data = loadmat('/Users/zsnd/Downloads/EEG_P2090.mat')
EEG = data['EEG_P2090_processed']

# Print the information of signals
num_channel = EEG.shape[0]
num_samples = EEG.shape[1]
print(f'Number of Channels is {num_channel}.')
print(f'Number of samples {num_samples}')

Fs = 500
time_duration = num_samples / Fs

print(f'The time duration of each signal is {time_duration}.')
print(f'The time duration of each signal in minute is {time_duration / 60}.')

# Plot the EEG signal from channel 2 and channel 1
time = np.arange(EEG.shape[1])

plt.figure(figsize=(10, 5))
plt.plot(time / Fs, EEG[1])
plt.title('EEG signal from channel 2')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Plot the first 30 channels (5*6 shape)
num_rows, num_cols = 5, 6
plt.figure(figsize=(18, 12))

for channel in range(num_channel):
    plt.subplot(num_rows, num_cols, channel + 1)
    plt.plot(time / Fs, EEG[channel])
    plt.title(f'EEG signal from channel {channel + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Zoom in to the first 5 seconds of the EEG signal from channel 1 and display its spectrogram
start_time, stop_time = 0, 5
start_sample = int(start_time * Fs)
stop_sample = int(stop_time * Fs)

plt.figure(figsize=(10, 5))
plt.plot(time[start_sample:stop_sample] / Fs, EEG[0][start_sample:stop_sample])
plt.title('EEG signal from channel 1 (0-5s)')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Display the spectrogram of the signal
nfft = 500
overlap = nfft // 2

freq, times, Sxx = spectrogram(EEG[0, start_sample:stop_sample], fs=Fs, nperseg=nfft, noverlap=overlap)
plt.pcolormesh(times, freq, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time (s)')
plt.title('Spectrogram (Channel 1, 0-5s)')
plt.show()

# Repeat the above procedure for different 'nfft' values (500, 250, 20)
nfft_values = [500, 250, 20]

for nfft in nfft_values:
    overlap = nfft // 2
    freq, times, Sxx = spectrogram(EEG[0, start_sample:stop_sample], fs=Fs, nperseg=nfft, noverlap=overlap)

    plt.pcolormesh(times, freq, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram (Channel 1, 0-5s, nfft={nfft})')
    plt.show()






