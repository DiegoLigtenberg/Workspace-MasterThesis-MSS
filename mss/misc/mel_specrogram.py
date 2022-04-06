
'''
# norm_stft_l = librosa.db_to_amplitude(log_stft_l)
mel_spectrogram = librosa.feature.melspectrogram(signal_l,n_fft=4096,hop_length=1024,sr=22050,n_mels=500)
mel_spectrogram = np.abs(mel_spectrogram)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# log_mel_spectrogram[450:] = 0
fig, ax = plt.subplots()

img = librosa.display.specshow(log_mel_spectrogram, y_axis='mel', sr=22050, hop_length=1024,
                    x_axis='time', ax=ax)
ax.set(title='Log-frequency power spectrogram')
# ax.label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()


mel_spectrogram = librosa.db_to_power(log_mel_spectrogram)
print(mel_spectrogram.shape)


import time					
tic = time.perf_counter() # Start Time


y = librosa.feature.inverse.mel_to_stft(mel_spectrogram,sr=22050,n_fft=4096)
toc = time.perf_counter() # End Time
print((toc-tic))
y = librosa.griffinlim(y)
'''
