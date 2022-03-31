# from pydub import AudioSegment
# song = AudioSegment.from_mp3("new_wav.mp3")
# print(song.frame_rate)
# # 44100
# print(song.channels)
# # song = song.set_channels(1)
# print(song.channels)
# print(type(song))
# song = (song.set_channels(2))
# print(song.channels)

# song.export("mono2.wav", format="wav")
# import torch as th
import math 
# def spectro(x, n_fft=2048, hop_length=None, pad=0):
#     *other, length = x.shape
#     x = x.reshape(-1, length)
#     z = th.stft(x,
#                 n_fft * (1 + pad),
#                 hop_length or n_fft // 4,
#                 window=th.hann_window(n_fft).to(x),
#                 win_length=n_fft,
#                 normalized=True,
#                 center=True,
#                 return_complex=True,
#                 pad_mode='reflect')
#     _, freqs, frame = z.shape
#     return z.view(*other, freqs, frame)


    
# def _spec( x):
#     hl = 512 #time detail           #multiply by 2 for increase     -    #hl should be 1/4 of nfft
#     nfft = 2048 # frequency detail  #multiply by 2 for increase  

    
#     z = spectro(x, nfft, hl)[..., :-1, :]
#     return z


# def ispectro(z, hop_length=None, length=None, pad=0):
#     *other, freqs, frames = z.shape
#     n_fft = 2 * freqs - 2
#     z = z.view(-1, freqs, frames)
#     win_length = n_fft // (1 + pad)
#     x = th.istft(z,
#                  n_fft,
#                  hop_length,
#                  window=th.hann_window(win_length).to(z.real),
#                  win_length=win_length,
#                  normalized=True,
#                  length=length,
#                  center=True)
#     _, length = x.shape
#     return x.view(*other, length)


if __name__== "__main__":
    import librosa, librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    # import torch
    file = "example5guitar.wav"
    '''able to convert any .wav file to spectrogram in pytorch and back''' 


    # torch.set_printoptions(precision=10)
    #numpy array                

    signal,sr = librosa.load(file,sr=44100)
    stft = librosa.core.stft(signal,hop_length=512,n_fft=2048) #overlap
    spectrogram = np.abs(stft) #first is frequency second is time

    # log spectrogram because that's how humans perceive sound 
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    librosa.display.specshow(log_spectrogram,sr=sr,hop_length=512)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig("guitar.png")
    plt.show()
    