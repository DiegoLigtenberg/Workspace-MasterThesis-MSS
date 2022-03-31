import math
import wave
import struct

# https://stackoverflow.com/questions/33879523/python-how-can-i-generate-a-wav-file-with-beeps
import librosa
import matplotlib.pyplot as plt
import librosa.display

'''
# sr == sampling rate 
x, sr = librosa.load("tensorsong.wav", sr=44100)

# stft is short time fourier transform
X = librosa.stft(x,hop_length=1024,n_fft=2048)            #overlap #431 - 2047   -> #nfft is number of samples per fft

# convert the slices to amplitude
Xdb = librosa.amplitude_to_db(abs(X))

# ... and plot, magic!
plt.figure(figsize=(14, 5))
Xdb[:500] = 0

print(Xdb.shape)
librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
plt.colorbar()

plt.show()
'''
def save_wav(file_name,audio_tensor,nchanels=1):
    # Open up a wav file
    wav_file=wave.open(file_name,"w")

    # wav params
    nchannels = nchanels
    sampwidth = 2
    sample_rate = 44100.0

    # 44100 is the industry standard sample rate - CD quality.  If you need to
    # save on file size you can adjust it downwards. The stanard for low quality
    # is 8000 or 8kHz.
    nframes =  max(audio_tensor.shape) #len(audio_tensor)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))

    # WAV files here are using short, 16 bit, signed integers for the 
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.

    if nchannels == 1: # mono sound
        for sample in audio_tensor:
            sample = sample[0]
            wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))
    if nchannels == 2: # stereo sound
        for sample in audio_tensor:
            for stereo in sample: # 1 left ear, 1 right ear iteratively
                wav_file.writeframes(struct.pack('h', int(stereo * 32767.0 )))

    wav_file.close()
    # return

