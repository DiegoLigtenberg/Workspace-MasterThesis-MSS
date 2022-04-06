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
    