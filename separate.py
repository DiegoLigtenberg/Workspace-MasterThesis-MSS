## edit
import sys
import subprocess
from audio import AudioFile, convert_audio, save_audio, encode_mp3
import torchaudio as ta

# comes from separate.py
def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'Ffmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


# from separate import load_track
import time
from os import path
from pydub import AudioSegment
start = time.time()
print("hello")
# from moviepy.editor import * #slow
# video = VideoFileClip('Steven Clark - Bounty.stem.mp4')
# video.audio.write_audiofile('new_wav.mp3')

# files
# src = ("Steven Clark - Bounty.stem.mp4")
# dst = ("new_wav")

# # convert mp3 to wav
# sound = AudioSegment.from_mp3(src)
# sound.export(dst, format="wav")

wav = load_track("database/train/Steven Clark - Bounty.stem.mp4", 2, 44100)
print(wav.shape)

from view_spectrogram import stft
# save_audio(wav,"test.wav")

# encode_mp3(wav,"test2.mp3")
# save_audio(wav,"test3.wav",44100)
import playsound
# playsound.playsound("new_wav.mp3",True)
# playsound.playsound("test3.wav",True)




end = time.time()
print(end - start)

