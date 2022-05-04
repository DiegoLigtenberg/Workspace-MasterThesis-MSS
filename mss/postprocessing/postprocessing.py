from audioop import minmax
from math import prod
from mss.preprocessing.preprocesssing import MinMaxNormalizer

import numpy as np
import matplotlib.pyplot as plt
# from auto_encoder_vanilla import VariationalAutoEncoder
from mss.models.auto_encoder import AutoEncoder
from mss.models.atrain import load_fsdd
import librosa, librosa.display
from scipy.io import wavfile
from scipy.signal import wiener

import museval
import musdb
output_dir = "track_output"
estimates_dir = "track_output"

mus_train = musdb.DB(root="databases/database",subsets="train", split='train',download=False,is_wav=False)


def estimate_and_evaluate(track):
    # generate your estimates
    estimates = {
        'other': track.audio,
        'accompaniment': track.audio
    }

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=output_dir
    )

    # print nicely formatted mean scores
    print(scores)

    # return estimates as usual
    return estimates
estimate_and_evaluate(mus_train[0])