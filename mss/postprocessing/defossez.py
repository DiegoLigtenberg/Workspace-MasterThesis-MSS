# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th

from scipy.stats import wasserstein_distance

# from .apply import apply_model
# from .audio import convert_audio, save_audio
# # from . import distrib
# from .utils import DummyPoolExecutor

import matplotlib.pyplot as plt
import soundfile as sf

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def scoring():
    """
    Add scoring function in the starter kit for participant's reference
    """
    def sdr(references, estimates):
        # compute SDR for one song
        delta = 1e-7  # avoid numerical errors
        num = np.sum(np.square(references), axis=(1, 2))
        den = np.sum(np.square(references - estimates), axis=(1, 2))
        # print(references[0][::400].shape)
        # print(references[0][::400][:,0].shape)
        num2 = np.sum(wasserstein_distance(references[0][::1][:,0],references[0][::1][:,0]),axis=0)
        den2 = np.sum(wasserstein_distance(references[0][::1][:,0],estimates[0][::1][:,0]),axis=0)

        num2+= delta
        den2+= delta
        num += delta
        den += delta
        print(num)
        print(den,"\n")

        print(num2)
        print( "wasserstein",den2,10*np.log10(num2/den2))
        return 10 * np.log10(num  / den)

    # music_names = self.get_all_music_names()
    music_names = ["1","2"]
    instruments = ["other"]
    scores = {}
    for music_name in music_names:
        print("Evaluating for: %s" % music_name)
        scores[music_name] = {}
        references = []
        estimates = []
        for instrument in instruments:
            reference_file = r"track_output/other_target.wav"
            estimate_file = r"track_output/other_predict.wav"

            reference, _ = sf.read(reference_file)
            estimate, _ = sf.read(estimate_file)

          

            # sdrr = signaltonoise(estimate,reference)
            # print("sdr:\t",sdrr)
            # print(reference)

            ref = np.mean(reference)
            std = np.std(reference)

            reference = (reference - ref) / std
            estimate = (estimate -ref) / std
            reference = reference[:len(estimate)-0] 
            estimate = estimate[0:]  # estimate is 30 delayed from reference
            # print(reference.shape,estimate.shape)
            # asd
            # print(reference[0:100],reference.shape)
            # print(estimate[0:100],estimate.shape)

            plt.figure(1)
            plt.title("signal wave")
            plt.plot(estimate[2400:3000:],"r") #+30
            plt.plot(reference[2400:3000:])
            plt.show()
            # asd
            reference = np.vstack((reference, np.zeros_like(reference))).T
            estimate = np.vstack((estimate, np.zeros_like(estimate))).T
            references.append(reference)
            estimates.append(estimate)
            
        references = np.stack(references)
        estimates = np.stack(estimates)
        references = references.astype(np.float32)
        estimates = estimates.astype(np.float32)
        print(references.shape)
        # print(5/0)
        song_score = sdr(references, estimates).tolist()
        # print(song_score)
        # scores[music_name]["sdr_bass"] = song_score[0]
        # scores[music_name]["sdr_drums"] = song_score[1]
        scores[music_name]["sdr_other"] = song_score[0]
        # scores[music_name]["sdr_vocals"] = song_score[3]
        scores[music_name]["sdr"] = np.mean(song_score)
    return scores

print(scoring())
import librosa
# file_reference = r"track_output/other_target.wav"
# file_estimate = r"track_output/other_predict.wav"

# signal_reference,sr = librosa.load(file_reference,mono=False,sr=44100)
# signal_estimate,sr = librosa.load(file_estimate,mono=False,sr=44100)
# signal_reference = np.vstack((signal_reference, np.zeros_like(signal_reference)))
# signal_estimate = np.vstack((signal_estimate, np.zeros_like(signal_estimate)))

# print(signal_reference.shape)
# signal_reference = th.from_numpy(signal_reference).t().float()
# signal_estimate = th.from_numpy(signal_estimate).t().float()

# print(signal_reference.shape)
# # N_FFT = 4096
# # HOP_LENGTH = 1024
# # SAMPLE_RATE = 44100 #41100
# eval_track(signal_reference,signal_estimate,4096,1024)

'''
def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    `new_only` means using only the MDX definition of the SDR, which is much faster to evaluate.
    """

    args = solver.args

    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    if args.test.nonhq is None:
        test_set = musdb.DB(args.dset.musdb, subsets=["test"], is_wav=True)
    else:
        test_set = musdb.DB(args.test.nonhq, subsets=["test"], is_wav=False)
    src_rate = args.dset.musdb_samplerate

    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    indexes = range(0,len(test_set),1) #range(distrib.rank, len(test_set), distrib.world_size)
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints,
                          name='Eval')
    pendings = []

    pool = futures.ProcessPoolExecutor if args.test.workers else DummyPoolExecutor
    with pool(args.test.workers) as pool:
        for index in indexes:
            track = test_set.tracks[index]

            mix = th.from_numpy(track.audio).t().float()
            if mix.dim() == 1:
                mix = mix[None]
            mix = mix.to(solver.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels) # converts audio to used samplerate of model

            #input here should be a waveform mixture original and estimate
            # Mix -> model -----> Estiamte + Target 
            estimates = apply_model(model, mix[None],
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]

                                    # make an apply_model function
            estimates = estimates * ref.std() + ref.mean()
            estimates = estimates.to(eval_device)

            references = th.stack(
                [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
            if references.dim() == 2:
                references = references[:, None]
            references = references.to(eval_device)
            references = convert_audio(references, src_rate,
                                       model.samplerate, model.audio_channels)
            if args.test.save:
                folder = solver.folder / "wav" / track.name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    save_audio(estimate.cpu(), folder / (name + ".mp3"), model.samplerate)

            pendings.append((track.name, pool.submit(
                eval_track, references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)')
        tracks = {}
        for track_name, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending
            tracks[track_name] = {}
            for idx, target in enumerate(model.sources):
                tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist()
                    }
                    tracks[track_name][target].update(values)

        all_tracks = {}
        for src in range(1) #range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = [
                    np.nanmedian(all_tracks[track][source][metric_name])
                    for track in all_tracks.keys()]
                mean = np.mean(medians)
                median = np.median(medians)
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med" + "_" + source] = median
                avg += mean / len(model.sources)
                avg_of_medians += median / len(model.sources)
            result[metric_name.lower()] = avg
            result[metric_name.lower() + "_med"] = avg_of_medians
        return result
'''