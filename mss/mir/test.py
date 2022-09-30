# cel     453.0
# cla     531.0
# flu     538.0
# gac     878.0
# gel    1215.0
# org     852.0
# pia    1096.0
# sax     720.0
# tru     633.0
# vio     656.0
# voi    1339.0

from re import M
from matplotlib.pyplot import close
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
VISUALISATION_FOLDER = "mss_evaluate_data/visualisation/"

def visualize_loss():
    model =3
    x_msa = np.load(f"{VISUALISATION_FOLDER}loss_metrics/database_msa_{model}.npy")
    x_sdr = np.load(f"{VISUALISATION_FOLDER}loss_metrics/database_sdr_{model}.npy")
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.plot(x_msa, label='msa loss')
    # ax.plot(np.ones_like(x_msa) * np.mean(x_msa), label = 'avg msa')
    # ax.set(title='MSA loss per song in MUSDB test')
    # plt.xlabel("test song")
    # plt.ylabel("MSA")
    # ax.legend()
    # folder = f"{VISUALISATION_FOLDER}loss_metrics/"
    # if not os.path.exists(folder): os.makedirs(folder)
    # plt.show()
    # fig.savefig(f"{folder}msa_loss{model}")
    # close(fig)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_sdr, label='sdr loss')
    ax.plot(np.ones_like(x_sdr)*np.mean(x_sdr), label = 'avg sdr')
    # ax.set(title='SDR loss per song in MUSDB test')
    plt.xlabel("test song")
    plt.ylabel("SDR")
    ax.legend()
    folder = f"{VISUALISATION_FOLDER}loss_metrics/"
    if not os.path.exists(folder): os.makedirs(folder)
    plt.ylim((-8,15))
    # fig.savefig(f"{folder}1_sdr_loss_with_post")
    plt.show()
    close(fig)




visualize_loss()





from ast import Raise
import pathlib
import numpy as np

a = [1,2,3]
b = [2,3,4]

Raise
a = 4 if 5==4 else 5
print(a)

asd
for path in pathlib.Path("G:/Thesis/MIR_datasets/test_dataset/spectrogram_with_post").iterdir():
    if path.is_file():
        old_name = path.stem
        old_extension = path.suffix
        directory = path.parent
        new_name = "z_" + old_name + old_extension
        path.rename(pathlib.Path(directory, new_name))

asd

from numpy import average
import numpy as np
import matplotlib.pyplot as plt
a=[
0.8101045296167249,
0.7530055214084801,
0.7335087899359265,
0.7737890601019092,
0.6976591528362646,
0.8204060628742516,
0.7959382389762136,
0.7019140989729225,
0.87195720826219]
print(np.mean(a))
asd
import numpy as np

x1 = np.linspace(0.0, 5.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
x2 = np.linspace(0.0, 2.0)
y2 = np.cos(2 * np.pi * x2)
fig, axes = plt.subplots(6,2)
# print(axes)
for i in range(2):
  axes[i+1][i].plot(x1,y1,"o-")
  axes[0][i].set_ylabel("dada")

plt.show()
asd


axes[0]
val = [  453.0,
     531.0,
    538.0,
     878.0,
   1215.0,
    852.0,
   1096.0,
    720.0,
    633.0,
     656.0,
   1339.0]


tot = 453+531+537+878+1215+852+1096+720+633+656+1339
print(tot)
nw  = [tot/(11*x) for x in val]
print (nw) 

a = {1:2,3:4,5:6}
print(a)

q = [
0.5054171180931744,
0.8448060075093867,
0.8279387493123052,
0.7365038024794249,
0.7400041053912586,
0.7144084965314016,
0.6569791924696556,
0.799934505988024,
0.8065794964529143,
0.685652038593215,
0.8764449011045466,
]

c = [
0.44110115236875796,
0.6389236545682103,
0.7972675591417567,
0.7715178664444211,
0.7453337829714235,
0.7197955675609308,
0.6115679960366609,
0.8407279191616766,
0.7999200166921686,
0.6560327834837638,
0.9030310814282045,
]
print(average(c))
print(average(q))