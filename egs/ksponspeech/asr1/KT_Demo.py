from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import subprocess
import scipy.io as sio
import scipy.io.wavfile
from scipy import signal
import sounddevice as sd
import pyaudio
import sys, os, time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

import difflib

import numpy as np
import wave

import KT_main

# constants
CHUNK = 1024             # samples per frame
WAVDIR = "data/test/wave"
TEXTDIR= "data/test/texts"
INF = 123456789

from difflib import SequenceMatcher

def show_diff(A, B):
    seqm = SequenceMatcher(None, A, B)

    """Unify operations between two compared strings
seqm is a difflib.SequenceMatcher instance whose a & b are strings"""
    output = []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            output.append(seqm.a[a0:a1])
        elif opcode == 'insert':
            continue
        elif opcode == 'delete':
            output.append("<font color=blue>" + seqm.a[a0:a1] + "</font>")
        elif opcode == 'replace':
            output.append("<font color=red>" + seqm.a[a0:a1] + "</font>")
        else:
            raise RuntimeError("unexpected opcode")
    return ''.join(output)

def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    cost = {}

    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return cost.get((c1, c2), 0.1)

    output = []
    cost_map = []
    cost_map = np.zeros((len(s1),len(s2)))
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            min_result = min(insertions, deletions, substitutions)
            current_row.append(min_result)

        cost_map[i] = np.array(current_row[1:])

            
        previous_row = current_row
    
    def ok1(p, q):
        return p > 0 and q > 0 and p < len(s1) and q < len(s2)
    def ok2(p):
        return p < len(s1)
    def ok3(q):
        return q < len(s2)
    x = [1, 1, 0] # substitution, insertion, deletion
    y = [1, 0, 1]
    
    i = 0
    j = 0
    if cost_map[i][j] != 0:
        output.append("<font color=red>" + s1[i] + "</font>")
    while True:
        sub, inst, dele = INF, INF, INF

        if ok1((i+x[0]), (j+y[0])):
            sub = min(sub, cost_map[i + x[0]][j + y[0]])
        if ok2(i+x[1]):
            inst = min(inst, cost_map[i + x[1]][j + y[1]])
        if ok3(j+y[2]):
            dele = min(dele, cost_map[i +x[2]][j + y[2]])

        cost_min = min(sub, inst, dele)
        if cost_min == sub:
            old_i = i
            old_j = j
            i = i + x[0]
            j = j + y[0]
            if cost_map[i][j] == cost_map[old_i][old_j]:
                output.append(s1[i])
            else:   
                output.append("<font color=red>" + s1[i] + "</font>")
        elif cost_min == inst:
            i = i + x[1]
            j = j + y[1]
            if cost_map[i][j] == cost_map[old_i][old_j]:
                output.append(s1[i])
            else:
                output.append("<font color=blue>" + s1[i] + "</font>")
        elif cost_min == dele:
            i = i + x[2]
            j = j + y[2]
            if cost_map[i][j] == cost_map[old_i][old_j]:
                output.append(s1[i])
            else:
                output.append("<font color=green>" + s1[i] + "</font>")
        
        if i == len(s1)-1:
            break
    
    return "".join(output)



def cutoffaxes(ax):  # facecolor='#000000'
    #ax.patch.set_facecolor(facecolor)

    ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


class Kaldi(QThread):
    finished = pyqtSignal()
    def __init__(self):
        super().__init__()

    def kaldi_decoding(self, Filename):
        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/kaldi_decode.sh --recog-name {}'.format(Filename), shell=True)
        self.finished.emit()


class E2E(QThread):
    finished = pyqtSignal()
    def __init__(self):
        super().__init__()

    def e2e_decoding(self, Filename):
        with open("n_split", 'r') as f:
            n_split=f.readlines()[0]
        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/e2e_decode.sh --n-split {} --recog-name {}'.format(n_split, Filename), shell=True)
        self.finished.emit()

class Play_wavFile(QObject):
    def __init__(self):
        super().__init__()

    def recoding_wav(self, wavFilePath):
        sample_rate, data = sio.wavfile.read(wavFilePath)

        times = np.arange(len(data))/float(sample_rate)

        sd.play(data,sample_rate)

class CanvasWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=50):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='none')
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")

        # buffer
        self.img_buffer = []

        # Initialize axis and lines
        gs = gridspec.GridSpec(1, 1)
        ax1 = self.fig.add_subplot(gs[:, :])
        #cutoffaxes(ax1, 'darkblue')
        cutoffaxes(ax1)
        self.axs = [ax1]
        self.fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        FigureCanvas.updateGeometry(self)

        # Compute init image to figure
        self._init_lines()


    def _init_lines(self, img=None):
        
        if img is None:
            line1, = self.axs[0].plot([]) #, animated=True, cmap='gray'
        else:
            self.axs[0].grid(which='both')
            self.axs[0].set_ylim([min(img), max(img)])
            line1, = self.axs[0].plot(np.zeros(np.shape(img))) #, animated=True, cmap='gray'
        self.lines = [line1]

    def update_(self, img):
        self._init_lines(img)
        self.lines[0].set_xdata(range(len(img)))
        self.lines[0].set_ydata(img)
        self.draw()
        

class KTDialog(QMainWindow, KT_main.Ui_KT):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

        self.wavFilePath = ""
        self.wavFileName = ""
        self.textFilePath = ""
        self.textFileName = ""

        self.sample_rate = None
        self.data = None
        # Thread
        self.play_thread = QtCore.QThread()
        self.play_thread.start()
        self.play_wavFile = Play_wavFile()
        self.play_wavFile.moveToThread(self.play_thread)

        self.kaldi_thread = QtCore.QThread()
        self.kaldi_thread.start()
        self.kaldi = Kaldi()
        self.kaldi.moveToThread(self.kaldi_thread)

        self.e2e_thread = QtCore.QThread()
        self.e2e_thread.start()
        self.e2e = E2E()
        self.e2e.moveToThread(self.e2e_thread)

        self.View_List = []
        self.Model = QtGui.QStandardItemModel(self)

        # Button Event
        self.Kaldi_play_button.clicked.connect(self.btnPlayKaldiClicked)
        self.Kaldi_play_button.pressed.connect(self.btnPlayKaldiPressed)
        self.Kaldi_play_button.released.connect(self.btnPlayKaldiReleased)
        self.Kaldi_stop_button.clicked.connect(self.btnStopKaldiClicked)
        self.Kaldi_stop_button.pressed.connect(self.btnStopKaldiPressed)
        self.Kaldi_stop_button.released.connect(self.btnStopKaldiReleased)

        self.E2E_play_button.clicked.connect(self.btnPlayE2EClicked)
        self.E2E_play_button.pressed.connect(self.btnPlayE2EPressed)
        self.E2E_play_button.released.connect(self.btnPlayE2EReleased)
        self.E2E_stop_button.clicked.connect(self.btnStopE2EClicked)
        self.E2E_stop_button.pressed.connect(self.btnStopE2EPressed)
        self.E2E_stop_button.released.connect(self.btnStopE2EReleased)

        self.kaldi.finished.connect(self.Display_KaldiScript)
        self.e2e.finished.connect(self.Display_E2EScript)

        self.FileList.clicked[QtCore.QModelIndex].connect(self.Display_Wavform)

        # Aux body
        self.canvas = CanvasWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.Wav_Widget.setLayout(layout)

        self.LoadFileList()
    # Reload Script
    def Display_OriginScript(self, textFileName):
        origin_file = open(textFileName, 'r')
        line = origin_file.readline()
        self.original_line=line.rstrip('\n')

        self.Original_Script.setText(self.original_line)


    def Display_KaldiScript(self):
        kaldi_file = open("KALDI_RESULT", "r")
        line = kaldi_file.readline()
        results = line.split('/')

        self.Kaldi_Script.setText(show_diff(results[0], self.original_line))
        # self.Kaldi_Script.setText(levenshtein(self.original_line, results[0]))
        self.Kaldi_WER.setText(results[1])
        self.Kaldi_CER.setText(results[2].rstrip('\n'))

    def Display_E2EScript(self):
        e2e_file = open("E2E_RESULT", "r")
        line = e2e_file.readline()
        results = line.split('/')

        self.E2E_Script.setText(show_diff(results[0], self.original_line))
        # self.E2E_Script.setText(levenshtein(self.original_line, results[0]))
        self.E2E_WER.setText(results[1])
        self.E2E_CER.setText(results[2].rstrip('\n'))

    # Button Press Event
    def btnPlayKaldiPressed(self):
        self.Kaldi_Decoding_label.setText("디코딩 진행중....")
        self.Kaldi_play_button.setStyleSheet("image: url(./Pictures/play_clicked.png);\nborder: 0px;")
    def btnStopKaldiPressed(self):
        self.Kaldi_stop_button.setStyleSheet("image: url(./Pictures/stop_clicked.png);\nborder: 0px;")
    def btnPlayE2EPressed(self):
        self.E2E_Decoding_label.setText("디코딩 진행중....")
        self.E2E_play_button.setStyleSheet("image: url(./Pictures/play_clicked.png);\nborder: 0px;")
    def btnStopE2EPressed(self):
        self.E2E_stop_button.setStyleSheet("image: url(./Pictures/stop_clicked.png);\nborder: 0px;")

    # Button Release Event
    def btnPlayKaldiReleased(self):
        self.Kaldi_play_button.setStyleSheet("image: url(./Pictures/play.png);\nborder: 0px;")
    def btnStopKaldiReleased(self):
        self.Kaldi_stop_button.setStyleSheet("image: url(./Pictures/stop.png);\nborder: 0px;")
    def btnPlayE2EReleased(self):
        self.E2E_play_button.setStyleSheet("image: url(./Pictures/play.png);\nborder: 0px;")
    def btnStopE2EReleased(self):
        self.E2E_stop_button.setStyleSheet("image: url(./Pictures/stop.png);\nborder: 0px;")

    
    # Button Click Event
    def btnPlayKaldiClicked(self):           # Play 버튼 클릭 시 
        # Play wavFile
        self.play_wavFile.recoding_wav(self.wavFilePath)

        # Start Decoding
        self.kaldi.kaldi_decoding(self.wavFileName.rstrip('.wav'))
        self.Kaldi_Decoding_label.setText("디코딩 완료")

    def btnPlayE2EClicked(self):           # Play 버튼 클릭 시 
        # Play wavFile
        self.play_wavFile.recoding_wav(self.wavFilePath)

        # Start Decoding
        self.e2e.e2e_decoding(self.wavFileName.rstrip('.wav'))
        self.E2E_Decoding_label.setText("디코딩 완료")

    def btnStopKaldiClicked(self):           # Stop 버튼 클릭 시
        sd.stop()

    def btnStopE2EClicked(self):           # Stop 버튼 클릭 시
        sd.stop()

    def LoadFileList(self):
        file_list = os.listdir(WAVDIR)
        for f in file_list:
            f_ext = os.path.splitext(f)
        
            if f_ext[1] == '.wav':
                listitem = QtGui.QStandardItem(f_ext[0] + f_ext[1])
                self.Model.appendRow(listitem)
        self.FileList.setModel(self.Model)

        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/feature_extraction.sh', shell=True)

     
    def Display_Wavform(self, index):           

        self.canvas.axs[0].cla()
        
        item = self.Model.itemFromIndex(index)

        self.wavFileName = item.text()
        self.wavFilePath = WAVDIR + "/" + self.wavFileName
        self.FileName_label.setText(self.wavFileName)

        self.textFileName = item.text().split('.')[0] + ".txt"
        self.textFilePath = TEXTDIR + "/" + self.textFileName
        self.Display_OriginScript(self.textFilePath)

        self.sample_rate, self.data = sio.wavfile.read(self.wavFilePath)

        self.canvas.update_(self.data)
        
        self.Kaldi_Decoding_label.setText("")
        self.E2E_Decoding_label.setText("")
        

app = QApplication(sys.argv)
dlg = KTDialog()
dlg.show()
app.exec_()
