from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject

import subprocess
import scipy.io as sio
import scipy.io.wavfile
from scipy import signal
import sounddevice as sd
import pyaudio
import sys, os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

import numpy as np
import wave

import KT_main

# constants
CHUNK = 1024             # samples per frame

def cutoffaxes(ax):  # facecolor='#000000'
    #ax.patch.set_facecolor(facecolor)

    ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


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

        self.OriginScript = "Origin"
        self.KaldiScript = "Kaldi"
        self.E2EScript = "E2E"

        self.KaldiWER = ""
        self.KaldiCER = ""
        self.E2EWER = ""
        self.E2ECER = ""

        self.wavFilePath = ""
        self.wavFileName = ""
        self.sample_rate = None
        self.data = None

        # Button Event
        self.play_button.clicked.connect(self.btnPlayClicked)
        self.play_button.pressed.connect(self.btnPlayPressed)
        self.play_button.released.connect(self.btnPlayReleased)

        self.stop_button.clicked.connect(self.btnStopClicked)
        self.stop_button.pressed.connect(self.btnStopPressed)
        self.stop_button.released.connect(self.btnStopReleased)
        
        self.openWav_button.clicked.connect(self.btnWavClicked)
        self.openWav_button.pressed.connect(self.btnWavPressed)
        self.openWav_button.released.connect(self.btnWavReleased)

        # Aux body
        self.canvas = CanvasWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.Wav_Widget.setLayout(layout)

    # Reload Script
    def Display_OriginScript(self):
        origin_file = open("data/test/texts/test_18.txt", 'r')
        line = origin_file.readline()

        self.OriginScript = line
        self.Original_Script.setText(self.OriginScript)

    def Display_KaldiScript(self):
        kaldi_file = open("KALDI_RESULT", "r")
        line = kaldi_file.readline()
        results = line.split('/')

        self.KaldiScript = results[0]
        self.KaldiWER = results[1]
        self.KaldiCER = results[2].strip('\n')

        self.Kaldi_Script.setText(self.KaldiScript)
        self.Kaldi_WER.setText(self.KaldiWER)
        self.Kaldi_CER.setText(self.KaldiCER)

    def Display_E2EScript(self):
        e2e_file = open("E2E_RESULT", "r")
        line = e2e_file.readline()
        results = line.split('/')

        self.E2EScript = results[0]
        self.E2EWER = results[1]
        self.E2ECER = results[2].strip('\n')

        self.E2E_Script.setText(self.E2EScript)
        self.E2E_WER.setText(self.E2EWER)
        self.E2E_CER.setText(self.E2ECER)

        self.E2E_WER.setAlignment(QtCore.Qt.AlignCenter)
        self.E2E_CER.setAlignment(QtCore.Qt.AlignCenter)


    # Button Press Event
    def btnPlayPressed(self):
        self.play_button.setStyleSheet("image: url(./Pictures/play_clicked.png);\nborder: 0px;")
    def btnStopPressed(self):
        self.stop_button.setStyleSheet("image: url(./Pictures/stop_clicked.png);\nborder: 0px;")
    def btnWavPressed(self):
        self.openWav_button.setStyleSheet("image: url(./Pictures/wav_clicked.png);\nborder: 0px;")

    # Button Release Event
    def btnPlayReleased(self):
        self.play_button.setStyleSheet("image: url(./Pictures/play.png);\nborder: 0px;")
    def btnStopReleased(self):
        self.stop_button.setStyleSheet("image: url(./Pictures/stop.png);\nborder: 0px;")
    def btnWavReleased(self):
        self.openWav_button.setStyleSheet("image: url(./Pictures/wav.png);\nborder: 0px;")

    # Options
    def recoding_wav(self):

        times = np.arange(len(self.data))/float(self.sample_rate)

        sd.play(self.data, self.sample_rate)

    def e2e_decoding(self):
        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/e2e_decode.sh', shell=True)

    def kaldi_decoding(self):
        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/kaldi_decode.sh', shell=True)

    # Button Click Event
    def btnPlayClicked(self):           # Play 버튼 클릭 시 

        self.Display_OriginScript()
        # Play wavFile
        self.recoding_wav()

        # Start Decoding
        self.kaldi_decoding()
        self.Display_KaldiScript()

        self.e2e_decoding()
        self.Display_E2EScript()

    def btnStopClicked(self):           # Stop 버튼 클릭 시
        sd.stop()

    def btnWavClicked(self):            # Wav File 클릭 시
        
        self.canvas.axs[0].cla()

        file_dialog = QFileDialog(self)

        file_dialog.setNameFilters(["Audio files (*.wav)", "All files (*.*)"])
        file_dialog.selectNameFilter("Audio files (*.wav)")
        
        fname = file_dialog.getOpenFileName(self)
        
        self.wavFilePath = fname[0]
        self.wavFileName = self.wavFilePath.split("/")[-1]
        self.FileName_label.setText(self.wavFileName)

        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/feature_extraction.sh', shell=True)

        self.sample_rate, self.data = sio.wavfile.read(self.wavFilePath)

        self.canvas.update_(self.data)


app = QApplication(sys.argv)
dlg = KTDialog()
dlg.show()
app.exec_()
