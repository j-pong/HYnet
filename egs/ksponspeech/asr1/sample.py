from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore

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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np
import wave

import KT_main

# constants
CHUNK = 1024             # samples per frame

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

    # Reload Script
    def Display_OriginScript(self):
        origin_file = open("data/test/texts/test_5.txt", 'r')
        line = origin_file.readline()

        self.OriginScript = line
        self.Original_Script.setText(self.OriginScript)

        self.Original_Script.setAlignment(QtCore.Qt.AlignCenter)
        
    def Display_KaldiScript(self):
        kaldi_file = open("KALDI_RESULT", "r")
        line = kaldi_file.readline()
        results = line.split('/')

        self.KaldiScript = results[0]
        self.KaldiWER = results[1]
        self.KaldiCER = results[2]

        self.Kaldi_Script.setText(self.KaldiScript)
        self.Kaldi_WER.setText(self.KaldiWER)
        self.Kaldi_CER.setText(self.KaldiCER)

        self.Kaldi_WER.setAlignment(QtCore.Qt.AlignCenter)
        self.Kaldi_CER.setAlignment(QtCore.Qt.AlignCenter)

    def Display_E2EScript(self):
        e2e_file = open("E2E_RESULT", "r")
        line = e2e_file.readline()
        results = line.split('/')

        self.E2EScript = results[0]
        self.E2EWER = results[1]
        self.E2ECER = results[2]

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
        sample_rate, data = sio.wavfile.read(self.wavFilePath)

        times = np.arange(len(data))/float(sample_rate)

        sd.play(data,sample_rate)

    def e2e_decoding(self):
        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/e2e_decode.sh', shell=True)

    def kaldi_decoding(self):
        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/kaldi_decode.sh', shell=True)

    # Button Click Event
    def btnPlayClicked(self):           # Play 버튼 클릭 시 

        # Play wavFile
        self.recoding_wav()

        # Start Decoding
        self.kaldi_decoding()
        self.e2e_decoding()

        # Print Script
        self.Display_OriginScript()
        self.Display_KaldiScript()
        self.Display_E2EScript()

    def btnStopClicked(self):           # Stop 버튼 클릭 시
        sd.stop()

    def btnWavClicked(self):            # Wav File 클릭 시
        file_dialog = QFileDialog(self)

        file_dialog.setNameFilters(["Audio files (*.wav)", "All files (*.*)"])
        file_dialog.selectNameFilter("Audio files (*.wav)")
        
        fname = file_dialog.getOpenFileName(self)
        
        self.wavFilePath = fname[0]
        self.wavFileName = self.wavFilePath.split("/")[-1]
        self.FileName_label.setText(self.wavFileName)

        subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/feature_extraction.sh', shell=True)

        # Display Waveform
        fig = plt.Figure()
        ax = fig.add_subplot(1,1,1)

        sample_rate, data = sio.wavfile.read(self.wavFilePath)

        times = np.arange(len(data))/float(sample_rate)
        
        sd.play(data,sample_rate)

        ax.fill_between(times, data)
        
        canvas = FigureCanvasQTAgg(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        self.Wav_Widget.setLayout(layout)

        

app = QApplication(sys.argv)
dlg = KTDialog()
dlg.show()
app.exec_()