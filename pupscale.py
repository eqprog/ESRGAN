import upscale as ups
import logging
from pathlib import Path
from rich.logging import RichHandler
import sys
import os
import glob
from PySide6 import QtCore, QtGui, QtWidgets
import shutil
import numpy as np
import cv2
import glob
import time

"""alpha_options = ups.AlphaOptions("swapping")
seamless_options = ups.SeamlessOptions("replicate")

model_str = '4x_lolly_df2k.pth'
input_path = Path('G:/upscaling utils/ESRGAN/tests')
output_path = Path('G:/upscaling utils/ESRGAN/results')
log = logging.getLogger("rich")

upscale = ups.Upscale(model_str, input_path, output_path, False, True, seamless_options, False, False, 0, True, False, False, 0.1, 0.1, 
	2, log)

upscale.run()
"""

class MainWindow(QtWidgets.QMainWindow):
    #Load Images and display them
	def __init__(self):
		super(MainWindow, self).__init__()
		self.model_str = None
		self.model_str_alpha = None
		self.input_path = None
		self.output_path = None
		self.log = logging.getLogger("rich")
		self.seamless_options = ups.SeamlessOptions("replicate")
		self.layout1 = QtWidgets.QVBoxLayout()
		self.layout2 = QtWidgets.QHBoxLayout()
		self.layout3 = QtWidgets.QVBoxLayout()
		self.widget = QtWidgets.QWidget()
		self.widget.setLayout(self.layout1)
		self.tabs = QtWidgets.QTabWidget()
		self.tabs.addTab(self.batchTabUI(), "Batch")
		self.tabs.addTab(self.advancedTabUI(), "Advanced")
		self.layout1.addWidget(self.tabs)
		self.setCentralWidget(self.widget)

		#self.createButtons()
		#self.widget = QtWidgets.QWidget()
		#self.widget.setLayout(self.layout2)
		#self.setCentralWidget(self.widget)

	def batchTabUI(self):
		batchTab = QtWidgets.QWidget()
		batchlayout = QtWidgets.QVBoxLayout()
		layout1 = QtWidgets.QVBoxLayout()
		layout2 = QtWidgets.QHBoxLayout()
		layout3 = QtWidgets.QVBoxLayout()
		layout1.addStretch()
		
		layout2.addStretch(1)
		layout3.addLayout(self.createToggles())
		layout3.addLayout(self.createButtons())
		layout2.addLayout(layout3)
		layout2.addStretch(1)
		layout1.addLayout(layout2)
		batchTab.setLayout(layout1)
		return batchTab
	
	def advancedTabUI(self):
		batchTab = QtWidgets.QWidget()
		batchlayout = QtWidgets.QVBoxLayout()
		layout1 = QtWidgets.QVBoxLayout()
		layout2 = QtWidgets.QHBoxLayout()
		layout3 = QtWidgets.QVBoxLayout()
		layout1.addStretch()
		
		layout2.addStretch(1)
		layout3.addLayout(self.createTogglesAdvanced())
		layout3.addLayout(self.createButtonsAdvanced())
		layout2.addLayout(layout3)
		layout2.addStretch(1)
		layout1.addLayout(layout2)
		batchTab.setLayout(layout1)
		return batchTab
	
	def open(self):
		self.input_path = Path(QtWidgets.QFileDialog.getExistingDirectory(self, str("Open input folder"), '.', 
			QtWidgets.QFileDialog.ShowDirsOnly))
	
	def outDirectory(self):
		self.output_path = Path(QtWidgets.QFileDialog.getExistingDirectory(self, str("Open input folder"), '.', 
			QtWidgets.QFileDialog.ShowDirsOnly))
	def upscale(self):
		"""def __init__(self, model: str, input: Path, output: Path,
        				reverse: bool, skip_existing: bool, seamless: SeamlessOptions,
        				cpu: bool, fp16: bool, device_id: int, cache_max_split_depth: bool,
        				binary_alpha: bool, ternary_alpha: bool, alpha_threshold: float,
        				alpha_boundary_offset: float, alpha_mode: AlphaOptions, log: logging.Logger)"""
		upscale = ups.Upscale(self.model_str, self.input_path, self.output_path, self.toggleReverse.isChecked(), True, 
			self.seamless_options, self.toggleCPU.isChecked(), False, 0, self.toggleCPU.isChecked(), False, False,
			0.1, 0.1, 2, self.log)
		upscale.run()
	
	def upscaleAdvanced(self):
		if self.toggleAlphaSplit.isChecked() == True:
			self.alpha_path = Path(os.path.join(self.input_path, 'alpha'))
			self.alpha_out = Path(os.path.join(self.output_path, "alpha"))
			print(self.alpha_path)
			try:
				os.makedirs(self.alpha_path)
				os.makedirs(self.alpha_out)
			except:
				print("alpha directories exists")
			alphas = self.alphaSplit(self.alpha_path)
		#upscale = ups.Upscale(self.model_str, self.input_path, self.output_path, self.toggleReverse.isChecked(), True, self.seamless_options, self.toggleCPU.isChecked(), False, 0, self.toggleCPU.isChecked(), False, False, 0.1, 0.1, 2, self.log)
		#upscale images w/o alpha
		#upscale.run()
		
		#upscale images' alpha channel separately
			if len(alphas) > 0:

				self.alphaSave(alphas)
				print("upscaling alphas")

				alpha_upscale = ups.Upscale(self.model_str_alpha, self.alpha_path, self.alpha_out, self.toggleReverse.isChecked(), True, self.seamless_options, self.toggleCPU.isChecked(), False, 0, self.toggleCPU.isChecked(), False, False, 0.1, 0.1, 2, self.log)

				alpha_upscale.run()

				upscale = ups.Upscale(self.model_str, self.input_path, self.output_path, self.toggleReverse.isChecked(), True, 
					self.seamless_options, self.toggleCPU.isChecked(), False, 0, self.toggleCPU.isChecked(), False, False,
					0.1, 0.1, 2, self.log)
				upscale.run()
				if self.toggleAlphaJoin.isChecked() == True:
					self.alphaJoin(self.output_path, self.alpha_out)
					self.alphaJoin(self.input_path, self.alpha_path)
			else:
				upscale = ups.Upscale(self.model_str, self.input_path, self.output_path, self.toggleReverse.isChecked(), True, 
					self.seamless_options, self.toggleCPU.isChecked(), False, 0, self.toggleCPU.isChecked(), False, False,
					0.1, 0.1, 2, self.log)
				upscale.run()
					

	def selectModel(self):
		file = QtWidgets.QFileDialog.getOpenFileName(self, str("Open model"), "./models")
		self.model_str = file[0]
	
	def selectModelAlpha(self):
		file = QtWidgets.QFileDialog.getOpenFileName(self, str("Open model"), "./models")
		self.model_str_alpha = file[0]
	def selectModelAlphaState(self):
		self.btnSelModelAlpha.setEnabled(self.toggleAlphaSplit.isChecked())
		self.toggleAlphaJoin.setEnabled(self.toggleAlphaSplit.isChecked())
	def createToggles(self):
		layout = QtWidgets.QHBoxLayout()
		self.toggleCPU = QtWidgets.QCheckBox("&CPU Upscaling", self)
		self.toggleCache = QtWidgets.QCheckBox("Cache M&ax Split Depth", self)
		self.toggleReverse = QtWidgets.QCheckBox("Re&verse")
		layout.addWidget(self.toggleCPU)
		layout.addWidget(self.toggleCache)
		layout.addWidget(self.toggleReverse)
		return layout

	def createTogglesAdvanced(self):
		layout1 = QtWidgets.QVBoxLayout()
		layout2 = QtWidgets.QHBoxLayout()
		layout3 = QtWidgets.QHBoxLayout()

		self.toggleCPU = QtWidgets.QCheckBox("&CPU Upscaling", self)
		self.toggleCache = QtWidgets.QCheckBox("Cache M&ax Split Depth", self)
		self.toggleReverse = QtWidgets.QCheckBox("Re&verse")
		self.toggleAlphaSplit = QtWidgets.QCheckBox("Upscale transparencies separately")
		self.toggleAlphaSplit.toggled.connect(lambda:self.selectModelAlphaState())
		self.toggleAlphaJoin = QtWidgets.QCheckBox("Rejoin transparencies")
		self.toggleAlphaJoin.setEnabled(False)

		layout2.addWidget(self.toggleCPU)
		layout2.addWidget(self.toggleCache)
		layout2.addWidget(self.toggleReverse)
		layout3.addWidget(self.toggleAlphaSplit)
		layout3.addWidget(self.toggleAlphaJoin)
		layout1.addStretch(1)
		layout1.addLayout(layout2)
		layout1.addLayout(layout3)
		return layout1


	def createButtons(self):
		layout1 = QtWidgets.QHBoxLayout()
		layout2 = QtWidgets.QHBoxLayout()
		btnOpen = QtWidgets.QPushButton("Open Directory")
		btnOpen.clicked.connect(self.open)
		btnOutput = QtWidgets.QPushButton("Output Directory")
		btnOutput.clicked.connect(self.outDirectory)
		btnSelModel = QtWidgets.QPushButton("Select model")
		btnSelModel.clicked.connect(self.selectModel)
		btnUpscale = QtWidgets.QPushButton("Upscale Images")
		btnUpscale.clicked.connect(self.upscale)
	
		for btn in [btnOpen, btnOutput, btnSelModel, btnUpscale]:
			layout2.addWidget(btn)
		layout1.addStretch(1)
		layout1.addLayout(layout2)
		return layout1

	def createButtonsAdvanced(self):
		layout1 = QtWidgets.QHBoxLayout()
		layout2 = QtWidgets.QHBoxLayout()
		btnOpen = QtWidgets.QPushButton("Open Directory")
		btnOpen.clicked.connect(self.open)
		btnOutput = QtWidgets.QPushButton("Output Directory")
		btnOutput.clicked.connect(self.outDirectory)
		btnSelModel = QtWidgets.QPushButton("Select model")
		btnSelModel.clicked.connect(self.selectModel)
		self.btnSelModelAlpha = QtWidgets.QPushButton("Transparency model")
		self.btnSelModelAlpha.clicked.connect(self.selectModelAlpha)
		self.btnSelModelAlpha.setEnabled(False)
		btnUpscale = QtWidgets.QPushButton("Upscale Images")
		btnUpscale.clicked.connect(self.upscaleAdvanced)
	
		for btn in [btnOpen, btnOutput, btnSelModel, self.btnSelModelAlpha, btnUpscale]:
			layout2.addWidget(btn)
		layout1.addStretch(1)
		layout1.addLayout(layout2)
		return layout1		

	def alphaSplit(self, path):
		alphas = []

		for file in glob.glob(f"{self.input_path}/*.png"):
			image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
			if image.ndim > 2 and image.shape[2] == 4:
				print(image.shape[2])
				blu, grn, red, alp = cv2.split(image)
				stripped = cv2.merge((blu, grn, red))
				
				#cv2.imwrite(file, stripped)
				

				alphas.append([file, alp])
				cv2.imwrite(file, stripped)
				print(file)

			else:
				print(f'{file} contains no transparencies!')
		time.sleep(3)
		return alphas
	def alphaSave(self, alphas):
		for i, j in alphas:
			base, file = os.path.split(i)
			base = os.path.join(base, "alpha")
			base = os.path.join(base, file)
			print(base)
			cv2.imwrite(base, j)
		return alphas
	def alphaJoin(self, output_path, alpha_path):
		print("Joining alphas:")
		originals = []
		alphas = []

		for image in glob.glob(f'{alpha_path}/*.png'):
			name = os.path.join(output_path, os.path.basename(image))
			print(name)
			original = cv2.imread(name)
			alpha = cv2.imread(image, 0)
			blu, grn, red = cv2.split(original)
			result = cv2.merge((blu, grn, red, alpha))
			cv2.imwrite(name, result)
			os.remove(image)


if (__name__ == '__main__'):

    app = QtWidgets.QApplication(sys.argv)

    mainWindow = MainWindow()

    mainWindow.show()

    sys.exit(app.exec_())