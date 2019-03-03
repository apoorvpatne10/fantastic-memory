#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script for forming an image cluster
import os
import cv2

tdir = '/home/apoorv/Work/TE_seminar/CNN-lipr/CNN-for-visual-speech-recognition/'
os.chdir('dataset')
name_list = []
for f in os.listdir(): # F01
	os.chdir(f)
	print(os.getcwd())
	for l in os.listdir(): # phrases/words
		os.chdir(l)
		for w in os.listdir(): # 01, 02, 03..
			os.chdir(w)
			for p in os.listdir(): # 01, 02, 03
				os.chdir(p)
				seq = np.zeros((224,224))
				orig = len(os.listdir())
				orig_image = []
				for i in os.listdir():
				    img = cv2.imread(i, 0)
				    img = cv2.resize(img, (32,32), interpolation = cv2.INTER_LINEAR)
				    orig_image.append(img)
				x_limit = 32
				y_limit = 32
				iterator_x = 0
				iterator_y = 0
				for i in range(0,49):
				    seq[iterator_y:iterator_y+y_limit, iterator_x:iterator_x+x_limit] = orig_image[int((i*orig)/49)] # movement along y-axis
				    iterator_x = iterator_x + x_limit
				    if iterator_x == 224:
				        iterator_x = 0
				        iterator_y = iterator_y + y_limit
				name = tdir + 'processed_data/' + f + '-' + l + '-' + w + '-' + p +'.jpg'
				cv2.imwrite(name, seq)
				os.chdir('..')
			os.chdir('..')
		os.chdir('..')
	os.chdir('..')
