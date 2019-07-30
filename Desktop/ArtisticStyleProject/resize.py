from skimage.transform import rescale, resize, downscale_local_mean
import imageio
import tensorflow as tf
import numpy as np
import scipy
import scipy.io
import scipy.misc
import cv2
from PIL import Image

def main():
	#file_name = 'tubingen.jpg'
	#file_name = 'samford-sign.jpg'
	file_name = 'hoovertowernight.jpg'
	pic = imageio.imread(file_name)
	print(pic.shape)

	x = 1.25
	#pic_crop = pic[:, 120:760, :]
	pic_crop = pic
	pic_resized = resize(pic_crop, (pic_crop.shape[0]/x, pic_crop.shape[1]/x))
	#n = 1
	#pic_resized = resize(pic, (pic.shape[0]/n, pic.shape[1]/n))
	print(pic_resized.shape)

	scipy.misc.imsave('hoovertowernight_0.jpg', pic_resized)

if __name__ == '__main__':
	main()
