import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import cv2
import os, sys

sys.path.insert(0, os.getcwd())

from model.PRNet import PRNet


def pre_img(im):
	means = [0.4052, 0.4042, 0.3605]
	stds = [0.2326, 0.2150, 0.1924]
	im = cv2.resize(im, (768, 256))
	im = im.astype(np.float32)
	im = im / 255.0
	im = im - means
	im = im / stds
	im = np.transpose(im, (2, 0, 1))
	im = im.astype(np.float32)
	im = torch.from_numpy(im)
	im = torch.unsqueeze(im, dim=0)
	return im


def demo(thresh=0.3):
	image_path = './demo/images'
	save_path = './demo/vis_results'


	model_path = './prnet.pth'
	net = PRNet('resnet18').cuda()
	net = nn.DataParallel(net)
	net.load_state_dict(torch.load(model_path), strict=True)
	net.eval()

	NEW_IMG_SIZE = [256.0, 768.0]
	IMG_SIZE = [590.0, 1640.0]
	image_scale_ratio = [float(NEW_IMG_SIZE[0]) / float(IMG_SIZE[0]),
	                     float(NEW_IMG_SIZE[1]) / float(IMG_SIZE[1])]

	down_sample = 8
	height_thresh = 16

	with torch.no_grad():
		for image_name in os.listdir(image_path):
			image = cv2.imread(os.path.join(image_path, image_name))
			paint_image = image.copy()
			image = pre_img(image).cuda()
			initial_confidence, poly_coefficients, height_end = net(image)
			initial_confidence = initial_confidence.cpu().numpy()
			poly_coefficients = poly_coefficients.cpu().numpy()
			height_end = height_end.cpu().numpy()

			for i in range(1, initial_confidence.shape[2] - 1):
				for j in range(1, initial_confidence.shape[3] - 1):

					if initial_confidence[0, 1, i, j] > thresh and \
							initial_confidence[0, 1, i, j] == np.max(
						initial_confidence[0, 1, i - 1:i + 2, j - 1:j + 2]):

						x_start = int((j - 1 + 0.5) * down_sample)
						y_start = int((i - 1 + 0.5) * down_sample)
						poly_coefficient = poly_coefficients[0, :, i - 1, j - 1]
						lane_height = [height_end[0, 0, i - 1, j - 1] * 256.0]
						for height in range(256, 0, -1):
							if height > y_start:
								continue
							if height < np.mean(lane_height):
								break
							delta_y = (height - y_start) / NEW_IMG_SIZE[0]
							delta_x = np.polyval(poly_coefficient, delta_y)
							real_y = height
							real_x = x_start + delta_x * NEW_IMG_SIZE[1]
							paint_x = int(real_x * IMG_SIZE[1] / NEW_IMG_SIZE[1])
							paint_y = int(real_y * IMG_SIZE[0] / NEW_IMG_SIZE[0])
							cv2.circle(paint_image, (paint_x, paint_y), 3, [0, 255, 0])
							if (y_start - height) % height_thresh == 0:

								i_current = round(float(real_y) / float(down_sample) - 0.5)
								j_current = round(float(real_x) / float(down_sample) - 0.5)
								i_current = np.clip(i_current, 0, 31)
								j_current = np.clip(j_current, 0, 95)
								x_start = (j_current + 0.5) * down_sample
								y_start = (i_current + 0.5) * down_sample
								poly_coefficient = poly_coefficients[0, :, i_current, j_current]
								lane_height.append(height_end[0, 0, i_current, j_current] * 256.0)
			cv2.imwrite(os.path.join(save_path, image_name), paint_image)
demo()

