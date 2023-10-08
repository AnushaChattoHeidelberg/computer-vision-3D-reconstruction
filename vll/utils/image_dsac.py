import torch
import torch.nn.functional as F
import random
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import skimage.io
import skimage.color
import skimage.exposure
from skimage import feature, measure
from skimage.draw import line
from skimage.transform import probabilistic_hough_line
from scipy.spatial import distance

import time
import math

class VanishingPointDSAC:
	'''
	Differentiable RANSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function
		

	def _greyscale_(self,img):
		grey_img = skimage.color.rgb2gray(img)
		return grey_img
	
	def _edgedetection_(self,img):
		edges = feature.canny(img, sigma=3)
		return edges
	'''
	def _linedetection_(self,edges):
		lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
		return lines

	def _accumation_(self,img,edges,lines):
		height, width = img.shape
		accumulation_matrix = np.zeros((height, width), dtype=int)
		for i in range(len(lines)):
			for j in range(i + 1, len(lines)):
				line1 = lines[i]
				line2 = lines[j]
				
				# Calculate the intersection point of two lines
				intersection_x = ((line1[0][0] * line1[1][1] - line1[0][1] * line1[1][0]) * (line2[0][0] - line2[1][0]) -
								(line1[0][0] - line1[1][0]) * (line2[0][0] * line2[1][1] - line2[0][1] * line2[1][0])) / \
								((line1[0][0] - line1[1][0]) * (line2[0][1] - line2[1][1]) -
								(line1[0][1] - line1[1][1]) * (line2[0][0] - line2[1][0]))
				
				intersection_y = ((line1[0][0] * line1[1][1] - line1[0][1] * line1[1][0]) * (line2[0][1] - line2[1][1]) -
								(line1[0][1] - line1[1][1]) * (line2[0][0] * line2[1][1] - line2[0][1] * line2[1][0])) / \
								((line1[0][0] - line1[1][0]) * (line2[0][1] - line2[1][1]) -
								(line1[0][1] - line1[1][1]) * (line2[0][0] - line2[1][0]))
				
				# Check if the intersection point is within the image bounds
				if 0 <= intersection_x < width and 0 <= intersection_y < height:
					# Increment the corresponding cell in the accumulation matrix
					accumulation_matrix[int(intersection_y), int(intersection_x)] += 1
		return accumulation_matrix
	'''
	def calculate_distance_between_line_segments(line1, line2):
    # Define the endpoints of the line segments as (x1, y1) and (x2, y2)
		x1, y1 = line1[0]
		x2, y2 = line1[1]
		x3, y3 = line2[0]
		x4, y4 = line2[1]

    # Calculate the direction vectors of the lines
		line1_direction = np.array([x2 - x1, y2 - y1])
		line2_direction = np.array([x4 - x3, y4 - y3])

    # Calculate the vector between the starting points of the lines
		start_vector = np.array([x3 - x1, y3 - y1])

    # Calculate the cross product of the direction vectors
		cross_product = np.cross(line1_direction, line2_direction)

    # Check if the lines are parallel (cross_product is zero)
		if np.abs(cross_product) < 1e-8:
        # Lines are parallel, return the minimum distance between their endpoints
			distances = [
            np.linalg.norm(start_vector),
            np.linalg.norm(np.array([x4 - x1, y4 - y1])),
            np.linalg.norm(np.array([x3 - x2, y3 - y2])),
            np.linalg.norm(np.array([x4 - x2, y4 - y2]))
        ]
			return min(distances)

    # Calculate the distance between the lines (assuming they are not parallel)
		distance = abs(np.dot(start_vector, cross_product) / np.linalg.norm(cross_product))
		return distance

	def calculate_intersection_between_line_segments(line1, line2):
    # Extract endpoints of the first line segment
		x1, y1 = line1[0]
		x2, y2 = line1[1]

		# Extract endpoints of the second line segment
		x3, y3 = line2[0]
		x4, y4 = line2[1]

		# Calculate the determinant of the coefficient matrix
		determinant = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

		# Check if the lines are parallel (determinant is close to zero)
		if abs(determinant) < 1e-8:
				return None  # Lines are parallel and do not intersect

		# Calculate the intersection point coordinates
		intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / determinant
		intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / determinant

		return (intersection_x, intersection_y)

	def _linedetection_(self,edges):
		contours = measure.find_contours(edges, level=0.8, fully_connected='low')
		min_line_length = 50
		# List to store line segments
		line_segments = []

		# Iterate through the detected contours
		for contour in contours:
    		# Approximate the contour with a polygon (line segments)
			epsilon = 0.02 * measure.perimeter(contour)
			approx_polygon = measure.approximate_polygon(contour, epsilon=epsilon)

    		# Check if the polygon (line segment) is long enough
			if len(approx_polygon) >= 2 and measure.perimeter(contour) >= min_line_length:
        	# Append the line segment to the list
				line_segments.append(approx_polygon)

		distances_between_segments = []
		for i in range(len(line_segments)):
			for j in range(i + 1, len(line_segments)):
				line1 = line_segments[i]
				line2 = line_segments[j]

        # Calculate the distance between line segments (assuming they are not parallel)
		distance = self.calculate_distance_between_line_segments(line1, line2)
		distances_between_segments.append((i, j, distance))  # Store the distance and indices of line segments

		return line_segments, distances_between_segments
	
	def _accumation_(self,img,lines):
		height, width = img.shape
		accumulation_matrix = np.zeros((height, width), dtype=int)
		for i in range(len(lines)):
			for j in range(i + 1, len(lines)):
				line1 = lines[i]
				line2 = lines[j]
				intersect = self.calculate_intersection_between_line_segments(line1, line2)
				# Check if the intersection point is within the image bounds
				if 0 <= intersect.intersection_x < width and 0 <= intersect.intersection_y < height:
					# Increment the corresponding cell in the accumulation matrix
					accumulation_matrix[int(intersect.intersection_y), int(intersect.intersection_x)] += 1
		return accumulation_matrix
		

	
	def _valuecalc__(self,img,edges,lines,distances,accumulation_matrix):
		height, width = img.shape
		#create a distance matrix
		distance_matrix = np.empty((height, width))
		for i in range(width):
			for j in range(height):
				if accumulation_matrix[j, i] > 0:
					distance_between_segment_and_vp = []
					line2 = [(), ()]
					#so I don't have to write another function I decide that the endpoints of line2 are the same
					# Assign coordinates to index 0 (i, j)
					line2[0] = (j, i)
					# Assign coordinates to index 1 (i, j)
					line2[1] = (j, i)
					for k in range(len(lines)):
						distance=self.calculate_distance_between_line_segments(lines[k], line2)
						distance_between_segment_and_vp.append(distance)
					#storing the distance array in the distance matrix for each line in the cell for a potential vanishing point
					distance_matrix[j,i]=distance_between_segment_and_vp
				else: distance_matrix[j,i] = []
		#create a vote matrix
		vote_matrix= np.empty((height, width))
		for i in range(width):
			for j in range(height):
				if distance_matrix[j, i] != []:
					votes = 0
					for k in range(len(lines)):
						#picking out the distance between a line and a point in the accumulation matrix
						distance = distance_matrix[j,i][k]
						#vote function to write vote function here
						vote = 1
						
						votes += vote
					distance_matrix[j,i] = votes
					
		
	
	def _search_(self,img,edges):
		pass

	def _sample_hyp(self,img):
		'''
		Calculate a sample hyp from an image
		'''
		return 0, 0

	def _soft_inlier_count(self, cX, cY, r, x, y):
		'''
		Soft inlier count for a given circle and a given set of points.

		cX -- x of circle center
		cY -- y of circle center
		r -- radius of the line
		x -- vector of x values
		y -- vector of y values
		'''
		return 0, torch.zeros(x.size())

	def _refine_hyp(self, x, y, weights):
		'''
		Refinement by weighted least squares fit.

		x -- vector of x values
		y -- vector of y values
		weights -- vector of weights (1 per point)		
		'''
		return 0, 0, 0
		
	def __call__(self, prediction, labels):
		'''
		Perform robust, differentiable line fitting according to DSAC.

		Returns the expected loss of choosing a good line hypothesis which can be used for backprob.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of parameters (intercept, slope)
		'''

		# working on CPU because of many, small matrices
		prediction = prediction.cpu()

		batch_size = prediction.size(0)

		avg_exp_loss = 0 # expected loss
		avg_top_loss = 0 # loss of best hypothesis

		self.est_parameters = torch.zeros(batch_size, 3) # estimated lines
		self.est_losses = torch.zeros(batch_size) # loss of estimated lines
		self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines

		for b in range(0, batch_size):

			hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
			hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis

			max_score = 0 	# score of best hypothesis

			y = prediction[b, 0] # all y-values of the prediction
			x = prediction[b, 1] # all x.values of the prediction

			for h in range(0, self.hyps):	

				# === step 1: sample hypothesis ===========================
				cX, cY, r, valid = self._sample_hyp(x, y)
				if not valid: continue

				# === step 2: score hypothesis using soft inlier count ====
				score, inliers = self._soft_inlier_count(cX, cY, r, x, y)

				# === step 3: refine hypothesis ===========================
				cX_ref, cY_ref, r_ref = self._refine_hyp(x, y, inliers)

				if r_ref > 0: # check whether refinement was implemented
					cX, cY, r = cX_ref, cY_ref, r_ref

				hyp = torch.zeros([3])
				hyp[0] = cX
				hyp[1] = cY
				hyp[2] = r

				# === step 4: calculate loss of hypothesis ================
				loss = self.loss_function(hyp, labels[b]) 

				# store results
				hyp_losses[h] = loss
				hyp_scores[h] = score

				# keep track of best hypothesis so far
				if score > max_score:
					max_score = score
					self.est_losses[b] = loss
					self.est_parameters[b] = hyp
					self.batch_inliers[b] = inliers

			# === step 5: calculate the expectation ===========================

			#softmax distribution from hypotheses scores			
			hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

			# expectation of loss
			exp_loss = torch.sum(hyp_losses * hyp_scores)
			avg_exp_loss = avg_exp_loss + exp_loss

			# loss of best hypothesis (for evaluation)
			avg_top_loss = avg_top_loss + self.est_losses[b]

		return avg_exp_loss / batch_size, avg_top_loss / batch_size