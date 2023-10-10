'''	
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
						w1=0.1
						w2=0.1
						ta=0.1
						# Calculate the diagonal length using the Pythagorean theorem
						max_length = math.sqrt(width ** 2 + height ** 2)
						x1,y1=lines[k][0]
						x2,y2=lines[k][1]
						length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
						vote = w1(1-(distance/ta))+w2(length/max_length)
						votes += vote
					distance_matrix[j,i] = votes
	'''	

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