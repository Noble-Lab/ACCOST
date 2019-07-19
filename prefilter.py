

def prefilter(matrix1, matrix2, threshole = 10):
	# find the row where the whole row are zeros
	zero_list = set()
	for x in range(0, len(matrix1)):
		if sum(matrix[x]) <= threshold:
			zero_list.add(x);
	
	for x in range(0, len(matrix2)):
		if sum(matrix[x]) <= threshold:
			zero_list.add(x);
	
	# matrix 1
	# turn the corresponding rows and cols into nan 
	# if whole row are zeros		
	for x in range(0, len(matrix1)):
		if x in zero_list:
			for z in range (0, len(matrix1)):
				matrix1[x][z] = np.nan
		else: 
			for y in range(0, len(matrix1)):
				if y in zero_list:
					matrix1[x][y] = np.nan
	
	# matrix 2
	# turn the corresponding rows and cols into nan 
	# if whole row are zeros
	for x in range(0, len(matrix2)):
		if x in zero_list:
			for z in range (0, len(matrix2)):
				matrix2[x][z] = np.nan
		else: 
			for y in range(0, len(matrix2)):
				if y in zero_list:
					matrix2[x][y] = np.nan