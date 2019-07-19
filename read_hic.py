# import matplotlib.pyplot as plt
import numpy as np
import h5py

#from mirnylib import genome
#from mirnylib import h5dict
# from mirnylib import plotting
#from hiclib import binnedData

import cooler

print 'finish importing'



h5 = h5py.File('../data/chr1.cool', 'r')


file = cooler.Cooler(h5)
length = len(file.matrix())
bias = cooler.ice.iterative_correction(h5)[0:length]

length = len(file.matrix())
mat = file.matrix()[0:length, 0:length]


# apply the balancing weights
mat.data = bias[mat.row] * bias[mat.col] * mat.data


print str(file.matrix()[0:length, 0:length].toarray())
print str(mat.toarray())


print 'finish'

#cooler makebins /net/noble/vol1/data/reference_genomes/mm9/chromosomes 1000000 > bins.100kb.bed











'''
chr_size_list = []


def find_max(filename, bin_size):
    # find the max length of chromosome
    chr_size_file = open(filename, 'r')
    max_length = 0
    cur_sum = 0
    chr_size_list.append(cur_sum)
    for line in chr_size_file:
        line = line[:-1]
        pair = line.split('\t')
        size = int(pair[1]) / bin_size + 1
        cur_sum = cur_sum + size
        chr_size_list.append(cur_sum)
        max_length = max(max_length, size)
    return max_length


# initialize the file and retrieve the matrix from the file
genome_db = genome.Genome('/net/noble/vol1/data/reference_genomes/mm9/chromosomes', readChrms=['#', 'X'])

# Read resolution from the dataset.
raw_heatmap = h5dict.h5dict('~/2016ACCOST/data/1000000.either.hdf5', mode='r') 
resolution = int(raw_heatmap['resolution'])

data = binnedData.binnedData(resolution, genome_db)
data.simpleLoad('~/2016ACCOST/data/1000000.either.hdf5', 'matrix')

matrix = data.dataDict['matrix']



# find the row where the whole row are zeros
zero_list = set()
for x in range(0, len(matrix)):
	if sum(matrix[x]) == 0:
		zero_list.add(x);


# turn the corresponding rows and cols into nan 
# if whole row are zeros		
for x in range(0, len(matrix)):
	if x in zero_list:
		for z in range (0, len(matrix)):
			matrix[x][z] = np.nan
	else: 
		for y in range(0, len(matrix)):
			if y in zero_list:
				matrix[x][y] = np.nan


	
# find max length of chromosomes and update the list 
# keeping record of the length of each chromosome
max = find_max('../data/mouse_chr_sizes.txt', 1000000)	


# construct numpy array with max length + 1
# row 1 - count
# row 2 - sum
# row 3 - sum of square
# each column stores the data of corresponding distance
# last column stores data of 'no distance'
array = np.zeros((3, max + 1))
list_pointer = 1
for x in range(0, len(matrix)):
	if x > chr_size_list[list_pointer]:
		list_pointer = list_pointer + 1
	for y in range(x, len(matrix)):
		if not np.isnan(matrix[x][y]):
			data = matrix[x][y]
			if y >= chr_size_list[list_pointer - 1] and y < chr_size_list[list_pointer]:
				index = y - x
			else:
				index = max
			array[0][index] += 1
			array[1][index] += data
			array[2][index] += data ** 2

	
# calculate the mean and variance
# row 1 - mean
# row 2 - variance
result = np.zeros((2, max + 1))
for x in range(0, max + 1):
    mean = 1.0 * array[1][x] / array[0][x]

    result[0][x] = mean
    result[1][x] = (1.0 * array[2][x] / array[0][x]) - mean ** 2	
	

print str(array)
print '\n\n\n Separator \n\n\n'
print str(result)	
	
'''

	
	
	
	

