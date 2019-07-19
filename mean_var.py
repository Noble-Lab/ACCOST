# import matplotlib.pyplot as plt
import numpy as np


class MeanVariance():
    def __init__(self, chr_file_name, data_file_name, data_format, bin_size):
        self.chr_file_name = chr_file_name
        self.data_file_name = data_file_name
        self.data_format = data_format
        self.bin_size = int(bin_size)
        self.chr_size_list = []

    def get_mean_variance(self):
        max_length = self.find_max()
        if self.data_format == 'TAB':
            count_result = self.tab_mean_variance(max_length)
        else:
            count_result = self.cool_mean_variance(self.data_file_name, max_length)

        # calculate the mean and variance
        # row 1 - mean
        # row 2 - variance
        result = np.zeros((2, max + 1))
        for x in range(0, max + 1):
            mean = 1.0 * array[1][x] / array[0][x]
        result[0][x] = mean
        result[1][x] = (1.0 * array[2][x] / array[0][x]) - mean ** 2
        return result

    def find_max(self):
        # find the max length of chromosome
        chr_size_file = open(self.chr_file_name, 'r')

        max_length = 0
        for line in chr_size_file:
            line = line[:-1]
            pair = line.split('\t')
            cur_length = pair[1]
            if (type(cur_length) == int):
                cur_length = int(cur_length)
                max_length = max(max_length, cur_length)
        return max_length

    chr_size_list = []

    '''
    def find_max(self, filename, bin_size):
        # find the max length of chromosome
        chr_size_file = open(filename, 'r')
        max_length = 0
        cur_sum = 0
        self.chr_size_list.append(cur_sum)
        for line in chr_size_file:
            line = line[:-1]
            pair = line.split('\t')
            size = int(pair[1]) / bin_size + 1
            cur_sum = cur_sum + size
            self.chr_size_list.append(cur_sum)
            max_length = max(max_length, size)
        return max_length
    '''

    def tab_mean_variance(self, array_length):
        array_length = int(array_length)

        # construct numpy array with max length + 1
        # row 1 - count
        # row 2 - sum
        # row 3 - sum of square
        # each column stores the data of corresponding distance
        # last column stores data of 'no distance'

        array = np.zeros((3, array_length + 1))

        # loop through the array and update corresponding data
        input_file = open(self.data_file_name, 'r')
        for lines in input_file:
            lines = lines[:-1]
            data = lines.split('\t')
            if data[0] != data[2]:
                index = array_length
            else:
                index = (int(data[3]) - int(data[1])) / self.bin_size
            contact_data = int(data[4])
            if not np.isnan(contact_data):
                array[0][index] += 1
            	array[1][index] += contact_data
            	array[2][index] += contact_data ** 2


	def bias_array(filename, array_length):
		bias_file = open(filename, 'r')
		bias_array = []
		for lines in bias_file:
			bias = float(lines)
			bias_array.append(bias)		
		array_length = int(array_length)
		result_array = np.zeros((1, array_length + 1))
		for i in range(length(result_array[0])):
			bias_total = 0
			for j in range(length(bias_array)):
				k = j + i
				if (k < length(bias_array)):
					bias_total += (bias_array[j] * bias_array[k])
			result_array[i] = bias_total


    def cool_mean_variance(self, filename, array_length):
        from mirnylib import genome
        from mirnylib import h5dict
        from hiclib import binnedData

        # initialize the file and retrieve the matrix from the file
        genome_db = genome.Genome('/net/noble/vol1/data/reference_genomes/mm9/chromosomes', readChrms=['#', 'X'])

        # Read resolution from the dataset.
        raw_heatmap = h5dict.h5dict(filename, mode='r')
        resolution = int(raw_heatmap['resolution'])

        data = binnedData.binnedData(resolution, genome_db)
        data.simpleLoad(filename, 'matrix')

        matrix = data.dataDict['matrix']

        # construct numpy array with max length + 1
        # row 1 - count
        # row 2 - sum
        # row 3 - sum of square
        # each column stores the data of corresponding distance
        # last column stores data of 'no distance'
        array = np.zeros((3, max + 1))
        list_pointer = 1
        for x in range(0, len(matrix)):
            if x > self.chr_size_list[list_pointer]:
                list_pointer = list_pointer + 1
            for y in range(x, len(matrix)):
                if not np.isnan(matrix[x][y]):
                    data = matrix[x][y]
                    if y >= self.chr_size_list[list_pointer - 1] and y < self.chr_size_list[list_pointer]:
                        index = y - x
                    else:
                        index = max
                    array[0][index] += 1
                    array[1][index] += data
                    array[2][index] += data ** 2




