import numpy as np

chr_size_file_name = '../data/chr_sizes.tab'
data_file_name = '../data/TROPHOZOITES-XL'
bin_size = 10000
# set to True if running tests
testing = True


def read_file(filename, array_length, bin_size):

    # construct numpy array with max length + 1
    # row 1 - count
    # row 2 - sum
    # row 3 - sum of square
    # each column stores the data of corresponding distance
    # last column stores data of 'no distance'
    array = np.zeros((3, array_length + 1))

    # loop through the array and update corresponding data
    input_file = open(filename, 'r')
    for lines in input_file:
        lines = lines[:-1]
        data = lines.split('\t')
        if data[0] != data[2]:
            index = array_length
        else:
            index = (int(data[3]) - int(data[1])) / bin_size
        array[0][index] += 1
        array[1][index] += int(data[4])
        array[2][index] += int(data[4]) ** 2


    # calculate the mean and variance
    # row 1 - mean
    # row 2 - variance
    result = np.zeros((2, array_length + 1))
    for x in range(0, array_length + 1):
        mean = 1.0 * array[1][x] / array[0][x]

        result[0][x] = mean
        result[1][x] = (1.0 * array[2][x] / array[0][x]) - mean ** 2
    return result


def find_max(filename):
    # find the max length of chromosome
    chr_size_file = open(filename, 'r')
    
    max_length = 0
    for line in chr_size_file:
        line = line[:-1]
        pair = line.split('\t')
        cur_length = pair[1]
        
        if (cur_length.isdigit()):
			cur_length = int(cur_length)
			max_length = max(max_length, cur_length)
	return max_length


def main():
    array_length = find_max(chr_size_file_name)
    result_array = read_file(data_file_name, array_length, bin_size)
    print str(result_array)



if __name__ == "__main__":
    main()


