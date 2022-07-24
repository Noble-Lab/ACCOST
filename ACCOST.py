import sys
import os

def wrapper():
	print("Welcome. Starting ACCOST.")
	
	accost_dir = os.path.dirname(os.path.abspath(__file__))
	fname_bins = sys.argv[2]
	fname_tabs = sys.argv[3]
	
	
	# quickly check if there are multiple chromosomes in the user input
	chr_names = {}
	for line in open(fname_bins):
		words = line.rstrip().split("\t")
		# we found a new chromosome
		if words[0] not in chr_names:
			chr_names[words[0]] = []
			
		chr_names[words[0]].append([words[0], words[1], words[2], words[3], words[4]])


	# handle where the output will be written
	if len(sys.argv) > 6:
		outputs_dir = sys.argv[6]
	else:
		outputs_dir = "output_dir"
	print("Output will be written to %s" % outputs_dir)
	os.makedirs(outputs_dir)
	
	
	# are there additional arguments
	extra_args = ""
	if len(sys.argv) > 7:
		extra_args = " ".join(sys.argv[7:])
	
	
	
	
	# if ACCOST is to be run on a single chromosome behave exactly as before
	if len(chr_names) == 1:
		cmd = "python ACCOST_internal.py %s %s %s %s %s -o %s/ %s" % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], outputs_dir, extra_args)
		os.system(cmd)

		# annotate the p-values
		cmd = "python annotate_pvals.py %s/_ln_pvals.txt %s %s/differential_ln_pvals_expanded.txt" % (outputs_dir, sys.argv[2], outputs_dir)
		os.system(cmd)
		
		print("ACCOST Done!")
		return



	# the user has multiple chromosomes in the input files. Split them.
	print("\nAttention. Multiple chromosomes detected in the input!\n")
	print("By design ACCOST finds differential contacts in a given chromosome between two conditions.")
	print("Will attempt to split the input into individual chromosomes and run on each of them separately.\n")
	
	# create inputs_per_chr directory
	inputs_dir = accost_dir + "/" + "inputs_per_chr/"
	os.makedirs(inputs_dir)


	# for each chromosome write its separate input
	for chr in chr_names:
		fo = open(inputs_dir + chr + "_bins.txt", "w")
		for row in chr_names[chr]:
			fo.write("%s\t%s\t%s\t%s\t%s\n" % (row[0], row[1], row[2], row[3], row[4]))
		fo.close()
			

	# now create separete input.info.tab for each chromosome
	for line_tab in open(fname_tabs):
		words = line_tab.rstrip().split("\t")
		
		# get the counts and biases file names
		counts_file = words[1].replace("/", "_")
		biases_file = words[2].replace("/", "_")
		
		for chr in chr_names:
			fo_tabs = open(inputs_dir + chr + "_info.tab", "a")
			fo_tabs.write("%s\t%s\t%s\n" % (words[0], "inputs_per_chr/" + chr + counts_file, "inputs_per_chr/" + chr + biases_file))
			fo_tabs.close()
		
		# split the biases per chromosome
		for line in open(words[2]):
			biases = line.rstrip().split("\t")
			chr = biases[0]
			fo_biases = open(inputs_dir + chr + biases_file, "a")
			fo_biases.write("%s\t%s\t%s\n" % (biases[0], biases[1], biases[2]))
			fo_biases.close()


		# split the contact counts per chromosome
		for line in open(words[1]):
			counts = line.rstrip().split("\t")
			chr = counts[0]
			
			# we care only about the cis contacts and ignore the trans-contacts
			if chr != counts[2]:
				continue
				
			fo_counts = open(inputs_dir + chr + counts_file, "a")
			fo_counts.write("%s\t%s\t%s\t%s\t%s\n" % (counts[0], counts[1], counts[2], counts[3], counts[4]))
			fo_counts.close()
			

	print("Done splitting the data into individual chromosomes.")
	
	
	# Now, run ACCOST separately on each chromosome
	for chr in chr_names:
		print("Running on chromosome %s\n" % chr)

		cmd = "python ACCOST_internal.py %s %s %s %s %s -o %s/%s %s" % (sys.argv[1], inputs_dir + chr + "_bins.txt", inputs_dir + chr + "_info.tab", sys.argv[4], sys.argv[5], outputs_dir, chr, extra_args)
		os.system(cmd)
		
		
		# annotate the p-values
		cmd = "python annotate_pvals.py %s/%s_ln_pvals.txt %s %s/%s_differential_ln_pvals_expanded.txt" % (outputs_dir, chr, inputs_dir + chr + "_bins.txt", outputs_dir, chr)
		os.system(cmd)
		
		print("Done with chromosome %s\n" % chr)
		
	# remove the inputs_per_chr directory
	cmd = "rm -rf %s" % inputs_dir
	os.system(cmd)
	
	print("ACCOST Done!")
	return





if __name__ == "__main__":
	wrapper()
