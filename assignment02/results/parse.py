import os
import sys

def writeToCSV():
	in_file = input("Please input the name of the file to be parsed: ")
	if not os.path.exists(in_file):
		print("That file does not exist!")
		exit()
	out_file = input("Please input the name of the output file: ")
	# By default, it'll be the name of the input file + .csv
	if len(out_file) == 0:
		out_file = in_file[:-4] + '.csv'
	# Checks to make sure you want to overwrite the file
	while in_file == out_file:
		confirm = input("Are you sure you want to overwrite the previous file?")
		if confirm.upper()[0] != 'Y':
			out_file = input("Please input the name of the output file: ")
		else:
			break
	num_times = input("Please enter the number of recordings made for each optimizer: ")


	with open(in_file, 'r') as f, open(out_file, 'w') as w:
		lines = f.readlines()
		curr_line_no = 0
		for i in range(4):
			w.write(lines[curr_line_no].strip() + "\n")
			curr_line_no += 1
			w.write("ITERATIONS,BEST PERFORMANCE,TIME\n")
			for i in range(int(num_times)):
				num_iter = lines[curr_line_no].replace("-","").split()[0]
				best_score = lines[curr_line_no + 1].split()[1]
				time = lines[curr_line_no + 2].split()[2]
				w.write(num_iter + "," + best_score + "," + time + "\n")
				curr_line_no += 3
			w.write("\n")
			curr_line_no += 1


if __name__=='__main__':
	writeToCSV()