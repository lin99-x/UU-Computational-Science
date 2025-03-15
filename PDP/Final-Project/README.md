# PDP_Project

To run this code, you can just follow the instructions below:

1. Use "make" command to build the executable file, the executable file named "simulator" should be build up
2. Run "mpirun -np <number of process> ./simulator <local size of experiments> <outputfile name (should be a .txt file)>
   For example, you can run this command "mpirun -np 8 ./simulator 10000 test.txt" to start a simulation with n = 10000 runs on 8 processes.
3. Use "make clean" to clean all the .o files and executable file.

After the program finished, you can see the output file in the path you set to. Also you can see in the terminal it will print out the time spent to finifsh the program. 

The output file is designed in this structure: first comes the bin intervel and then followed by the frequency of that bin.

In the tar file I submitted, it contains three result files for N = 1100000, N = 1500000, and N = 1800000 respectively. 