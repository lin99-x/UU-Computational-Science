CC = gcc
LD = gcc
CFLAGS = -Wall -O1 -g -fopenmp
LDLIBS = -lm # -lm is the math library
RM = rm -f
EXECUTABLE = v1_openmp


all: clean v1_openmp.o

execute:
	./$(EXECUTABLE) v1_openmp

v1_openmp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -o $(EXECUTABLE) v1_openmp.c $(LDLIBS)

clean:
	$(RM) $(EXECUTABLE)