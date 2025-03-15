###############################################################################
# Makefile for project.
###############################################################################

CC = mpicc
CFLAGS = -std=c99 -g -O3
LIBS = -lm

BIN = mc

all: $(BIN)

mc: simulate.o prop.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

simulate.o: simulate.c prop.h
	$(CC) $(CFLAGS) -c $<

prop.o: prop.c prop.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o $(BIN)