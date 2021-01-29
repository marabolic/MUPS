# C compiler
CC = mpicc
CC_FLAGS = -O2 -o

nbody : nbody.c 
	$(CC) $(CC_FLAGS) nbody nbody.c -lm 

clean:
	rm -f nbody
