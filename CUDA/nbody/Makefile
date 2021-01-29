CC = nvcc
CC_FLAGS = -lm

nbody: nbody.cu
	$(CC) $(CC_FLAGS) nbody.cu -o nbody -lm 

clean:
	rm -f nbody
