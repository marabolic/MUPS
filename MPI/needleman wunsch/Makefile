# C compiler
CC = mpicc
CC_FLAGS = -O2 -o

needle: needle.c
	$(CC) $(CC_FLAGS) needle needle.c -lm

clean:
	rm -f needle
