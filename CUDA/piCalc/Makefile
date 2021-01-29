CC = nvcc
CC_FLAGS = -lm

piCalc: piCalc.cu
	$(CC) $(CC_FLAGS) piCalc.cu -o piCalc -lm 

clean:
	rm -f piCalc
