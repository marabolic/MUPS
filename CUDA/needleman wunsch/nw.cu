#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#define MASTER 0

void runTest_seq( int dims, int pen);
void runTest_par(int dims, int pen);
__device__ int maximum_device( int a, int b, int c){
	int k;
	if( a <= b )
		k = b;
	else 
	k = a;
	if( k <=c )
		return(c);
	else
		return(k);
}

__host__ int maximum_host( int a, int b, int c){
	int k;
	if( a <= b )
		k = b;
	else 
	k = a;
	if( k <=c )
		return(c);
	else
		return(k);
}

int blosum62[24][24] = {
	{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
	{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
	{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
	{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
	{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
	{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
	{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
	{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
	{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
	{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};


int main( int argc, char** argv) {

	cudaEvent_t start = cudaEvent_t(), stop = cudaEvent_t();
	
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("\n\n 2048 10");
	printf("\nSEQUENTIAL\n");
	cudaEventRecord(start, 0);
	runTest_seq(2048, 10); // seq 2048
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);

	printf("\nPARALLEL\n");
	cudaEventRecord(start, 0);
	
	runTest_par(2048, 10); // par 2048
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);
	// TODO TEST

	printf("\n\n 6144 10");
	printf("\nSEQUENTIAL\n");
	cudaEventRecord(start, 0);
	runTest_seq(6144, 10); 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);

	printf("\nPARALLEL\n");
	cudaEventRecord(start, 0);
	runTest_par(6144, 10); 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);
	// TODO TEST

	
	printf("\n\n 16384 10");
	printf("\nSEQUENTIAL\n");
	cudaEventRecord(start, 0);
	runTest_seq(16384, 10);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);

	printf("\nPARALLEL\n");
	cudaEventRecord(start, 0);
	runTest_par(16384, 10);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);
	// TODO TEST


	printf("\n\n 22528 10");
	printf("\nSEQUENTIAL\n");
	cudaEventRecord(start, 0);
	runTest_seq(22528, 10);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n",elapsed);

	printf("\nPARALLEL\n"); 
	cudaEventRecord(start, 0);
	runTest_par(22528, 10);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The process took: %f\n", elapsed);
	// TODO TEST

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	

	return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	//exit(1);
}

__host__ void runTest_seq(int dims, int pen) {
	int max_rows, max_cols, penalty,idx, index;
	int *input_itemsets, *output_itemsets, *referrence;
		
	max_cols = max_rows = dims;
	penalty = pen;

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
	input_itemsets = (int *)calloc( max_rows * max_cols, sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	
	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

	//srand ( time(NULL) );



	srand(1);
	for( int i=1; i< max_rows ; i++){     
	   input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
	for( int j=1; j< max_cols ; j++){    
	   input_itemsets[j] = rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

	for( int i = 1; i< max_rows ; i++)
	   input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
	   input_itemsets[j] = -j * penalty;

	for( int i = 0 ; i < max_cols-2 ; i++){
		for( idx = 0 ; idx <= i ; idx++){
		 index = (idx + 1) * max_cols + (i + 1 - idx);
		 input_itemsets[index]= maximum_host( input_itemsets[index-1-max_cols]+ referrence[index], 
										 input_itemsets[index-1]         - penalty, 
										 input_itemsets[index-max_cols]  - penalty);

		}
	}
 	for( int i = max_cols - 4 ; i >= 0 ; i--){
		for( idx = 0 ; idx <= i ; idx++){
		  index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;
		  input_itemsets[index]= maximum_host( input_itemsets[index-1-max_cols]+ referrence[index], 
										  input_itemsets[index-1]         - penalty, 
										  input_itemsets[index-max_cols]  - penalty);
		}

	}

	#define TRACEBACK
	#ifdef TRACEBACK
	
	FILE *fpo = fopen("result_seq.txt","w");
	fprintf(fpo, "print traceback value:\n");
	
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]);
		if ( i == 0 && j == 0 )
		   break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
			w  = input_itemsets[ i * max_cols + j - 1 ];
			n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
			nw = n = LIMIT;
			w  = input_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
			nw = w = LIMIT;
			n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum_host(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
			traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

		else if(traceback == w )
		{j--; continue;}

		else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

	#endif

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

}

__global__ void top_left(int max_cols, int i, int penalty, int * referrence_gpu, int * input_itemsets){
	int idx = blockDim.x * blockIdx.x + threadIdx.x, index;

	if (idx <= i){
		index = (idx + 1) * max_cols + (i + 1 - idx);
		input_itemsets[index]= maximum_device( input_itemsets[index-1-max_cols] + referrence_gpu[index], 
										input_itemsets[index-1]         - penalty, 
										input_itemsets[index-max_cols]  - penalty);
	}
}

__global__ void bottom_right(int max_cols, int i, int penalty, int * referrence_gpu, int * input_itemsets){
	int idx = blockDim.x * blockIdx.x + threadIdx.x, index;

	if (idx <= i){
		index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;
		input_itemsets[index]= maximum_device( input_itemsets[index-1-max_cols]+ referrence_gpu[index], 
										input_itemsets[index-1]         - penalty, 
										input_itemsets[index-max_cols]  - penalty);
	}
}


void runTest_par(int dims, int pen) 
{
	int max_rows, max_cols, penalty;
	int *input_itemsets, *output_itemsets, *referrence;
	int * input_itemsets_gpu, * referrence_gpu;
		
	max_cols = max_rows = dims;
	penalty = pen;
	

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int*)malloc(max_rows * max_cols * sizeof(int));
	input_itemsets = (int*)calloc( max_rows * max_cols, sizeof(int));
	output_itemsets = (int*)malloc(max_rows * max_cols * sizeof(int));

	//srand ( time(NULL) );


	srand(1);
	for( int i=1; i< max_rows ; i++){     
	   input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
	for( int j=1; j< max_cols ; j++){    
	   input_itemsets[j] = rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

	for( int i = 1; i< max_rows ; i++)
	   input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
	   input_itemsets[j] = -j * penalty;

 
	cudaMalloc(&input_itemsets_gpu, max_rows * max_cols * sizeof(int) );
	cudaMalloc(&referrence_gpu, max_rows * max_cols * sizeof(int) );
	cudaMemcpy(input_itemsets_gpu, input_itemsets, max_rows * max_cols * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(referrence_gpu, referrence, max_rows * max_cols * sizeof(int), cudaMemcpyHostToDevice);
	for( int i = 0 ; i < max_cols-2 ; i++){
		top_left<<<ceil((i+1.0f)/1024), 1024>>>(max_cols, i, penalty, referrence_gpu, input_itemsets_gpu);
	}

 	for( int i = max_cols - 4 ; i >= 0 ; i--){
		bottom_right<<<ceil((i+1.0f)/1024), 1024>>>(max_cols, i, penalty, referrence_gpu, input_itemsets_gpu);
	}
	cudaMemcpy(input_itemsets, input_itemsets_gpu, max_rows * max_cols * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(referrence, referrence_gpu, max_rows * max_cols * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(input_itemsets_gpu);
	cudaFree(referrence_gpu);

#define TRACEBACK
#ifdef TRACEBACK
	
	FILE *fpo = fopen("result_par.txt","w");
	fprintf(fpo, "print traceback value:\n");
	
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]);
		if ( i == 0 && j == 0 )
		   break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
			w  = input_itemsets[ i * max_cols + j - 1 ];
			n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
			nw = n = LIMIT;
			w  = input_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
			nw = w = LIMIT;
			n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum_host(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
			traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

		else if(traceback == w )
		{j--; continue;}

		else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

#endif

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

}

