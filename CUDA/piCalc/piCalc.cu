#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define MASTER 0

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   exit(1);
}


__host__
double sequential(long long n) {
   long long i;
   double factor = 1;
   double sum = 0.0;

   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0;
      sum += factor/(2*i+1);
   }

   sum = 4.0*sum;
   printf("With n = %lld terms\n", n);
   printf("   Our estimate of pi = %.14f\n", sum);

   return sum;
}

__global__ void parallel(long long n, double* g_odata) 
{
   extern __shared__ double s_data[];
   double factor;

   unsigned int tid = threadIdx.x;
   uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;

   if(i < n){
      factor = (i % 2 == 0) ? 1.0 : -1.0;
      s_data[tid] = factor/(2*i+1);
   }
   else{
      s_data[tid] = 0;
   }

   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
   {
      if (tid < s) 
      {
         s_data[tid] = s_data[tid] + s_data[tid + s];
      }
      __syncthreads();
   }

   if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

int main(int argc, char* argv[]){
   long long n;
   double* cudaMem;
   double* almost;
   cudaEvent_t start = cudaEvent_t(), stop = cudaEvent_t();
   n = 100000;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for(int i = 0; i < 5; i++){
      double res_seq;
      double res_par = 0;
      n*=10;

      printf("\nSEQUENTIAL:\n");
      
      cudaEventRecord(start, 0);
      res_seq = sequential(n);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float elapsed = 0.;
      cudaEventElapsedTime(&elapsed, start, stop);
      
      printf("Sequential Time: %f\n", elapsed);

      printf("\nPARALLEL:\n");
      cudaMalloc(&cudaMem, ceil(n/1024)*sizeof(double));
     
      cudaEventRecord(start, 0);
      parallel<<<ceil(n/1024),1024,1024*sizeof(double)>>>(n, cudaMem);

      almost = (double*)calloc(ceil(n/1024), sizeof(double));
      cudaMemcpy(almost, cudaMem, ceil(n/1024)*sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(cudaMem);
      for(int j = 0; j < ceil(n/1024); j++)
      {
         res_par += almost[j];
      }
      free(almost);
      res_par *= 4;
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      elapsed = 0.;
      cudaEventElapsedTime(&elapsed, start, stop);
      
      printf("   Our estimate of pi = %.14f\n", res_par);
      printf("Parallel Time: %f\n", elapsed);
      if(abs(res_par - res_seq) <= 0.01)
      {
         printf("\nTEST PASSED\n");
      }
      else
      {
         printf("\nTEST FAILED\n");
      }
   }
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   return 0;
}

