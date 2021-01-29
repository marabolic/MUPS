#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define MARGIN 1e-12

void Usage(char* prog_name);
int workshare(long long points);
int sequential(long long points);
int glob_sum_seq;
int glob_sum_par;

int main(){
	printf("\n\n1000000 points\n");
	printf("\nSEQUENTIAL\n");
	sequential(1000000);
	printf("\nPARALLEL\n");
	workshare(1000000);
	if(abs(glob_sum_seq - glob_sum_par) <= MARGIN)
		printf("\nTest PASSED \n");
	else
		printf("\nTest FAILED \n");
	
	printf("\n\n10000000 points\n");
	printf("\nSEQUENTIAL\n");
	sequential(10000000);
	printf("\nPARALLEL\n");
	workshare(10000000);
	if(abs(glob_sum_seq - glob_sum_par) <= MARGIN)
		printf("\nTest PASSED \n");
	else
		printf("\nTest FAILED \n");

	printf("\n\n100000000 points\n");
	printf("\nSEQUENTIAL\n");
	sequential(100000000);
	printf("\nPARALLEL\n");
	workshare(100000000);
	if(abs(glob_sum_seq - glob_sum_par) <= MARGIN)
		printf("\nTest PASSED \n");
	else
		printf("\nTest FAILED \n");

	printf("\n\n1000000000 points\n");
	printf("\nSEQUENTIAL\n");
	sequential(1000000000);
	printf("\nPARALLEL\n");
	workshare(1000000000);
	if(abs(glob_sum_seq - glob_sum_par) <= MARGIN)
		printf("\nTest PASSED \n");
	else
		printf("\nTest FAILED \n");

	printf("\n\n10000000000 points\n");
	printf("\nSEQUENTIAL\n");
	sequential(10000000000);
	printf("\nPARALLEL\n");
	workshare(10000000000);
	if(abs(glob_sum_seq - glob_sum_par) <= MARGIN)
		printf("\nTest PASSED \n");
	else
		printf("\nTest FAILED \n");


}


int sequential(long long points) {
   long long n, i;
   double factor;
   double sum = 0.0;
	
   n = points;
   
   double timeStart = omp_get_wtime();

   printf("Before for loop, factor = %f.\n", factor);
   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }
   printf("After for loop, factor = %f.\n", factor);

   sum = 4.0*sum;
   printf("With n = %lld terms\n", n);
   printf("   Our estimate of pi = %.14f\n", sum);
   printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));

   double timeStop = omp_get_wtime();
   printf("Elapsed time: %f ", timeStop - timeStart);
   return 0;
}

int workshare(long long points) {
   long long n, i;
   double factor;
   double sum = 0.0;
	
   n = points;

   double timeStart = omp_get_wtime();

   printf("Before for loop, factor = %f.\n", factor);
#pragma omp parallel for \
   private(i, factor) \
   shared(n)\
   reduction(+:sum)
   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }
   printf("After for loop, factor = %f.\n", factor);

   sum = 4.0*sum;
   printf("With n = %lld terms\n", n);
   printf("   Our estimate of pi = %.14f\n", sum);
   printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));

   double timeStop = omp_get_wtime();
   printf("Elapsed time: %f ", timeStop - timeStart);
   return 0;
}

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   exit(0);
}
