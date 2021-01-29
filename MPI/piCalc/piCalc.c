#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MASTER 0


void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   exit(1);
}


double sequential(long long n) {
   long long i;
   double factor;
   double sum = 0.0;

   double start = MPI_Wtime();

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

   double end = MPI_Wtime();
   printf( "The process took: %f", end - start);

   return sum;
}

double parallel(long long n, int rank, int size) {
   long long i;
   double factor;
   double sum = 0.0;
   double reduced_sum = 0.0;

   double start = MPI_Wtime();

   MPI_Bcast(&n, 1, MPI_LONG_LONG, MASTER, MPI_COMM_WORLD);
   
   if (rank == MASTER){
      printf("Before for loop, factor = %f.\n", factor);
   }
   for (i = rank; i < n; i+=size) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }
   if (rank == MASTER){
      printf("After for loop, factor = %f.\n", factor);
   }

   MPI_Reduce(&sum, &reduced_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);


   reduced_sum = 4.0*reduced_sum;

   if (rank == MASTER){
      printf("With n = %lld terms\n", n);
      printf("   Our estimate of pi = %.14f\n", reduced_sum);
      printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));
   }

   double end = MPI_Wtime();
   if (rank == MASTER){
      printf( "The process took: %f", end - start);
   }
   
   return reduced_sum;
}

int main(int argc, char* argv[]){
   int rank, size;
   long long n;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   n = 100000;
   for(int i = 0; i < 5; i++){
      double res_seq;
      if(rank == MASTER){
         n*=10;
         printf("\nSEQUENTIAL:\n");
         res_seq = sequential(n);
         printf("\nPARALLEL:\n");
      }
      double res_par;
      res_par = parallel(n, rank, size);
      if(rank == MASTER){
         if( abs(res_par - res_seq) <= 0.01)
         {
            printf("\nTEST PASSED\n");
         }
         else
         {
            printf("\nTEST FAILED\n");
         }
      }
   }
   MPI_Finalize();
   return 0;
}

