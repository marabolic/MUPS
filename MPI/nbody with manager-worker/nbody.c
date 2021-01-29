#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi/mpi.h>
#include "timer.h"

#define MASTER 0
#define MARGIN 1e+30

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

#define GRAIN_SIZE 10
#define WORK_TAG 1
#define KILL_TAG  2

const double G = 6.673e-11;  
double kinetic_energy_seq, potential_energy_seq;
double kinetic_energy_par, potential_energy_par;

typedef double vect_t[DIM];  /* Vector type for position, etc. */

struct particle_s {
   double m;  /* Mass     */
   vect_t s;  /* Position */
   vect_t v;  /* Velocity */
};

struct work {
    int part;
    int grain_size;
    int n;
};

void Usage(char* prog_name);
void Get_args(int part, int steps, double dt, int freq, char* c, int* n_p, int* n_steps_p, double* delta_t_p,
       int* output_freq_p, char* g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t);
void Compute_energy_par(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p, int rank, int size);
void Compute_energy_seq(struct particle_s curr[], int n, double* kin_en_p,
double* pot_en_p);

void gen_work(int grain_size, int n, int part, struct work *w)
{
    w->grain_size = grain_size;
    w->n = n;
    w->part = part;
}

int sequential(int parts, int steps, double dt, int freq, char* c)
{
   int n;                      /* Number of particles        */
   int n_steps;                /* Number of timesteps        */
   int step;                   /* Current step               */
   int part;                   /* Current particle           */
   int output_freq;            /* Frequency of output        */
   double delta_t;             /* Size of timestep           */
   double t;                   /* Current Time               */
   struct particle_s* curr;    /* Current state of system    */
   vect_t* forces;             /* Forces on each particle    */
   char g_i;                   /*_G_en or _i_nput init conds */
   double start, finish;       /* For timings                */

   Get_args(parts, steps, dt, freq, c, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(curr, n);
   else
      Gen_init_cond(curr, n);

   Compute_energy_seq(curr, n, &kinetic_energy_seq, &potential_energy_seq);
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energy_seq, kinetic_energy_seq, kinetic_energy_seq+potential_energy_seq);   
   //Output_state(0, curr, n);
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));
      for (part = 0; part < n-1; part++)
         Compute_force(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      Compute_energy_seq(curr, n, &kinetic_energy_seq, &potential_energy_seq);
   }
   //Output_state(t, curr, n);


   printf("   PE = %e, KE = %e, Total Energy = %e\n",
       potential_energy_seq, kinetic_energy_seq, kinetic_energy_seq+potential_energy_seq);
   
   free(curr);
   free(forces);
   return 0;
   
}

int parallel(int parts, int steps, double dt, int freq, char* c, int rank, int size)
{
   int n;                      /* Number of particles        */
   int n_steps;                /* Number of timesteps        */
   int step;                   /* Current step               */
   int part;                   /* Current particle           */
   int output_freq;            /* Frequency of output        */
   double delta_t;             /* Size of timestep           */
   double t;                   /* Current Time               */
   struct particle_s* curr;    /* Current state of system    */
   vect_t* forces;             /* Forces on each particle    */
   char g_i;                   /*_G_en or _i_nput init conds */
   double start, finish;       /* For timings                */

   Get_args(parts, steps, dt, freq, c, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (rank == MASTER){
      if (g_i == 'i')
         Get_init_cond(curr, n);
      else
         Gen_init_cond(curr, n);
   }

   MPI_Bcast(curr, 5*n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
   Compute_energy_par(curr, n, &kinetic_energy_par, &potential_energy_par, rank, size);
   if (rank == MASTER){
      printf("   PE = %e, KE = %e, Total Energy = %e\n",
            potential_energy_par, kinetic_energy_par, kinetic_energy_par+potential_energy_par);
   }
   //Output_state(0, curr, n);
   double temp;
   int grain_size = GRAIN_SIZE;
   struct work w;
   int result;
   MPI_Status status;
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));

      if(rank == MASTER)
      {
			int done = 0;
         while(done < n)
         {
           MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
           if(n - done >= grain_size)
           {
              gen_work(grain_size, n, done, &w);
              done += grain_size;
           }
           else
           {
              gen_work(n - done, n, done, &w);
              done = n;
           }
           MPI_Send(&w, sizeof(struct work)/sizeof(int), MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
         }

         for(int m = 1; m < size; m++)
         {
           MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
           MPI_Send(&w, sizeof(struct work)/sizeof(int), MPI_INT, status.MPI_SOURCE, KILL_TAG, MPI_COMM_WORLD);
         }

      }
      else
      {
         while(1){
            MPI_Send(&result, 1, MPI_INT, MASTER, WORK_TAG, MPI_COMM_WORLD);
            MPI_Recv(&w, sizeof(struct work)/sizeof(int), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if(status.MPI_TAG == KILL_TAG)
               break;

            for(int m = 0; m < w.grain_size; m++)
               Compute_force(w.part + m, forces, curr, w.n);
         }
      }
      
      for (int i = 0; i < n; i++){
         MPI_Reduce(&forces[i][0], &temp, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
         if (rank == MASTER){
            forces[i][0] = temp;
         }
         MPI_Reduce(&forces[i][1], &temp, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
         if (rank == MASTER){
            forces[i][1] = temp;
         }
      }
      MPI_Bcast(forces, 2*n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
      
      if (rank == MASTER){
         for (part = 0; part < n; part++)
            Update_part(part, forces, curr, n, delta_t);
      }
      
      MPI_Bcast(curr, 5*n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
     
      Compute_energy_par(curr, n, &kinetic_energy_par, &potential_energy_par, rank, size);
   }
   
   
   //Output_state(t, curr, n);
   if (rank == MASTER){
      printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energy_par, kinetic_energy_par, kinetic_energy_par+potential_energy_par);
   }
   free(curr);
   free(forces);
   return 0;
   
}

int main(int argc, char* argv[]) {
   int rank, size;
   double start = 0, end = 0;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (rank == MASTER){
      printf("\n\n 100");
      printf("\nSEQUENTIAL\n");
      start = MPI_Wtime();
      sequential( 100, 500, 0.01, 500, "g");
      end = MPI_Wtime();
      printf( "The process took: %f\n", end - start);
      
      printf("\nPARALLEL\n");
      start = MPI_Wtime();
   }
   
   
   parallel( 100, 500, 0.01, 500, "g", rank, size);
   

   if (rank == MASTER){
      end = MPI_Wtime();
      printf( "The process took: %f\n", end - start);
      if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
         && abs(potential_energy_seq - potential_energy_par) <= MARGIN )
         printf("\nTest PASSED \n");
      else
         printf("\nTest FAILED \n");
   }

   if (rank == MASTER){
      printf("\n\n 500");
      printf("\nSEQUENTIAL\n");
      start = MPI_Wtime();
      sequential( 500, 500, 0.01, 500, "g");
      end = MPI_Wtime();
      printf( "The process took: %f\n", end - start);
      printf("\nPARALLEL\n");
      start = MPI_Wtime();
   }
   parallel( 500, 500, 0.01, 500, "g", rank, size);
   
   if (rank == MASTER){
      end = MPI_Wtime();
      printf( "The process took: %f\n", end - start);
      if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
         && abs(potential_energy_seq - potential_energy_par) <= MARGIN )
         printf("\nTest PASSED \n");
      else
         printf("\nTest FAILED \n");
   }

    if (rank == MASTER){
      printf("\n\n 5000");
      printf("\nSEQUENTIAL\n");
      start = MPI_Wtime();
      sequential( 5000, 500, 0.01, 500, "g");
      end = MPI_Wtime();
      printf( "The process took: %f\n", end - start);
      printf("\nPARELLEL\n");
      start = MPI_Wtime();
   }
   
   parallel( 5000, 500, 0.01, 500, "g", rank, size);
   
   if (rank == MASTER){
      end = MPI_Wtime();
      printf( "The process took: %f\n", end - start);
      if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
         && abs(potential_energy_seq - potential_energy_par) <= MARGIN )
         printf("\nTest PASSED \n");
      else
         printf("\nTest FAILED \n");
   }
   MPI_Finalize();
   return 0;
}

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
         prog_name);
   fprintf(stderr, "   <size of timestep> <output frequency>\n");
   fprintf(stderr, "   <g|i>\n");
   fprintf(stderr, "   'g': program should generate init conds\n");
   fprintf(stderr, "   'i': program should get init conds from stdin\n");
    
   //exit(0);
}  /* Usage */

void Get_args(int part, int steps, double dt, int freq, char* c, int* n_p, int* n_steps_p, double* delta_t_p,
       int* output_freq_p, char* g_i_p) {
   
   *n_p = part;
   *n_steps_p = steps;
   *delta_t_p = dt;
   *output_freq_p = freq;
   *g_i_p = c;

}  

void Get_init_cond(struct particle_s curr[], int n) {
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++) {
      scanf("%lf", &curr[part].m);
      scanf("%lf", &curr[part].s[X]);
      scanf("%lf", &curr[part].s[Y]);
      scanf("%lf", &curr[part].v[X]);
      scanf("%lf", &curr[part].v[Y]);
   }
}  /* Get_init_cond */

void Gen_init_cond(struct particle_s curr[], int n) {
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   srandom(1);
   for (part = 0; part < n; part++) {
      curr[part].m = mass;
      curr[part].s[X] = part*gap;
      curr[part].s[Y] = 0.0;
      curr[part].v[X] = 0.0;
      if (part % 2 == 0)
         curr[part].v[Y] = speed;
      else
         curr[part].v[Y] = -speed;
   }
}  /* Gen_init_cond */

void Output_state(double time, struct particle_s curr[], int n) {
   int part;
   printf("%.2f\n", time);
   for (part = 0; part < n; part++) {
      printf("%3d %10.3e ", part, curr[part].s[X]);
      printf("  %10.3e ", curr[part].s[Y]);
      printf("  %10.3e ", curr[part].v[X]);
      printf("  %10.3e\n", curr[part].v[Y]);
   }
   printf("\n");
}  /* Output_state */

void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n) {
   int k;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact;

   for (k = part+1; k < n; k++) {
      f_part_k[X] = curr[part].s[X] - curr[k].s[X];
      f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
      len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
      len_3 = len*len*len;
      mg = -G*curr[part].m*curr[k].m;
      fact = mg/len_3;
      f_part_k[X] *= fact;
      f_part_k[Y] *= fact;

      forces[part][X] += f_part_k[X];
      forces[part][Y] += f_part_k[Y];
      forces[k][X] -= f_part_k[X];
      forces[k][Y] -= f_part_k[Y];
   }
}  /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t) {
   double fact = delta_t/curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
}  /* Update_part */

void Compute_energy_seq(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p) {
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;

   for (i = 0; i < n; i++) {
      speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
      ke += curr[i].m*speed_sqr;
   }
   ke *= 0.5;

   for (i = 0; i < n-1; i++) {
      for (j = i+1; j < n; j++) {
         diff[X] = curr[i].s[X] - curr[j].s[X];
         diff[Y] = curr[i].s[Y] - curr[j].s[Y];
         dist = sqrt(diff[X]*diff[X] + diff[Y]*diff[Y]);
         pe += -G*curr[i].m*curr[j].m/dist;
      }
   }

   *kin_en_p = ke;
   *pot_en_p = pe;
}  /* Compute_energy */

void Compute_energy_par(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p, int rank, int size) {
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0, ke_reduce = 0.0, pe_reduce = 0.0;
   double dist, speed_sqr;


   for (i = rank; i < n; i+=size) {
      speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
      ke += curr[i].m*speed_sqr;
   }
   ke *= 0.5;
   MPI_Reduce(&ke, &ke_reduce, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
   

   for (i = rank; i < n-1; i+=size) {
      for (j = i+1; j < n; j++) {
         diff[X] = curr[i].s[X] - curr[j].s[X];
         diff[Y] = curr[i].s[Y] - curr[j].s[Y];
         dist = sqrt(diff[X]*diff[X] + diff[Y]*diff[Y]);
         pe += -G*curr[i].m*curr[j].m/dist;
      }
   }
   MPI_Reduce(&pe, &pe_reduce, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

   *kin_en_p = ke_reduce;
   *pot_en_p = pe_reduce;
}  /* Compute_energy */
