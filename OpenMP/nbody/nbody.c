#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */
#define MARGIN 1e-12

const double G = 6.673e-11;  
double kinetic_energy_seq, potential_energy_seq;
double kinetic_energy_par, potential_energy_par;

typedef double vect_t[DIM];  /* Vector type for position, etc. */

struct particle_s {
   double m;  /* Mass     */
   vect_t s;  /* Position */
   vect_t v;  /* Velocity */
};

void Usage(char* prog_name);
void Get_args(int part, int steps, double dt, int freq, char * c, int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force_PAR(int part, vect_t forces[], struct particle_s curr[], 
      int n);
void Compute_force_SEQ(int part, vect_t forces[], struct particle_s curr[], 
      int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t);
void Compute_energy_PAR(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p);
void Compute_energy_SEQ(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p);

int main_SEQ(int parts, int steps, double dt, int freq, char* c) {
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
   double kinetic_energy, potential_energy;
   double start, finish;       /* For timings                */

   Get_args(parts, steps, dt, freq, c, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(curr, n);
   else
      Gen_init_cond(curr, n);

   start = omp_get_wtime();
   Compute_energy_SEQ(curr, n, &kinetic_energy, &potential_energy);
   kinetic_energy_seq = kinetic_energy;
   potential_energy_seq = potential_energy;
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energy, kinetic_energy, kinetic_energy+potential_energy);
   //Output_state(0, curr, n);
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));
      for (part = 0; part < n-1; part++)
         Compute_force_SEQ(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      Compute_energy_SEQ(curr, n, &kinetic_energy, &potential_energy);
	  kinetic_energy_seq = kinetic_energy;
	  potential_energy_seq = potential_energy;
   }
   //Output_state(t, curr, n);

   printf("   PE = %e, KE = %e, Total Energy = %e\n",
  		 potential_energy, kinetic_energy, kinetic_energy+potential_energy);
   
   finish = omp_get_wtime();
   printf("Elapsed time = %e seconds\n", finish-start);

   free(curr);
   free(forces);
   return 0;
}  /* main */


int main_PAR(int parts, int steps, double dt, int freq, char* c) {
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
   double kinetic_energy, potential_energy;
   double start, finish;       /* For timings                */

   Get_args(parts, steps, dt, freq, c, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(curr, n);
   else
      Gen_init_cond(curr, n);

   start = omp_get_wtime();
   Compute_energy_PAR(curr, n, &kinetic_energy, &potential_energy);
   kinetic_energy_par = kinetic_energy;
   potential_energy_par = potential_energy;
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energy, kinetic_energy, kinetic_energy+potential_energy);
   //Output_state(0, curr, n);
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));
      for (part = 0; part < n-1; part++)
         Compute_force_PAR(part, forces, curr, n);

      #pragma omp parallel for private(part)
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      
      Compute_energy_PAR(curr, n, &kinetic_energy, &potential_energy);
	  kinetic_energy_par = kinetic_energy;
	  potential_energy_par = potential_energy;
   }
   //Output_state(t, curr, n);

   printf("   PE = %e, KE = %e, Total Energy = %e\n",
  		 potential_energy, kinetic_energy, kinetic_energy+potential_energy);
   
   finish = omp_get_wtime();
   printf("Elapsed time = %e seconds\n", finish-start);

   free(curr);
   free(forces);
   return 0;
}  /* main */


int main(){
  
   printf("\n\n 100");
   printf("\nSEQUENTIAL\n");
   main_SEQ( 100, 500, 0.01, 500, "g");
   printf("\nPARALLEL\n");
   main_PAR( 100, 500, 0.01, 500, "g");
   if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
		&& abs(potential_energy_seq - potential_energy_par) <= MARGIN )
      printf("\nTest PASSED \n");
   else
      printf("\nTest FAILED \n");
	
   printf("\n\n 500");
   printf("\nSEQUENTIAL\n");
   main_SEQ( 500, 500, 0.01, 500, "g");
   printf("\nPARALLEL\n");
   main_PAR( 500, 500, 0.01, 500, "g");
   if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
		&& abs(potential_energy_seq - potential_energy_par) <= MARGIN )
      printf("\nTest PASSED \n");
   else
      printf("\nTest FAILED \n");

	
   printf("\n\n 5000");
   printf("\nSEQUENTIAL\n");
   main_SEQ( 5000, 500, 0.01, 500, "g");
   printf("\nPARELLEL\n");
   main_PAR( 5000, 500, 0.01, 500, "g");
   if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
		&& abs(potential_energy_seq - potential_energy_par) <= MARGIN )
      printf("\nTest PASSED \n");
   else
      printf("\nTest FAILED \n");

}


void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
         prog_name);
   fprintf(stderr, "   <size of timestep> <output frequency>\n");
   fprintf(stderr, "   <g|i>\n");
   fprintf(stderr, "   'g': program should generate init conds\n");
   fprintf(stderr, "   'i': program should get init conds from stdin\n");
    
   exit(0);
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

void Compute_force_SEQ(int part, vect_t forces[], struct particle_s curr[], int n) {
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
}  

void Compute_force_PAR(int part, vect_t forces[], struct particle_s curr[], int n) {
   int k;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact, tempForcesX = 0, tempForcesY = 0;

   #pragma omp parallel for \
      shared(part, curr, n, forces) \
      private(k, fact, len, len_3, f_part_k, mg) \
      reduction(+:tempForcesX, tempForcesY)
   for (k = part+1; k < n; k++) {
      f_part_k[X] = curr[part].s[X] - curr[k].s[X];
      f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
      len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
      len_3 = len*len*len;
      mg = -G*curr[part].m*curr[k].m;
      fact = mg/len_3;
      f_part_k[X] *= fact;
      f_part_k[Y] *= fact;

      
      tempForcesX += f_part_k[X];
      tempForcesY += f_part_k[Y];
      forces[k][X] -= f_part_k[X];
      forces[k][Y] -= f_part_k[Y];

      
   }

   forces[part][X] += tempForcesX;
   forces[part][Y] += tempForcesY;


}  /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t) {
   double fact = delta_t/curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
} 

void Compute_energy_SEQ(struct particle_s curr[], int n, double* kin_en_p, double* pot_en_p) {
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
}

void Compute_energy_PAR(struct particle_s curr[], int n, double* kin_en_p, double* pot_en_p) {
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;

   #pragma omp parallel for \
      shared(curr, n) \
      private(speed_sqr, i) \
      reduction(+:ke)
   for (i = 0; i < n; i++) {
      speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
      ke += curr[i].m*speed_sqr;
   }
   ke *= 0.5;

   for (i = 0; i < n-1; i++) {
      #pragma omp parallel for \
         private(j, diff, dist) \
         shared(i, curr, n) \
         reduction(+:pe)
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
