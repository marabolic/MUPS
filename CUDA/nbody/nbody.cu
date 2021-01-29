#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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

void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n);
__global__
void Compute_forces_gpu(struct particle_s curr[], int n, vect_t forces[]);

void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t);
__global__
void Update_parts_par(struct particle_s* curr, int n, double delta_t, vect_t* forces);
__global__ void Compute_energy_par(struct particle_s curr[], int n, double* g_odata_ke,
      double* g_odata_pe);
void Compute_energy_seq(struct particle_s curr[], int n, double* kin_en_p,
double* pot_en_p);

void gen_work(int grain_size, int n, int part, struct work *w)
{
    w->grain_size = grain_size;
    w->n = n;
    w->part = part;
}

int sequential(int n, int n_steps, double delta_t, int output_freq, char c)
{
   int step;                   /* Current step               */
   int part;                   /* Current particle           */
   struct particle_s* curr;    /* Current state of system    */
   vect_t* forces;             /* Forces on each particle    */

   
   curr = (struct particle_s*)malloc(n*sizeof(struct particle_s));
   forces = (vect_t*)malloc(n*sizeof(vect_t));
   Gen_init_cond(curr, n);

   Compute_energy_seq(curr, n, &kinetic_energy_seq, &potential_energy_seq);
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energy_seq, kinetic_energy_seq, kinetic_energy_seq+potential_energy_seq);   

   for (step = 1; step <= n_steps; step++) {
      //t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));
      for (part = 0; part < n-1; part++)
         Compute_force(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
   }
   Compute_energy_seq(curr, n, &kinetic_energy_seq, &potential_energy_seq);

   printf("   PE = %e, KE = %e, Total Energy = %e\n",
       potential_energy_seq, kinetic_energy_seq, kinetic_energy_seq+potential_energy_seq);
   
   free(curr);
   free(forces);
   return 0;
   
}

int parallel(int n, int n_steps, double delta_t, int output_freq, char c)
{
   int step;
   //double t;
   struct particle_s *curr, *gpu_curr;
   vect_t *gpu_forces;
   double *kinetic_energies, *kinetic_energies_gpu, *potential_energies, *potential_energies_gpu;

   int cl = ceil((double)n/1024);
   curr = (struct particle_s*)calloc(1024*cl, sizeof(struct particle_s));
   cudaMalloc(&gpu_curr, 1024*cl*sizeof(struct particle_s));
   Gen_init_cond(curr, n);
   cudaMalloc(&gpu_forces, 1024*cl*sizeof(vect_t));

   cudaMemcpy(gpu_curr, curr, 1024*cl*sizeof(struct particle_s), cudaMemcpyHostToDevice);
   free(curr);

   cudaMalloc(&kinetic_energies_gpu, cl*sizeof(double));
   cudaMalloc(&potential_energies_gpu, cl*sizeof(double));

   Compute_energy_par<<<cl, 1024, 2*1024*sizeof(double)>>>(gpu_curr, n, kinetic_energies_gpu, potential_energies_gpu);
   cudaDeviceSynchronize();

   kinetic_energies = (double*)malloc(cl*sizeof(double));
   potential_energies = (double*)malloc(cl*sizeof(double));
   cudaMemcpy(kinetic_energies, kinetic_energies_gpu, cl*sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy(potential_energies, potential_energies_gpu, cl*sizeof(double), cudaMemcpyDeviceToHost);
   
   kinetic_energy_par = 0;
   potential_energy_par = 0;
   for(int i = 0; i < cl; i++){
      kinetic_energy_par += kinetic_energies[i];
      potential_energy_par +=  potential_energies[i];
   }
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energy_par, kinetic_energy_par, kinetic_energy_par+potential_energy_par);   

   for (step = 1; step <= n_steps; step++) {
      //t = step*delta_t;
      cudaMemset(gpu_forces, 0, cl*1024*sizeof(vect_t));
      Compute_forces_gpu<<<cl, 1024>>>(gpu_curr, n, gpu_forces);
      cudaDeviceSynchronize();
      Update_parts_par<<<cl, 1024>>>(gpu_curr, n, delta_t, gpu_forces);
      cudaDeviceSynchronize();
   }


   Compute_energy_par<<<cl, 1024, 2*1024*sizeof(double)>>>(gpu_curr, n, kinetic_energies_gpu, potential_energies_gpu);
   cudaDeviceSynchronize();
   cudaMemcpy(kinetic_energies, kinetic_energies_gpu, cl*sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy(potential_energies, potential_energies_gpu, cl*sizeof(double), cudaMemcpyDeviceToHost);
   cudaFree(kinetic_energies_gpu);
   cudaFree(potential_energies_gpu);
   kinetic_energy_par = 0;
   potential_energy_par = 0;
   for(int i = 0; i < cl; i++){
      kinetic_energy_par += kinetic_energies[i];
      potential_energy_par +=  potential_energies[i];
   }
   free(kinetic_energies);
   free(potential_energies);

   printf("   PE = %e, KE = %e, Total Energy = %e\n",
       potential_energy_par, kinetic_energy_par, kinetic_energy_par+potential_energy_par);
   
   cudaFree(gpu_curr);
   cudaFree(gpu_forces);
   return 0;
}

int main(int argc, char* argv[]) {
   float elapsed;
   cudaEvent_t start = cudaEvent_t(), stop = cudaEvent_t();
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   printf("\n\n 100");
   printf("\nSEQUENTIAL\n");
   cudaEventRecord(start, 0);
   sequential( 100, 500, 0.01, 500, 'g');
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   elapsed = 0.;
   cudaEventElapsedTime(&elapsed, start, stop);
   printf( "The process took: %f\n", elapsed);
   
   printf("\nPARALLEL\n");
   cudaEventRecord(start, 0);
   parallel(100, 500, 0.01, 500, 'g');
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   elapsed = 0.;
   cudaEventElapsedTime(&elapsed, start, stop);
   
    printf( "The process took: %f\n", elapsed);
    if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
        && abs(potential_energy_seq - potential_energy_par) <= MARGIN )
        printf("\nTest PASSED \n");
    else
        printf("\nTest FAILED \n");

    
    printf("\n\n 500");
    printf("\nSEQUENTIAL\n");
    cudaEventRecord(start, 0);
    sequential( 500, 500, 0.01, 500, 'g');
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed = 0.;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf( "The process took: %f\n", elapsed);
    
    printf("\nPARALLEL\n");
    cudaEventRecord(start, 0);
    parallel(500, 500, 0.01, 500, 'g');
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed = 0.;
    cudaEventElapsedTime(&elapsed, start, stop);
      
      
   
    printf( "The process took: %f\n", elapsed);
    if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
        && abs(potential_energy_seq - potential_energy_par) <= MARGIN )
        printf("\nTest PASSED \n");
    else
        printf("\nTest FAILED \n");


    printf("\n\n 5000");
    printf("\nSEQUENTIAL\n");
    cudaEventRecord(start, 0);
    sequential( 5000, 500, 0.01, 500, 'g');
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed = 0.;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf( "The process took: %f\n", elapsed);
    
    printf("\nPARALLEL\n");
    cudaEventRecord(start, 0);
    parallel(5000, 500, 0.01, 500, 'g');
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed = 0.;
    cudaEventElapsedTime(&elapsed, start, stop);
    
    printf( "The process took: %f\n", elapsed);
    if(abs(kinetic_energy_seq - kinetic_energy_par) <= MARGIN 
    && abs(potential_energy_seq - potential_energy_par) <= MARGIN )
    printf("\nTest PASSED \n");
    else
    printf("\nTest FAILED \n");
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}


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

void Compute_force(int part, vect_t forces[], struct particle_s curr[], int n) {
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

__global__
void Compute_forces_gpu(struct particle_s curr[], int n, vect_t forces[])
{
   int k;
   int part = blockIdx.x*blockDim.x + threadIdx.x;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact;
   double fpx=0, fpy=0;
   

   if(part < n){
      for (k = 0; k < n; k++) {
         f_part_k[X] = curr[part].s[X] - curr[k].s[X];
         f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         if(len != 0){
            len_3 = len*len*len;
            mg = -G*curr[part].m*curr[k].m;
            fact = mg/len_3;
            f_part_k[X] *= fact;
            f_part_k[Y] *= fact;

            fpx += f_part_k[X];
            fpy += f_part_k[Y];
         }
      }
   }
   __syncthreads();
   forces[part][X] += fpx;
   forces[part][Y] += fpy;

}

void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t) {
   double fact = delta_t/curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
}  /* Update_part */

__global__
void Update_parts_par(struct particle_s* curr, int n, double delta_t, vect_t* forces)
{
   uint32_t part = blockIdx.x*blockDim.x + threadIdx.x;
   if(part < n)
   {  
      double fact = delta_t/curr[part].m;
      curr[part].s[X] += delta_t * curr[part].v[X];
      curr[part].s[Y] += delta_t * curr[part].v[Y];
      curr[part].v[X] += fact * forces[part][X];
      curr[part].v[Y] += fact * forces[part][Y];
   }
}

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

__global__
void Compute_energy_par(struct particle_s curr[], int n, double* g_odata_ke, double* g_odata_pe) {
   vect_t diff;
   int j;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;
   uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
   uint32_t thx = threadIdx.x;
   extern __shared__ double ke_arr[];

    speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
    if(i < n)
       ke = 0.5*curr[i].m*speed_sqr;

    if (i < n-1)
        for (j = i+1; j < n; j++) {
            diff[X] = curr[i].s[X] - curr[j].s[X];
            diff[Y] = curr[i].s[Y] - curr[j].s[Y];
            dist = sqrt(diff[X]*diff[X] + diff[Y]*diff[Y]);
            pe += -G*curr[i].m*curr[j].m/dist;
        }
    
    double * pe_arr = ke_arr + 1024;
    if(i < n)
    {
      ke_arr[thx] = ke;
      pe_arr[thx] = pe;
    }
    else
    {
      ke_arr[thx] = 0;
      pe_arr[thx] = 0;
    }

   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
   {
      if (thx < s) 
      {
        ke_arr[thx] = ke_arr[thx] + ke_arr[thx + s];
        pe_arr[thx] = pe_arr[thx] + pe_arr[thx + s];
      }
      __syncthreads();
   }

   if (thx == 0) 
   {
      g_odata_ke[blockIdx.x] = ke_arr[0];
      g_odata_pe[blockIdx.x] = pe_arr[0];
   }
}  /* Compute_energy */
