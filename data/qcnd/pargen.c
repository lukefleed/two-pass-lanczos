/******************************************************************************
 *  This code is part of the NetDsn project by A. Frangioni, C. Gentile,
 *  E. Grande and A. Pacifici
 *
 *  This program produces a randomly-generated .par file to be fed to netgen.
 *
 *  Claudio Gentile, May 28, 2009
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void wait( int seconds )
{
 clock_t endwait;
 endwait = clock () + seconds * CLOCKS_PER_SEC;
 while( clock() < endwait ) {}
 }

int main(int argc , char** argv)
{
 if( argc < 6 ) {
  printf("Usage is: ./pargen m rho k cf cq s\n");
  return( 1 );
  }

 int n, m, seed, rho_par, i, k;
 double rho, sum;
 int MAXVAL;
 FILE *fin, *fout ;
 double *weight, *weight2;
 char filename[ 100 ];
 time_t seconds;

 m = atoi( argv[ 1 ] );
 rho_par = atoi(argv[ 2 ] );
 if( rho_par == 1 ) rho = 0.25;
 else if( rho_par == 2 ) rho = 0.5;
 else if( rho_par == 3 ) rho = 0.75;
 else {
  printf( "Error in parameter 'rho' of pargen.\n" );
  exit( 1 );
  }

 double temp = ( 1.0 + sqrt( 1.0 + ( 8.0 * m ) / rho ) ) / 2;
 n = floor( temp );

 wait( 2 );

 seconds = time( NULL );
 srand( seconds );

 sprintf( filename , "netgen-%s-%s-%s-%s-%s-%s.par" , argv[1] , argv[2] ,
	  argv[3] , argv[4] , argv[5] , argv[6] );      

 fout = fopen(filename,"w");

 // seme per netgen
 // attesa per generare numeri correttamente
 fprintf( fout , "%d\n", rand() );

 // numero istanza
 fprintf( fout , "1 " );

 // numero nodi
 fprintf( fout , "%d ", n ); 

 // source and sink nodes
 int max_nodes = 0.1 * n;
 int num = rand() % max_nodes + 1;
 fprintf( fout , "%d ", num );
 num = rand() % max_nodes + 1;
 fprintf( fout , "%d ", num );

 // numero archi
 fprintf( fout , "%d ", m );

 // min_cost e max_cost, ovvero il minimo e massimo b_ij
 fprintf( fout , "1 " ); //min cost = 1
 num = rand() % 99 + 10;
 fprintf( fout , "%d " , num );

 // total supply
 num = rand() % 900 + 100;
 fprintf( fout , "%d " , num );

 // dati per il transshipment (costanti)
 fprintf( fout , "0 0 0 100 " );

 // capacità u_min e u_max (ricorda che num = total supply in quedto punto)
 int num1 = 0.05 * num;
 int num2 = 0.05 * num;
 int num3 = rand() % num2 + num1; 
 fprintf( fout , "%d " , num3 );
 num1 = 0.2 * num;
 num2 = 0.4 * num - num1;
 num3 = rand() % num2 + num1;
 if( argv[ 6 ] == "s" ) num3 = num3 * 0.7;
 fprintf( fout , "%d " , num3 );

 fclose( fout );

 return( 0 );
 }
