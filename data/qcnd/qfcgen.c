/******************************************************************************
 *  This code is part of the NetDsn project by A. Frangioni, C. Gentile,
 *  E. Grande and A. Pacifici
 *
 *  This program produces the cost file for fixed and quadratic costs.
 *
 *  Claudio Gentile, May 28, 2009
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main( int argc , char** argv )
{
 if( argc < 2 ) {
  printf( "Usage is: ./qfcgen namefile \n" );
  return( 1 );
  }

 double *opU;
 double *Cb;
 FILE *file = fopen( argv[ 1 ] , "r" );
   
 char str[ 100 ];
 int nstr , n , m;
 int arc = 0;
 int mode = 0;

 while( ! feof( file ) ) {
  // leggo il primo carattere della riga
  fgets( str , 2 , file );
  // se la  riga è un commento
  if( str[ 0 ] == 'c' ) {
   fgets( str , 100 , file );
   // se l'ultimo carattere è '\n' allora è terminata la riga
   // altrimenti la continuo a leggere
   if( str[ strlen( str ) - 1 ] != '\n' )
    fgets( str , 100 , file );
    }
  else
   if( str[ 0 ] == 'p' ) {
    // se e' la definizione del problema
    // lettura tipo problema (min o max)
    fscanf( file , "%s" , str );

    // lettura numero nodi
    fscanf( file , "%d" , &n );

    // lettura numero archi
    fscanf( file , "%d" , &m );

    // allocazione vettore capacita' archi
    opU = new double[ m ];
    // allocazione vettore costi archi
    Cb = new double[ m ];

    // per finire la linea  (ci sara' un '\n' e degli spazi da scartare)
    fgets( str , 100 , file );
    }
   else
    if( str[0] == 'n' )	{
     // se e' la descrizione dei nodi e relativi deficit/surplus
     fscanf( file , "%d" , &nstr );
     fscanf ( file , "%d" , &nstr );
     // per finire la linea  (ci sara' un '\n' e degli spazi da scartare)
     fgets( str , 100 , file );
     }
    else
     if( str[ 0 ] == 'a' ) {
      // se e' la descrizione degli archi e dei costi ...
      // start node
      fscanf( file , "%d" , &nstr );
      fscanf( file , "%d" , &nstr );
      fscanf( file , "%d" , &nstr );
      // upper bound
      fscanf( file , "%d" , &nstr );
      opU[ arc ] = (double) nstr;
      // costo
      fscanf( file , "%d" , &nstr );
      Cb[ arc ] = (double) nstr;

      arc++;

      // per finire la linea  (ci sara' un '\n' e degli spazi da scartare)
      fgets( str , 100 , file );
      }
  }

 fclose( file );

 // lettura informazioni dal nomefile
 char nomefile[ 100 ];
 strcpy( nomefile , argv[ 1 ] );

 bool eof = 0;
 int c = 0;

 double Cc[ m ];
 double Ca[ m ];
 double Ccm_1, Ccm_2;
 double Cam_1, Cam_2;

 time_t seconds;

 // c_ij estratto casualmente in b_ij * u_ij * [ m_1 , m_2 ]
 // dove
 //       | 0.1 nel caso di costi c_ij bassi
 // m_1 = |
 //       | 3.0 nel cado di costi c_ij alti
 //
 //       | 0.3 nel caso di costi c_ij bassi
 // m_2 = |
 //       | 10.0 nel cado di costi c_ij alti

 while( ! eof )	{
  int conta = 0;
  while( conta < 4 ) {
   if( nomefile[ c ] == '-' )
    conta++;
   c++;
   }

  // ora c  posizionato sul parametro cf
  //costi fissi
  if( nomefile[ c ] == 'a' ) {
   Ccm_1 = 3.0;
   Ccm_2 = 10.0;
   }
  else
   if( nomefile[ c ] == 'b' ) {
    Ccm_1 = 0.5;
    Ccm_2 = 1.0;
    }
   else {
    printf( "Errore parametro cf\n" );
    exit( 1 );
    }
	  
  //andiamo avanti di due caratteri
  c = c + 2;

  //costi quadratici
  if( nomefile[ c ] == 'a' ) {
   Cam_1 = 100;
   Cam_2 = 1000;
   }
  else
   if( nomefile[ c ] == 'b' ) {
    Cam_1 = 1;
    Cam_2 = 3;
    }
   else {
    printf( "Errore parametro cq\n" );
    exit( 1 );
    }
	  
  //andiamo avanti di due caratteri
  c = c + 2;

  //scalatura
  if( nomefile[ c ] == 's' ) {
   for( int e = 0 ; e < m ; e++ )
    opU[ e ] = 0.7 * opU[ e ];
   }
  else
   if( ( nomefile[ c ] != 'n' ) || ( nomefile[ c + 1 ] != 's' ) ) {
    printf( "Errore parametro s\n" );
    exit( 1 );
    }

  eof = 1;
  }

 seconds = time( NULL );
 srand( seconds );

 int num1, num2;
 double dnum1, diff;
 double num, div;
 for( int e = 0 ; e < m ; e++ )	{
  num1 = Cb[ e ] * Ccm_1;
  num2 = Cb[ e ] * Ccm_2 - num1;
  num = rand() % num2 + num1;
  Cc[ e ] = num + 1;

  div = Cc[ e ];
  dnum1 = div * Cam_1;
  diff = div * Cam_2 - dnum1;
  num = ( ( (double)rand() ) / RAND_MAX) * diff;
  num = num + dnum1;
  Ca[ e ] = num + 1;
  }

 // generazione file .qfc
     
 char filename[ 100 ];
 FILE *fout;

 filename[ 0 ] = '\000';
 strncat( filename , nomefile , strlen( nomefile ) - 3 ); 

 strcat( filename , "qfc" );
 fout = fopen( filename , "w" );

 //numero archi
 fprintf( fout , "%d\n" , m );
 // prima riga: costi fissi c_ij
 for( int i = 0 ; i < m ; i++ )
  fprintf( fout , "%f " , Cc[ i ] );
 fprintf( fout , "\n" );
 // seconda riga: costi quadratici a_ij
 for( int i = 0 ; i < m ; i++ )
  fprintf( fout , "%f ", Ca[ i ] );
 fprintf( fout , "\n" );

 fclose( fout );

 return( 0 );
 }
