/*** This is a portable random number generator whose origins are
 *** unknown.  As far as can be told, this is public domain software. */

/*** portable random number generator */

/*** Note that every variable used here must have at least 31 bits
 *** of precision, exclusive of sign.  Long integers should be enough.
 *** The generator is the congruential:  i = 7**5 * i mod (2^31-1).
 ***/

//!! modified by A. Frangioni on October 9, 2021: the "portable" random
//!! number generator was creating problems on modern systems that led
//!! netgen to stall, the guess being that it was not really right with
//!! 64-bits long integers, so it has been replaced by calls to drand48
#include <stdlib.h>

//!! #define MULTIPLIER 16807
//!! #define MODULUS    2147483647
//!!
//!! static long saved_seed;


/*** set_random - initialize constants and seed */

void set_random(seed)
long seed;
{
 //!!  saved_seed = seed;
 srand48(seed);
}


/*** random - generate a random integer in the interval [a,b] (b >= a >= 0) */

long random1(a, b)
long a, b;
{
 //!!  register long hi, lo;
 //!!
 //!!  hi = MULTIPLIER * (saved_seed >> 16);
 //!!  lo = MULTIPLIER * (saved_seed & 0xffff);
 //!!  hi += (lo>>16);
 //!! lo &= 0xffff;
 //!! lo += (hi>>15);
 //!! hi &= 0x7fff;
 //!! lo -= MODULUS;
 //!! if ((saved_seed = (hi<<16) + lo) < 0)
 //!!   saved_seed += MODULUS;
 //!!
 //!!  if (b <= a)
 //!!    return b;
 //!!  return a + saved_seed % (b - a + 1);
 return a + ( b - a ) * drand48();
}

