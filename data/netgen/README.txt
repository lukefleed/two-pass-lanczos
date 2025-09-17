Distribution of the C version of the NETGEN network generator by
Norbert Schlenker based on the original work D. Klingman, A. Napier and J. Stutz

The distribution has been updated on October 8, 2021 by A. Frangioni by:

- replacing the original "portable" random number generator with drand48:
  this makes this version not equivalent to the original one, but it also
  makes it work on modern machines when generating problems with 2^16 or
  more nodes (the original version stalls on these, likely because of the
  random number generation disliking 64-bits long int)

- fixing a few very minor warnings

- adding a bunch of parameter files to generate smallish to largish instances
  with nodes from 2^8 to 2^16 and density between 8 and 256 (not uniformly
  for all node sizes): the istances would be the ones used in

  http://pages.di.unipi.it/frangio/abstracts.html#SIOPT04
  http://pages.di.unipi.it/frangio/abstracts.html#COAP06
  http://pages.di.unipi.it/frangio/abstracts.html#OMS06

  were it not for the incompatible random generator, but they should be
  functionally equivalent to those

- added a batch file to generate all instances in one shot

Usage:

  cd src
  make
  cd ..
  ./batch

All the instances should now be found in the dmx folder.
