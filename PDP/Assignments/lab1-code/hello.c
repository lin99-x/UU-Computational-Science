/**********************************************************************
 * A simple "hello world" program for MPI/C
 *
 **********************************************************************/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int size, rank;

  MPI_Init(&argc, &argv);               /* Initialize MPI               */
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* Get the number of PEs        */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get my number                */
  printf("We are %i of us!\n", size);
  printf("Hello World from %i!\n", rank);             /* Print a message              */

  MPI_Finalize();                       /* Shut down and clean up MPI   */

  return 0;
}
