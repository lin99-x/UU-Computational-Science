#include <stdio.h>

int main(int argc, char** argv) {

#pragma omp parallel num_threads(5)
  {
    printf("Bonjour!\n");
  }

  return 0;
}

// just compile with gcc hello_openmp.c
// Just print out Bonjour 1 time