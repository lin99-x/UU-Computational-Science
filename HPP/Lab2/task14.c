#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[]) {
    char filename[64];
    FILE * fp;
    char * buffer;
    printf("Program name %s\n", argv[0]);
    if(argc == 2) {
        strcpy(filename,argv[1]);
        printf("The argument supplied is %s\n", argv[1]);
    }
    else if(argc>2) {
        printf("Too many arguments supplied.\n");
    }
    else {
        printf("One argument expected.\n");
    }
    fp = fopen(filename, "r");
    fclose(fp);
    return 0;
}