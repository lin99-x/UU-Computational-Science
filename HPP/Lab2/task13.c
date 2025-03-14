#include <stdio.h>
#include <stdlib.h>

int main() {
    char * product;
    double price;
    FILE * fp;
    product = (char*)malloc(10*sizeof(char));
    fp = fopen("data.txt", "r");
    // skip the first line
    fscanf(fp,"%*[^\n]%*c");
    while(fscanf(fp,"%s %lf", product, &price) != EOF) {
        printf("%s \t %.1lf \n", product, price);
    }
    fclose(fp);
    return 0;
}