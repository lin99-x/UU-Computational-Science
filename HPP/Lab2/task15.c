#include <stdio.h>

typedef struct product
{
    char name[50];
    double price;
}
product_t;

int main() {
    product_t arr_of_prod[100];
    int i, count=0;
    FILE * fp;
    fp = fopen("data.txt", "r");
    // skip the first line
    fscanf(fp,"%*[^\n]%*c");
    while(fscanf(fp,"%s %lf", arr_of_prod[count].name, &arr_of_prod[count].price) != EOF) {
        count++;
    }
    for(i=0; i<5; i++) {
        printf("%c \t", *arr_of_prod[i].name);
        printf("%.1lf\n",arr_of_prod[i].price);
    }
    fclose(fp);
    return 0;
}