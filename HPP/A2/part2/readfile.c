#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE * fp;
    int data1 = 0;
    double data2 = 0;
    char data3;
    float data4;
    fp = fopen ("little_bin_file", "r");
    fread(&data1, sizeof(data1), 1, fp);
    fread(&data2, sizeof(data2), 1, fp);
    fread(&data3, sizeof(data3), 1, fp);
    fread(&data4, sizeof(data4), 1, fp);
    printf("%d\n%lf\n%c\n%f\n", data1, data2, data3, data4);
    fclose(fp);
    return 0;
}