#include <stdio.h>

void swap_nums(int *num1, int *num2) {
    int temp = *num1;
    *num1 = *num2;
    *num2 = temp;
}

void swap_pointers(char *ch1, char *ch2) {
    char temp = *ch1;
    *ch1 = *ch2;
    *ch2 = temp;
}

int main() {
    int a,b;
    char *s1,*s2;
    a = 3, b = 4;
    swap_nums(&a,&b);
    printf("a=%d, b=%d\n", a, b);

    s1 = "second"; s2 = "first";
    swap_pointers(&s1,&s2);
    printf("s1=%s, s2=%s\n", s1, s2);
    return 0;
}