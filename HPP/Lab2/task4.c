#include <stdio.h>

int main() {
    double i, j;
    char c;
    int result;
    printf("Input: ");
    scanf("%lf%c%lf", &i, &c, &j);
    switch (c)
    {
    case '+':
        result = i + j;
        printf("Output: %d", result);
        break;
    case '-':
        result = i - j;
        printf("Output: %d", result);
        break;
    case '*':
        result = i * j;
        printf("Output: %d", result);
        break;
    case '/':
        result = i / j;
        printf("Output: %d", result);
    default:
        break;
    }
    return 0;
}