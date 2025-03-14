#include <stdio.h>
#include <string.h>

int CmpString(const void* p1, const void* p2) {
    printf("Comparing %s %s", *(char* const*)p1, *(char* const*)p2);
    return strcmp(*(char* const*)p1, *(char* const*)p2);
}
int main() {
    char *arrStr[] = {"daa", "cbab", "bbbb", "bababa", "ccccc", "aaaa"};
    // printf("%s", arrStr[1]);
    int arrStrLen = sizeof(arrStr) / sizeof(char *);
    qsort(arrStr, arrStrLen, sizeof(char *), CmpString);
    for(int i=0; i<arrStrLen; i++) {
        printf("%s ", arrStr[i]);
    }
    return 0;
}