//
//  func.c
//  
//
//  Created by MacBook return 0; on 11.11.14.
//
//

#include <stdio.h>

int main(void)
{
    unsigned long long n;
    long long res;
    scanf("%I64d", &n);
    if (n%2) {
        res = -((n + 1)/2);
    } else {
        res = n/2;
    }
    printf("%I64d", res);
    return 0;
}