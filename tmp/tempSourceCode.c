#include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 #define MAX_VALUE 10000
 
 int A[MAX_VALUE + 1];
 int B[MAX_VALUE + 1];
 
 int intersection_count(int x, int y, int length)
 {
     int i, count;
     count = i = 0;
     for (; i < length; ++i) {
         if (x > A[i] && y < B[i])
             count++;
         else if (x < A[i] && y > B[i])
             count++;
     }
     return count;
 }
 
 int main(int argc, char *argv[])
 {
     int t, s;
     int i, j, sum;
 
     scanf("%d", &t);
     for (i = 1; i <= t; ++i) {
         scanf("%d", &s);
         sum = 0;
         for (j = 0; j < s; ++j) {
             scanf("%d %d", &A[j], &B[j]);
             sum += intersection_count(A[j], B[j], j);
         }
         printf("Case #%d: %d\n", i, sum);
     }
     return 0;
 }
