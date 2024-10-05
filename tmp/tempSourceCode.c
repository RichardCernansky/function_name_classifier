#include <stdio.h>
 
 #define SINK 0
 #define NORTH 1
 #define WEST 2
 #define SOUTH 3
 #define EAST 4
 
 int basin[102][102], direction[102][102];
 char result[102][102];
 char latestBasin;
 
 char sinkFinder(int x, int y);
 
 int main() {
   int n;
   int h, w;
   int chooseDir, minorValue;
   int i, j, k;
 
   scanf("%d", &n);
   for (i = 1; i <= n; i++) {
     scanf("%d %d", &h, &w);
 
     for (j = 0; j <= h+1; j++) {
       for (k = 0; k <= w+1; k++) {
         if (j == 0 || k == 0 || j == h+1 || k == w+1) {
           basin[j][k] = 10000;
         } else {
           scanf("%d", &basin[j][k]);
         }
         result[j][k] = ' ';
         direction[j][k] = SINK;
       }
     }
 
     latestBasin = 'a';
 
     for (j = 1; j <= h; j++) {
       for (k = 1; k <= w; k++) {
         minorValue = 10000;
         if (basin[j+1][k] <= minorValue) {
           chooseDir = SOUTH;
           minorValue = basin[j+1][k];
         }
         if (basin[j][k+1] <= minorValue) {
           chooseDir = EAST;
           minorValue = basin[j][k+1];
         }
         if (basin[j][k-1] <= minorValue) {
           chooseDir = WEST;
           minorValue = basin[j][k-1];
         }
         if (basin[j-1][k] <= minorValue) {
           chooseDir = NORTH;
           minorValue = basin[j-1][k];
         }
         if (minorValue < basin[j][k]) {
           direction[j][k] = chooseDir;
         } 
       }
     }
     for (j = 1; j <= h; j++) {
       for (k = 1; k <= w; k++) {
         if (result[j][k] == ' ') {
           sinkFinder(j, k);
         }
       }
     }
 
     printf("Case #%d:\n", i);
     for (j = 1; j <= h; j++) {
       for (k = 1; k <= w; k++) {
         printf("%c", result[j][k]);
         if (k != w) {
           printf(" ");
         }
       }
       printf("\n");
     }
   }
 
   return 1;
 }
 
 char sinkFinder(int x, int y) {
   if (direction[x][y] == SINK) {
     if (result[x][y] == ' ') {
       result[x][y] = latestBasin++;
     }
     return result[x][y];
   } else {
     if (result[x][y] == ' ') {
       switch (direction[x][y]) {
         case NORTH:
           result[x][y] = sinkFinder(x-1, y);
           break;
         case WEST:
           result[x][y] = sinkFinder(x, y-1);
           break;
         case SOUTH:
           result[x][y] = sinkFinder(x+1, y);
           break;
         case EAST:
           result[x][y] = sinkFinder(x, y+1);
           break;
         default:
           break;
       }
     }
     return result[x][y];
   }
 }
