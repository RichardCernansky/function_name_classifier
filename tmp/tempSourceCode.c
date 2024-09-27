#include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 
 FILE* fin;
 FILE* fout;
 	
 int is_connect(int *a, int *b, int k) {
 	if (a[0] == b[0]) return 1;
 	int i;
 	
 	for (i = 1; i < k; i++) {
 		if (a[i] == b[i])
 			return 1;
 		if ((a[i-1] < b[i-1] && a[i] > b[i]) ||
 			(a[i-1] > b[i-1] && a[i] < b[i])) {
 			return 1;
 		}
 	}
 	return 0;
 }
 
 int max_pos(int *a, int n) {
 	int i, max;
 	max = 0;
 	for (i = 0; i < n; i++) {
 		if (a[i] > a[max])
 			max = i;
 	}
 	return max;
 }
 
 int main() {
 	fin = fopen("stock.in", "r");
 	fout = fopen("stock.out", "w");
 	
 	int num_test, t;
 	fscanf(fin, "%d\n", &num_test);
 	for (t = 1; t <= num_test; t++) {
 		printf("Test %d\n", t);
 		int a[100][100], n;
 		
 		int stock[100][100];
 		int i, j, k;
 		
 		fscanf(fin, "%d %d\n", &n, &k);
 		printf("N = %d K = %d ", n , k);
 		
 		for (i = 0; i < n; i++) {
 			for (j = 0; j < k; j++) {
 				fscanf(fin, "%d ", &stock[i][j]);
 			}
 			fscanf(fin, "\n");
 		}
 		
 		memset(a, 0, sizeof(a));
 		for (i = 0; i < n; i++) {
 			for (j = 0; j < n; j++) {
 				if (i != j) {
 					if (is_connect(stock[i], stock[j], k) == 1) {
 						a[i][j] = 1;
 					}
 				}
 			}
 		}
 			
 		for (i = 0; i < n; i++) {
 			for (j = 0; j < n; j++) {
 				printf("%d ", a[i][j]);
 			}
 			printf("\n");
 		}
 		
 		int bac[1000], color[1000];
 		memset(bac, 0, sizeof(bac));
 		for (i = 0; i < n; i++) {
 			for (j = 0; j < n; j++) {
 				if (a[i][j] == 1) {
 					bac[i]++;
 				}
 			}
 		}
 		
 		printf("Bac :");
 		for (i = 0; i < n; i++) 
 			printf("%d ", bac[i]);
 		printf("\n");
 		
 		int times;
 		int co[100];
 		
 		for (times = 0; times < n; times++) {
 			int pos = max_pos(bac, n);
 			
 			memset(co, 0,sizeof(co));
 			for (i = 0; i < n; i++) {
 				if (a[pos][i] == 1) {
 					co[color[i]] = 1;
 				}		
 			}
 			
 			int j = 1;
 			while (co[j] != 0) j++;
 			
 			color[pos] = j;
 			for (i = 0; i < n; i++) {
 				if (a[pos][i] == 1) {
 					bac[i]--;
 					//a[pos][i] = a[i][pos] = 0;
 				}
 			}
 			bac[pos] = -1;
 		}
 		
 		int max = 0;
 		
 		for (i = 0; i < n; i++)  {
 			printf("%d ", color[i]);
 			if (color[i] > max)
 				max = color[i];
 		}
 		printf("\n");
 		
 		fprintf(fout, "Case #%d: %d\n", t, max);
 	}
 		
 	
 	return 0;	
 }
 
