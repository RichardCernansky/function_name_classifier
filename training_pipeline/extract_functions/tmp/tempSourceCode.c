#include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 
 typedef struct lst_element lstElement;
 struct lst_element {
 	char *name;
 	lstElement *next;
 };
 
 lstElement *first, *now;
 
 int main(int argc, char *argv[]) {
 	FILE *in, *out;
 	in = fopen(argv[1], "r");
 	out = fopen("output.out", "w");
 
 	int N;
 	fscanf(in, "%i\n", &N);
 	
 	int n;
 	for (n = 0; n < N; n++) {
 		int S;
 		fscanf(in, "%i\n", &S);
 		
 		int s;
 		char *buffer = malloc(sizeof(char) * 100);
 		for (s = 0; s < S; s++) {
 			fgets(buffer, 100, in);
 		}
 		
 		int Q;
 		fscanf(in, "%i\n", &Q);
 
 		first = malloc(sizeof(lstElement));
 		first->next = NULL;
 		first->name = malloc(sizeof(char) * 100);
 		fgets(buffer, 100, in);
 		strcpy(first->name, buffer);
 		int anzahl = 1;
 		int wechsel = 0;
 
 //printf("query: %s\t\tanzahl: %i\n", buffer, anzahl);
 		int q;
 		for (q = 1; q < Q; q++) {
 			fgets(buffer, 100, in);
 
 			lstElement *tmp = first;
 			lstElement *ins = NULL;
 			while (tmp) {
 				if (strcmp(tmp->name, buffer)) {
 					ins = tmp;
 					tmp = tmp->next;
 				} else
 					break;
 			}
 
 			if (tmp == NULL) {
 				ins->next = malloc(sizeof(lstElement));
 				tmp = ins->next;
 				tmp->next = NULL;
 				tmp->name = malloc(sizeof(char) * 100);
 				
 				strcpy(tmp->name, buffer);
 
 				anzahl++;
 			}
 
 			if (anzahl == S) {
 //printf("---> Wechsel\n");
 				wechsel++;
 				anzahl = 1;
 
 				tmp = first;
 				while (tmp) {
 					lstElement *del = tmp;
 					tmp = tmp->next;
 					free(del->name);
 					free(del);
 				}
 
 				first = malloc(sizeof(lstElement));
 				first->next = NULL;
 				first->name = malloc(sizeof(char) * 100);
 				strcpy(first->name, buffer);			
 			}
 //printf("query: %s\t\tanzahl: %i\n\t\t[%i/%i]\n", buffer, anzahl, q+1, Q);
 		}
 
 		lstElement *tmp = first;
 		while (tmp) {
 			lstElement *del = tmp;
 			tmp = tmp->next;
 			free(del->name);
 			free(del);
 		}
 
 		fprintf(out, "Case #%i: %i\n", n+1, wechsel);
 	}
 }
