#include <stdio.h>
 #include <string.h>
 #include <stdlib.h>
 #define MAXLEN 510
 #define MAXSYMB 25
 
 int main ()
 {
      int T;
      scanf("%d\n",&T);
      static char data[MAXLEN];
      static char symb[MAXSYMB];
      static int a[MAXSYMB];
      strcpy(symb,"welcome to code jam");
      int iT;
      int i, j, N;
      
      for (iT = 0; iT < T; iT++)
      {
           gets(data);
           N = strlen(data);
           memset(a,0,sizeof(a));
           a[0] = 1;
           for (i = 0; i < N; i++)
           {
                for (j = 19; j >= 1; j--)
                {
                     if (data[i] == symb[j-1]) a[j] = (a[j] + a[j-1]) % 10000;
                }
           }
           printf("Case #%d: ",iT+1);
           if (a[19] < 1000) printf("0");
           if (a[19] < 100) printf("0");
           if (a[19] < 10) printf("0");
           printf("%d\n",a[19]);
      }
      
      return 0;
 }
