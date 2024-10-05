/* gcc -o pb1 pb1.c  */
 
 #include <stdio.h>
 /* #include <stdlib.h> */
 
 #define TRUE 1
 #define FALSE 0
 
 
 static void solve (void);
 
 int main () 
 {
     freopen ("in.txt", "r", stdin);
     freopen ("out.txt", "w", stdout);
     int t,i;
     scanf ("%d", &t);
     for (i = 0; i < t; i++) {
         printf ("Case #%d: ", i + 1);
         solve ();
     }
     return FALSE;
 }
 
 
 void solve (void) 
 {
     int i;
     int res = 0;
     int r,c,w;
 
     scanf("%d %d %d", &r, &c, &w);
     
     res = r*(c/w)+w;
 
     if(c%w == 0){
         res --;
     }
 
     printf("%d\n",res);
 }
 
 
