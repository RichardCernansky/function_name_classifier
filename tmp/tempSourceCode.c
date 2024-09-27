#include <stdio.h>
 
 typedef struct s_point
 {
     int alt;
     int resolved;
     char letter;
     struct s_point *basin;
 }
 point;
 
 point pts[100][100];
 char letter;
 int T, H, W;
 
 char resolve_point(int h, int w)
 {
     int a_n = 10000, a_w = 10000, a_e = 10000, a_s = 10000;
     int a = pts[h][w].alt;
 
     if (pts[h][w].resolved)
         return pts[h][w].letter;
 
     if (h > 0) a_n = pts[h-1][w].alt;
     if (w > 0) a_w = pts[h][w-1].alt;
     if (h < H-1) a_s = pts[h+1][w].alt;
     if (w < W-1) a_e = pts[h][w+1].alt;
 
     if (a <= a_n && a <= a_s && a <= a_w && a <= a_e)
     {
         // we have a basin
         
         pts[h][w].letter = letter;
         letter++;
     }
     else if (a_n <= a_w && a_n <= a_s && a_n <= a_e)
     {
         // get n
         pts[h][w].letter = resolve_point(h-1, w);
     }
     else if (a_w <= a_n && a_w <= a_s && a_w <= a_e)
     {
         // get w
         pts[h][w].letter = resolve_point(h, w-1);
     }
     else if (a_e <= a_n && a_e <= a_s && a_e <= a_w)
     {
         // get e
         pts[h][w].letter = resolve_point(h, w+1);
     }
     else if (a_s <= a_n && a_s <= a_e && a_s <= a_w)
     {
         // get s
         pts[h][w].letter = resolve_point(h+1, w);
     } 
 
     pts[h][w].resolved = 1;
     return pts[h][w].letter;
 }
 
 int main()
 {
     int iT, iH, iW;
     char line[512];
 
     scanf("%d\n", &T);
     
     for (iT = 0; iT < T; iT++)
     {
         scanf("%d %d\n", &H, &W);
 
         for (iH = 0; iH < H; iH++)
         {
             for (iW = 0; iW < W; iW++)
             {
                 scanf("%d", &pts[iH][iW].alt);
                 pts[iH][iW].basin = NULL;
                 pts[iH][iW].resolved = 0;
                 pts[iH][iW].letter = '?';
             }
             fgets(line, 512, stdin);
         }
 
         printf("Case #%d:\n", (iT+1));
         letter = 'a';
 
         for (iH = 0; iH < H; iH++)
         {
             for (iW = 0; iW < W; iW++)
             {
                 char outc;
                 outc = resolve_point(iH, iW);
                 printf("%c ", outc);
             }
             printf("\n");
         }
     }
 }
