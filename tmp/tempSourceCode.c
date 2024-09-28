#include <stdio.h>
 
 #define true 1
 #define false 0
 
 typedef unsigned char bool;
 
 char mat[55][55], aux[55][55];
 
 void rotate(int size)
 {
    int i, k, j;
 
    for(i = 0; i < size; i++)
       for(k = 0; k < size; k++)
       {
          aux[i][k] = mat[size-k-1][i];
          mat[size-k-1][i] = '.';
       }
 
    /*for(i = 0; i < size; i++, printf("\n"))
       for(k = 0; k < size; k++)
          printf("%c", aux[i][k]);
    printf("\n");*/
 
    for(k = 0; k < size; k++)
       for(i = size-1, j = size-1; i >= 0; i--)
          if(aux[i][k] != '.')
             mat[j--][k] = aux[i][k];
 
    /*for(i = 0; i < size; i++, printf("\n"))
       for(k = 0; k < size; k++)
          printf("%c", mat[i][k]);
    printf("\n");*/
 }
 
 int verifyHoriz(char c, int min, int size)
 {
    int i, k;
    int tot;
 
    for(i = 0; i < size; i++)
    {
       tot = 0;
       for(k = 0; k < size; k++)
          if(mat[i][k] == c)
          {
             tot++;
             if(tot == min)
                return 1;
          }
          else
          {
             tot = 0;
          }
    }
 
    return 0;
 }
 
 int verifyVert(char c, int min, int size)
 {
    int i, k;
    int tot;
 
    for(i = 0; i < size; i++)
    {
       tot = 0;
       for(k = 0; k < size; k++)
          if(mat[k][i] == c)
          {
             tot++;
             if(tot == min)
                return 1;
          }
          else
          {
             tot = 0;
          }
    }
 
    return 0;
 }
 
 int verifyDiag(char c, int min, int size)
 {
    int i, k, j;
    int tot;
 
    for(i = 0; i < size; i++)
    {
       tot = 0;
       for(j = 0; (i+j) < size; j++)
          if(mat[i+j][j] == c)
          {
             tot++;
             if(tot == min)
                return 1;
          }
          else
          {
             tot = 0;
          }
    }
 
    for(k = 0; k < size; k++)
    {
       tot = 0;
       for(j = 0; (k+j) < size; j++)
          if(mat[j][k+j] == c)
          {
             tot++;
             if(tot == min)
                return 1;
          }
          else
          {
             tot = 0;
          }
    }
 
    for(i = 0; i < size; i++)
    {
       tot = 0;
       for(j = 0; (i-j) >= 0; j++)
          if(mat[i-j][j] == c)
          {
             tot++;
             if(tot == min)
                return 1;
          }
          else
          {
             tot = 0;
          }
    }
 
   for(i = 0; i < size; i++)
   {
     tot = 0;
     for(j = 0; i+j < size; j++)
       if(mat[i+j][size-i-j] == c)
       {
         tot++;
         if(tot == min)
           return 1;
       }
       else
       {
         tot = 0;
       }
   }
 
    return 0;
 }
 
 
 int verify(int min, int size)
 {
    int red = 0, blue = 0;
 
    red = (verifyHoriz('R', min, size) == 1) ? 1 : red;
    /*printf("Red %d\n", red);*/
    red = (verifyVert('R', min, size) == 1) ? 1 : red;
    /*printf("Red %d\n", red);*/
    red = (verifyDiag('R', min, size) == 1) ? 1 : red;
    /*printf("Red %d\n", red);*/
 
    blue = (verifyHoriz('B', min, size) == 1) ? 1 : blue;
    /*printf("Blue %d\n", blue);*/
    blue = (verifyVert('B', min, size) == 1) ? 1 : blue;
    /*printf("Blue %d\n", blue);*/
    blue = (verifyDiag('B', min, size) == 1) ? 1 : blue;
    /*printf("Blue %d\n", blue);*/
 
    return 2*red + blue;
 }
 
 int main()
 {
    int T, cont = 0;
 
    scanf("%d", &T);
    while(T--)
    {
       int n, k, ret;
       int i;
 
       scanf("%d %d\n", &n, &k);
       for(i = 0; i < n; i++)
          gets(mat[i]);
 
       rotate(n);
 
       ret = verify(k, n);
 
       if(ret == 0)
          printf("Case #%d: Neither\n", ++cont);
       else if(ret == 1)
          printf("Case #%d: Blue\n", ++cont);
       else if(ret == 2)
          printf("Case #%d: Red\n", ++cont);
       else
          printf("Case #%d: Both\n", ++cont);
    }
 
    return 0;
 }
 
