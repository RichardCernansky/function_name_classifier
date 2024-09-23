#include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 
 double R, t, g, f, r, Rmt;
 struct point {double x; double y;};
 struct polygon
 {
   int type;
   int npoints;
   struct point points[6];
 };
 
 int checkdistance(struct point a);
 struct polygon clip(struct point a);
 double computearcarea(struct point a, struct point b);
 double computearea(struct polygon p);
 double computesquarearea();
 struct polygon createsquare(struct point a);
 
 int main(int argc, char** argv)
 {
   int ncases;
   //float a, b, c, d,e;
   //f = 1; R = 100; t = 1, r = 1, g = 10;
   scanf("%d", &ncases);
   int i;
   for (i = 0; i < ncases; ++i)
   {
     scanf("%lf %lf %lf %lf %lf", &f, &R, &t, &r, &g);
     double circlearea = M_PI * R * R;
     t += f;
     g = g - 2 * f;
     r += f;
     Rmt = R - t;
     printf("Case #%d: ", i + 1);
     printf("%.6f\n", (circlearea - computesquarearea()) / circlearea);
     //printf("%f %f %f %f %f\n", f, b, c, d,e);
   }
   return (EXIT_SUCCESS);
 }
 
 int checkdistance(struct point a)
 {
   if (a.x * a.x + a.y * a.y >= Rmt * Rmt) return 1; // outside
   else return -1; // inside
 }
 
 struct polygon clip(struct point a)
 {
   int type;
   int npoints = 0;
   struct polygon retval;
   struct polygon square = createsquare(a);
 
   int i;
   for (i = 0; i <= 3; ++i)
   {
     int current = checkdistance(square.points[i]);
     int next = checkdistance(square.points[(i + 1) % 4]);
     if (current < 0)
     {
       retval.points[npoints].x = square.points[i].x;
       retval.points[npoints].y = square.points[i].y;
       npoints++;
     }
 
     if (current * next < 0)
     {
       if (i % 2 == 0)
       {
         retval.points[npoints].x = square.points[i].x;
         retval.points[npoints].y = sqrt(Rmt * Rmt - square.points[i].x * square.points[i].x);
       } else
       {
         retval.points[npoints].x = sqrt(Rmt * Rmt - square.points[i].y * square.points[i].y);
         retval.points[npoints].y = square.points[i].y;
       }
       npoints++;
     }
   }
 
   if (npoints == 3) type = 2;
   else if (npoints == 5) type = 5;
   else
   {
     if (retval.points[1].y > retval.points[2].y) type = 3;
     else if (retval.points[3].x > retval.points[2].x) type = 4;
     else type = 1;
   }
 
   retval.npoints = npoints;
   retval.type = type;
   return retval;
 }
 
 double computearcarea(struct point a, struct point b)
 {
   struct point midpoint;
   midpoint.x = (a.x + b.x) / 2;
   midpoint.y = (a.y + b.y) / 2;
 
   double h = sqrt(midpoint.x * midpoint.x + midpoint.y * midpoint.y);
   double d = sqrt((a.x - midpoint.x) * (a.x - midpoint.x) + (a.y - midpoint.y) * (a.y - midpoint.y));
   double alpha = atan(d / h);
   return alpha * Rmt * Rmt - d * h;
 }
 
 double computearea(struct polygon p)
 {
   double retval;
   switch (p.type)
   {
     case 1:
       retval = g * g;
       break;
     case 2:
       retval = 0.5 * (p.points[1].y - p.points[0].y) * (p.points[2].x - p.points[0].x) +
               computearcarea(p.points[1], p.points[2]);
       break;
     case 3:
       retval = 0.5 * (p.points[1].y - p.points[0].y + p.points[2].y - p.points[3].y) * g +
               computearcarea(p.points[1], p.points[2]);
       break;
     case 4:
       retval = 0.5 * (p.points[2].x - p.points[1].x + p.points[3].x - p.points[0].x) * g +
               computearcarea(p.points[2], p.points[3]);
       break;
     case 5:
       retval = (p.points[2].x - p.points[1].x) * g + 0.5 * (p.points[3].y - p.points[4].y + g) *
               (p.points[4].x - p.points[2].x) + computearcarea(p.points[2], p.points[3]);
       break;
   }
   return retval;
 }
 
 double computesquarearea()
 {
   double retval = 0;  
   struct point currentpoint;
   currentpoint.x = r;
   currentpoint.y = r;
   while (checkdistance(currentpoint) < 0)
   {    
     while (checkdistance(currentpoint) < 0)
     {
       retval += computearea(clip(currentpoint));
       currentpoint.y = currentpoint.y + (g + 2 * r);
     }
     currentpoint.x = currentpoint.x + (g + 2 * r);
     currentpoint.y = r;
   }
   return retval * 4;
 }
 
 struct polygon createsquare(struct point a)
 {
   struct polygon retval;
   retval.type = 1; // square
   retval.npoints = 4;
   retval.points[0].x = a.x;
   retval.points[0].y = a.y;
   retval.points[1].x = a.x;
   retval.points[1].y = a.y + g;
   retval.points[2].x = a.x + g;
   retval.points[2].y = a.y + g;
   retval.points[3].x = a.x + g;
   retval.points[3].y = a.y;
   return retval;
 }
 
 
