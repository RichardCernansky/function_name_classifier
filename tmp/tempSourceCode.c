#include <stdio.h>
 #include <memory.h>
 #define MAXARR 10000
 long n;
 long base[20];
 int sumbase;
 long arr[MAXARR];
 long arrsum;
 void main()
 {
 	int i,j,k,p,q;
 	int sum;
 	char c;
 	int yu;
 	int flag;
 	int f,f2;
 
 
 	freopen("input.txt", "rt", stdin);
     freopen("output.txt", "wt", stdout);
 
 
 	scanf("%d",&n);
 	for (i=1;i<=n;i++)
 	{
 		sumbase=0;
 		c=0;
 		c=getchar();
 		while (c=='\n'||c=='\r') c=getchar();
 		while (c!='\n'&&c!='\r')
 		{
 			if (c=='\n'||c=='\r')break;
 			while(c==' ') c=getchar();
 			if (c=='\n'||c=='\r') break;
 			base[sumbase]=c-48;
 			if (c==49) 
 			{
 				if ((c=getchar())=='0') base[sumbase]=10;
 			}
 			sumbase++;
 			c=getchar();
 		}
 		//2
 		for (j=2;1;j++)//ÿ
 		{
 			flag=1;
 			for (k=0;k<sumbase;k++)//ÿbase
 			{
 				sum=j;					
 				memset(arr,sizeof(arr),0);
 				arrsum=0;
 				f2=1;
 				while (f2&flag&&sum!=1)
 				{
 					yu=0;
 					p=sum;
 					sum=0;
 					while (p>0)
 					{
 						yu=p%base[k];
 						sum+=yu*yu; p=p/base[k];
 					}
 					if (sum==1) {f2=0;continue;}
 					else {
 						f=1;
 						for (q=0;q<arrsum;q++)
 						{
 							if (arr[q]==sum) {f=0;break;}
 						}
 						if (f) {
 							arr[arrsum]=sum;
 							arrsum++;
 						}
 						else {
 							flag=0;
 							break;
 						}
 					}
 				}
 			}
 			if (flag) {printf("Case #%d: %d\n",i,j);break;}
 		}
 	}
 }
