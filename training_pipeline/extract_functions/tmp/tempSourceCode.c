int is_palindrome(long long int a)
{
     char tmp[100];
     int l,flag = 0,i,j;
     sprintf(tmp, ""%lld"", a);
     l = strlen(tmp);
     //printf(""%d\n"",l);
     for(i=0,j=l-1;i<j;i++,j--)
     {
         if(tmp[i] != tmp[j])
         {
             flag = 1;
             break;
         }
     }
     if(flag == 1)
     return 0;
     return 1;
}
