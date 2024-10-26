#include<stdio.h>
#include<stdlib.h>
#include<string.h>


int main(){
	int i,j,n,c;
	scanf("%d",&n);
	c=n;
	char **s=(char**)malloc(sizeof(char**)*n);
	for(i=0;i<n;i++)
	s[i]=(char*)malloc(sizeof(char*)*10);
	
	char **s1=(char**)malloc(sizeof(char**)*n);
	for(i=0;i<n;i++)
	s1[i]=(char*)malloc(sizeof(char*)*10);
	
	for(i=0;i<n;i++){
		scanf("%s",s[i]);
	}
	for(i=0;i<n;i++){
		scanf("%s",s1[i]);
	}
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if(strcmp(s[i],s1[j])==0){
			   strcpy(s1[j],"xyz");	
			c--;
			break;
	}
		}
	}
	printf("%d",c);
	
	return 0;
}