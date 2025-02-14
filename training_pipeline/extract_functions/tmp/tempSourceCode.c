 
 int main(int argc , char *argv[])
 {
 	FILE *ifp;
 
 	int i,j,k,l,ncase,nserver,nswitch=0,nq,count;
 	char querry[102];
 	if(argc != 2)
 	{
 		printf("Error:\n");
 		printf("Usage: filename <input file>\n");
 		printf("Example: A-small.out A-small.in\n");
 		exit(1);
 	}
 	ifp = fopen(argv[1] , "r");
 	if( ifp == NULL )
 	{
 		printf("error in opening file");
 	}
 
 	// Read from the file 
 	
 	fscanf(ifp,"%d",&ncase);
 	Server *sname = NULL;
 	for(i = 0; i < ncase ; i++)
 	{
 		fscanf(ifp,"%d",&nserver);
 		fgetc(ifp);
 		sname = (Server *) calloc(nserver,sizeof(Server));
 		//. create memory for server names
 		nswitch=0;
 		count=nserver;
 		for( j = 0 ; j < nserver ;j++)
 		{
 			
 			fgets(sname[j].servername,101,ifp);
 			sname[j].flag = 0;
 		}
 
 		fscanf(ifp,"%d",&nq);
 		fgetc(ifp);
 		for( j = 0 ; j < nq ; j++)
 		{
 			fgets(querry,101,ifp);
 						
 			for(k = 0 ; k < nserver ; k++)
 			{
 				if(strcmp(querry,sname[k].servername) == 0)
 				{
 					if(sname[k].flag==0)
 					{
 						sname[k].flag++;
 						count--;
 						if(count==0)
 						{
 							nswitch++;
 							count = nserver-1;
 							for(l=0;l<nserver;l++)
 							{
 								if(l!=k)
 								{
 									sname[l].flag = 0;
 								}
 							}
 							
 						}
 					}
 					
 					break;
 				}
 			}
 		}
 		printf("Case #%d: %d\n",i+1,nswitch); 
 		free(sname);
 	}
 	return 0;
 }