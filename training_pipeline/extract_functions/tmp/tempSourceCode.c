 
 int main() {
 
 int no_cases, c;
 int turnover_time;
 int no_Adept, no_Bdept, tA, tB;
 
 
 void read_table(int no, int* dept, int* ariv, int turn) {
 	int i, hd, ha, md, ma;
 
 	for (i=0; i<no; i++) {
 		read_line();
 		sscanf(input, "%2d:%2d %2d:%2d", &hd, &md, &ha, &ma);
 		dept[hd*60+md]--;
 		ariv[ha*60+ma+turn]++;
 	}
 }
 
 int find_start(int* change) {
 	int i, t, at;
 	t=at=0;
 	for (i=0; i<TIMESLOTS; i++) {
 		if (change[i]) {
 			at+=change[i];
 			if (at<0) {
 				t=t-at;
 				at=0;
 			}
 		}
 	}
 	return t;
 }
 
 no_cases=read_number();
 for (c=0; c<no_cases; c++) {
 	memset(Achange, 0, sizeof(Achange));
 	memset(Bchange, 0, sizeof(Bchange));
 	turnover_time=read_number();
 
 	read_line();
 	sscanf(input, "%d %d", &no_Adept, &no_Bdept);
 	read_table(no_Adept, Achange, Bchange, turnover_time);
 	read_table(no_Bdept, Bchange, Achange, turnover_time);
 	tA=find_start(Achange);
 	tB=find_start(Bchange);
 
 	printf("Case #%d: %d %d\n", c+1, tA, tB);
 }
 return 0;
 }