//
 //  main.c
 //  CodeJam2014
 //
 //  Created by Baris Tumerkan on 4/12/14.
 //  Copyright (c) 2014 Baris Tumerkan. All rights reserved.
 //
 
 #import <Cocoa/Cocoa.h>
 
 #import <stdlib.h>
 #import <string.h>
 #import <ctype.h>
 #import <math.h>
 
 #import "QuestionFour.h"
 
 int main(int argc, char *argv[])
 {
     @autoreleasepool {
         FILE* fi, *fo;
         int testCases = 0;
         
         fi = fopen("/Users/baris/Desktop/input.txt", "r");
         fo = fopen("/Users/baris/Desktop/CodeJam2014/Output/OUTPUT.txt", "w");
         
         fscanf(fi, "%d\n", &testCases);
         
         for (int currentCase = 1; currentCase <= testCases; currentCase++) {
             @autoreleasepool {
                 NSString* caseAnswer = [NSString stringWithFormat:@"Case #%d: %@\n", currentCase, solveD(fi, currentCase)];
                 fprintf(fo, "%s", [caseAnswer cStringUsingEncoding:NSUTF8StringEncoding]);
                 NSLog(@"%@", caseAnswer);
             }
         }
         
         fclose(fi);
         fclose(fo);
     }
     
     return 0;
 }
