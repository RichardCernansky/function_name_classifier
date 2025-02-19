 int main(int argc, char** argv) {
     unsigned tests, total_cores, required_cores;
     double trainings;
     double* cores;
 
     unsigned t;
 
     get_tests(&tests);
 
     for (t = 0; t < tests; ++t) {
         get_cores(&total_cores, &required_cores);
         get_trainings(&trainings);
 
         cores = malloc(sizeof (double) * total_cores);
         get_probabilities(cores, total_cores);
 
         give_aid_to_cores(cores, total_cores, trainings);
 
         printf("Case #%d: %f\n", t + 1, give_final_result(cores, total_cores));
 
         free(cores);
     }
 
     return (EXIT_SUCCESS);
 }