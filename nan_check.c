#include <stdio.h> 
#include <stdlib.h> 
#include <inttypes.h>
#include <math.h>
#include <cmath>

int main (int argc, char **argv) {

    union {
        float f;
        uint32_t u;
    } f2u;

    FILE * fp_database = fopen("/mnt/data/data/seismic/data_size100M_seismic_len256_znorm.bin", "r");
    float * sequence_cache = (float *) malloc(sizeof(float) * 256);
    long long length = 4L * 256L;

    fseek(fp_database, length * 1338400L, SEEK_SET);
    fread(sequence_cache, sizeof(float), 256, fp_database);
    for (int i = 0; i < 256; ++i) {
        if (isnan(sequence_cache[i])) {
            f2u.f = sequence_cache[i];
            printf("%f 0x%" PRIx32 "\n", f2u.f, f2u.u);
        }
    }

    // fseek(fp_database, length * 11376159L, SEEK_SET);
    // fread(sequence_cache, sizeof(float), 256, fp_database);
    // for (int i = 0; i < 256; ++i) {
    //     f2u.f = sequence_cache[i];
    //     printf("%f 0x%" PRIx32 "\n", f2u.f, f2u.u);
    // }

    free(sequence_cache);
    fclose(fp_database);
    return(0);
}