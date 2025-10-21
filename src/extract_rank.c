#include <stdlib.h>      
#include <stdio.h>      
#include <math.h> 
#include <time.h>
#ifdef VALUES
	#include <values.h>
#endif

#include "sax_breakpoints.h"

int compare_float (const void * a, const void * b) {
    if (* (float *) a < * (float *) b) return -1;
	if (* (float *) a == * (float *) b) return 0;
	if (* (float *) a > * (float *) b) return 1;
}

void generate_adhoc_breakpoints (float * data,
                                 long int dataset_size,
                                 float ** adhoc_breakpoints, 
								 int embedding_length,
								 int sax_segments,
								 int sax_cardinality) {
	if (embedding_length != sax_segments) {
		fprintf(stderr, "error: currently only (embedding_length == sax_segments) is supportted\n");
	}

	float * values_per_segment = malloc(sizeof(float) * dataset_size);
	if (values_per_segment == NULL) {
		fprintf(stderr, "error: could not allocate memory for ad-hoc saxbreakpoints\n");
	}

	int i, j, k;
	int alphabeta_size, num_breakpoints, offset, period_length;
	for (i = 0; i < sax_segments; ++ i) {
		for (j = 0; j < dataset_size; ++ j) {
			values_per_segment[j] = data[i + embedding_length * j];
		}

		qsort(values_per_segment, dataset_size, sizeof(float), compare_float);

		for (j = 1; j <= sax_cardinality; ++ j) {
			alphabeta_size = pow(2, j);
			num_breakpoints = alphabeta_size - 1;
			period_length = dataset_size / alphabeta_size;
			offset = num_breakpoints * (num_breakpoints - 1) / 2;

			for (k = 1; k < alphabeta_size; ++ k) {
				adhoc_breakpoints[i][offset + k] = values_per_segment[k * period_length];
			}
		}
	}
}

typedef struct node { 
    float distance;
    long index;

    struct node * next; 
} Node; 
  
typedef struct priority_queue {
    Node * head;
    Node * tail;

    int size;
    int capacity;
} Queue;

Queue * initialize_queue (int capacity) {
    Queue * tmp = (Queue *) malloc(sizeof(Queue));
    tmp->head = NULL;
    tmp->tail = NULL;
    tmp->size = 0;
    tmp->capacity = capacity;
    return tmp;
}

int push (Queue * queue, int index, float distance) { 
    if (queue->size == queue->capacity && distance >= queue->tail->distance) {
        return -1;
    }

    Node * tmp = (Node *) malloc(sizeof(Node)); 
    tmp->distance = distance;
    tmp->index = index;
    tmp->next = NULL; 

    queue->size += 1;

    if (queue->size == 1) {
        queue->head = tmp;
        queue->tail = tmp;
        return 0;
    }

    Node * iter = queue->head;

    if (iter->distance >= distance) { 
        tmp->next = iter;
        queue->head = tmp;
    } else { 
        while (iter->next != NULL && iter->next->distance < distance) { 
            iter = iter->next; 
        }
  
        tmp->next = iter->next;
        iter->next = tmp;

        if (tmp->next == NULL) {
            queue->tail = tmp;
        }
    } 

    if (queue->size > queue->capacity) {
        iter = queue->head;
        while (iter->next->next != NULL) { 
            iter = iter->next; 
        } 

        tmp = iter->next;
        queue->tail = iter;
        iter->next = NULL;
        queue->size -= 1;

        free(tmp);
    }

    return 1;
} 

typedef struct {
    unsigned char * values;
    long index;
} SAX;

int compare_breakpoints(const void * a, const void * b) {
    float * c = (float *) b - 1;
    if (*(float *) a > *(float *) c && *(float *) a <= *(float *) b) {
        return 0;
    } else if (*(float *) a <= *(float *) c) {
        return -1;
    } else {
        return 1;
    }
}

int sequence2sax (float * sequence, 
                  unsigned char * sax, 
                  float * paa,
                  int len_segment,
                  int size_segments, 
                  int size_alphabeta, 
                  int cardinality) {
    for (int i = 0; i < size_segments; ++ i) {
        paa[i] = 0;

        for (int j = 0; j < len_segment; ++ j) {
            paa[i] += sequence[(i * len_segment) + j];
        }
        paa[i] /= len_segment;
    }

    int offset = (size_alphabeta - 1) * (size_alphabeta - 2) / 2;
    for (int i = 0; i < size_segments; ++ i) {
        sax[i] = 0;
        float * breakpoint = (float *) bsearch (&paa[i], 
                                                &sax_breakpoints[offset], 
                                                size_alphabeta - 1,
                                                sizeof(float), 
                                                compare_breakpoints);
        if (breakpoint != NULL) {
            sax[i] = (int) (breakpoint -  &sax_breakpoints[offset]);
        } else if (paa[i] > 0) {
            sax[i] = size_alphabeta - 1;
        }
    }

    return 1;
}

int embedding2sax (float * embedding,
                   unsigned char * sax,
                   int size_segments, 
                   int size_alphabeta, 
                   int cardinality, 
                   float ** adhoc_breakpoints) {
    int offset = (size_alphabeta - 1) * (size_alphabeta - 2) / 2;
    for (int i = 0; i < size_segments; ++ i) {
        sax[i] = 0;
        float * breakpoint = (float *) bsearch (&embedding[i], 
                                                &adhoc_breakpoints[i][offset], 
                                                size_alphabeta - 1,
                                                sizeof(float), 
                                                compare_breakpoints);
        if (breakpoint != NULL) {
            sax[i] = (int) (breakpoint -  &adhoc_breakpoints[i][offset]);
        } else if (embedding[i] > 0) {
            sax[i] = size_alphabeta - 1;
        }
    }

    return 1;
}

float sax_distance (unsigned char * sax_1, unsigned char * sax_2, int paa_segments, int offset) {
    float distance = 0;

    int sax_value_1, sax_value_2, tmp;
    for (int i = 0; i < paa_segments; ++i) {
        if (sax_1[i] != sax_2[i]) {
            sax_value_1 = (int) sax_1[i];
            sax_value_2 = (int) sax_2[i];

            if (sax_value_1 > sax_value_2) {
                tmp = sax_value_1;
                sax_value_1 = sax_value_2;
                sax_value_2 = tmp;
            }

            distance += (sax_breakpoints[offset + sax_value_2 - 1] - sax_breakpoints[offset + sax_value_1]) 
                        * (sax_breakpoints[offset + sax_value_2 - 1] - sax_breakpoints[offset + sax_value_1]);
        }
    }

    return distance;
}


float sax_distance_adhoc (unsigned char * sax_1, unsigned char * sax_2, int paa_segments, int offset, float ** adhoc_breakpoints) {
    float distance = 0;

    int sax_value_1, sax_value_2, tmp;
    for (int i = 0; i < paa_segments; ++i) {
        if (sax_1[i] != sax_2[i]) {
            sax_value_1 = (int) sax_1[i];
            sax_value_2 = (int) sax_2[i];

            if (sax_value_1 > sax_value_2) {
                tmp = sax_value_1;
                sax_value_1 = sax_value_2;
                sax_value_2 = tmp;
            }

            distance += (adhoc_breakpoints[i][offset + sax_value_2 - 1] - adhoc_breakpoints[i][offset + sax_value_1]) 
                        * (adhoc_breakpoints[i][offset + sax_value_2 - 1] - adhoc_breakpoints[i][offset + sax_value_1]);
        }
    }

    return distance;
}

float euclidean_distance(float * series_1, float * series_2, int length) {
    float distance = 0;
    
    for (int i = 0; i < length; ++i) {
        distance += (series_1[i] - series_2[i]) * (series_1[i] - series_2[i]);
    }

    return sqrt(distance);
}


int generate_nn_ranked (char * path_database, 
                        char * path_embedding, 
                        long size_database, 
                        char * path_query, 
                        char * path_query_embedding, 
                        long size_query, 
                        char * path_ranks, 
                        int size_ranks,
                        int len_sequence, 
                        int paa_segments, 
                        int sax_cardinality) {
    int len_embedding = paa_segments;

    FILE * fp_database = fopen(path_database, "rb");
    FILE * fp_embedding = fopen(path_database, "rb");

    float * database = malloc(sizeof(float) * len_sequence * size_database);
    float * embeddings = malloc(sizeof(float) * len_embedding * size_database);

    int num_database = fread(database, sizeof(float), len_sequence * size_database, fp_database);
    int num_embedding = fread(embeddings, sizeof(float), len_embedding * size_database, fp_embedding);

    float ** adhoc_breakpoints = malloc(sizeof(float *) * paa_segments);
    for (int i = 0; i < paa_segments; ++ i) {
        adhoc_breakpoints[i] = malloc(sizeof(float) * BREAKPOINTS_LIST_LENGTH);
        for (int j = 0; j < BREAKPOINTS_LIST_LENGTH; ++ j) {
            adhoc_breakpoints[i][j] = 0;
        }
    }

    generate_adhoc_breakpoints(embeddings, size_database, adhoc_breakpoints, len_embedding, paa_segments, sax_cardinality);

    SAX * saxs_embedding = (SAX *) malloc(sizeof(SAX) * size_database);
    SAX * saxs_paa = (SAX *) malloc(sizeof(SAX) * size_database);

    for (int i = 0; i < size_database; ++ i) {
        saxs_embedding[i].values = (unsigned char *) malloc(sizeof(unsigned char) * paa_segments);
        saxs_paa[i].values = (unsigned char *) malloc(sizeof(unsigned char) * paa_segments);
    }

    float * paa_cache = malloc(sizeof(float) * paa_segments);

    int len_segment = len_sequence / paa_segments;
    int size_alphabeta = (int) pow(2, sax_cardinality);

    for (long i = 0; i < size_database; ++ i) {
        saxs_paa[i].index = i;
        saxs_embedding[i].index = i;

        sequence2sax(&(database[i * len_sequence]), saxs_paa[i].values, paa_cache, len_segment, paa_segments, size_alphabeta, sax_cardinality);
        embedding2sax(&(embeddings[i * len_embedding]), saxs_embedding[i].values, paa_segments, size_alphabeta, sax_cardinality, adhoc_breakpoints);
    }

    FILE * fp_query = fopen(path_query, "rb");
    FILE * fp_query_embedding = fopen(path_query_embedding, "rb");

    float * series_cache = malloc(sizeof(float) * len_sequence);
    float * embedding_cache = malloc(sizeof(float) * len_embedding);

    unsigned char * sax_paa_query = (unsigned char *) malloc(sizeof(unsigned char) * paa_segments); 
    unsigned char * sax_embedding_query = (unsigned char *) malloc(sizeof(unsigned char) * paa_segments); 

    int offset = ((size_alphabeta - 1) * (size_alphabeta - 2)) / 2;

    FILE * fp_ranks = fopen(path_ranks, "w");

    for (int i = 0; i < size_query; ++ i) {
        fread(series_cache, sizeof(float), len_sequence, fp_query);
        fread(embedding_cache, sizeof(float), len_embedding, fp_query_embedding);

        sequence2sax(series_cache, sax_paa_query, paa_cache, len_segment, paa_segments, size_alphabeta, sax_cardinality);
        embedding2sax(embedding_cache, sax_embedding_query, paa_segments, size_alphabeta, sax_cardinality, adhoc_breakpoints);

        Queue * queue = initialize_queue(size_ranks);
        for (int j = 0; j < size_database; ++j) {
            push(queue, j, sax_distance(sax_paa_query, saxs_paa[j].values, paa_segments, offset));
        }

        Node * prev;
        Node * iter = queue->head;
        while (iter != NULL) {
            fprintf(fp_ranks, "%f", euclidean_distance(&(database[iter->index * len_sequence]), series_cache, len_sequence));
            if (iter->next != NULL) {
                fprintf(fp_ranks, " ");
            } else {
                fprintf(fp_ranks, "\n");
            }
            prev = iter;
            iter = iter->next;
            free(prev);
        }
        free(queue);

        queue = initialize_queue(size_ranks);
        for (int j = 0; j < size_database; ++j) {
            push(queue, j, sax_distance_adhoc(sax_embedding_query, saxs_embedding[j].values, paa_segments, offset, adhoc_breakpoints));
        }

        iter = queue->head;
        while (iter != NULL) {
            fprintf(fp_ranks, "%f", euclidean_distance(&(database[iter->index * len_sequence]), series_cache, len_sequence));
            if (iter->next != NULL) {
                fprintf(fp_ranks, " ");
            } else {
                fprintf(fp_ranks, "\n");
            }
            prev = iter;
            iter = iter->next;
            free(prev);
        }
        free(queue);

        srand(time(NULL));
        for (int j = 0; j < size_ranks; ++j) {
            fprintf(fp_ranks, "%f", euclidean_distance(&(database[(rand() % size_database) * len_sequence]), series_cache, len_sequence));
            if (j + 1 != size_ranks) {
                fprintf(fp_ranks, " ");
            } else {
                fprintf(fp_ranks, "\n");
            }
        }
    }

    free(database);
    free(embeddings);

    free(saxs_embedding);
    free(saxs_paa);

    free(series_cache);
    free(embedding_cache);
    free(paa_cache);

    free(sax_paa_query);
    free(sax_embedding_query);

    for (int i = 0; i < paa_segments; ++ i) {
        free(adhoc_breakpoints[i]);
    }
    free(adhoc_breakpoints);

    fclose(fp_database);
    fclose(fp_embedding);
    fclose(fp_query);
    fclose(fp_query_embedding);
    fclose(fp_ranks);

    return 0;
}

int main (int argc, char **argv) {
    return generate_nn_ranked ("/mnt/hddhelp/data/archive/seismic/data_size100M_seismic_len256_znorm.bin", 
                               "/mnt/data/qwang/ts-embedding/embedded/WaveAlex-alt-16-coconut-16-100k-seismic-256-1m.bin", 
                               1000000L, 
                               "/mnt/hddhelp/data/archive/seismic/queries_size1K_seismic_len256_znorm.bin", 
                               "/mnt/data/qwang/ts-embedding/embedded/WaveAlex-alt-16-coconut-16-100k-seismic-256-1m-1k.bin", 
                               1L, 
                               "/home/qwang/projects/ts-embedding/indexing/tmp/WaveAlex-alt-16-coconut-16-100k-seismic-256-1m-1k-1.csv", 
                               1000,
                               256, 
                               16, 
                               8);
}