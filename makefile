CC = gcc
IDIR = include
CFLAGS = -I$(IDIR) -lm -fPIC 
LDFLAGS = -shared -lm
RM = rm

TARGET_LIB_1 = coconut_sample.so
TARGET_LIB_2 = extract_ranks.so

SRCS_1 = src/coconut_sampling.c
SRCS_2 = src/extract_rank.c

.PHONY: all
all: ${TARGET_LIB_1} ${TARGET_LIB_2}

$(TARGET_LIB_1): 
	$(CC) ${LDFLAGS} -o $@ $(SRCS_1) $(CFLAGS)

$(TARGET_LIB_2): 
	$(CC) ${LDFLAGS} -o $@ $(SRCS_2) $(CFLAGS)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB_1} ${TARGET_LIB_2}