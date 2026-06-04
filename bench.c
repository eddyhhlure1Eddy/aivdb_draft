#include "aivdb_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double now_us() {
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e6;
}
static unsigned long process_id(void) {
    return (unsigned long)GetCurrentProcessId();
}
#else
#include <time.h>
#include <unistd.h>
static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}
static unsigned long process_id(void) {
    return (unsigned long)getpid();
}
#endif

static void rand_vec(float *v, uint32_t dim) {
    float n = 0;
    for (uint32_t i = 0; i < dim; i++) {
        v[i] = (float)rand() / RAND_MAX - 0.5f;
        n += v[i] * v[i];
    }
    n = sqrtf(n);
    if (n > 0) for (uint32_t i = 0; i < dim; i++) v[i] /= n;
}

static aivdb_t *make_db(uint32_t nchunks, uint32_t dim) {
    aivdb_t *db = NULL;
    static unsigned counter = 0;
    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "_bench_tmp_%lu_%u.aivdb", process_id(), counter++);
    aivdb_create(tmp_path, dim);
    aivdb_open(tmp_path, &db);
    remove(tmp_path);
    aivdb_load_synthetic(db, nchunks, NULL, 42);
    return db;
}

static void bm_vector(aivdb_t *db, uint32_t topk, int iters) {
    uint32_t dim = aivdb_dim(db);
    uint32_t n = aivdb_chunk_count(db);
    float *q = (float *)malloc(dim * sizeof(float));
    aivdb_hit_t *hits = (aivdb_hit_t *)malloc(topk * sizeof(aivdb_hit_t));
    rand_vec(q, dim);

    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        q[0] += 0.00001f;
        aivdb_search_vector(db, q, topk, hits);
    }
    double t1 = now_us();
    double avg = (t1 - t0) / iters;

    printf("=== BM1: Vector Search ===\n");
    printf("  chunks=%u  dim=%u  topk=%u  iters=%d\n", n, dim, topk, iters);
    printf("  avg = %.1f us  (%.4f ms)\n", avg, avg / 1000.0);
    printf("  QPS = %.0f\n", iters / ((t1 - t0) * 1e-6));
    printf("  >>> %s\n\n", avg < 1000.0 ? "PASS (<1ms)" : "FAIL (>=1ms)");

    free(q);
    free(hits);
}

static void bm_keyword(aivdb_t *db, uint32_t topk, int iters) {
    aivdb_hit_t *hits = (aivdb_hit_t *)malloc(topk * sizeof(aivdb_hit_t));
    const char *qs[] = {"test", "benchmark", "vector", "search"};
    int nq = 4;

    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        aivdb_search_keyword(db, qs[i % nq], topk, hits);
    }
    double t1 = now_us();
    double avg = (t1 - t0) / iters;

    printf("=== BM2: Keyword Search ===\n");
    printf("  chunks=%u  topk=%u  iters=%d\n", aivdb_chunk_count(db), topk, iters);
    printf("  avg = %.1f us  (%.4f ms)\n", avg, avg / 1000.0);
    printf("  QPS = %.0f\n\n", iters / ((t1 - t0) * 1e-6));

    free(hits);
}

static void bm_hybrid(aivdb_t *db, uint32_t topk, int iters) {
    uint32_t dim = aivdb_dim(db);
    float *q = (float *)malloc(dim * sizeof(float));
    aivdb_hit_t *hits = (aivdb_hit_t *)malloc(topk * sizeof(aivdb_hit_t));
    rand_vec(q, dim);
    const char *qs[] = {"test", "benchmark", "vector"};
    int nq = 3;

    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        q[0] += 0.00001f;
        aivdb_search_hybrid(db, qs[i % nq], q, 0.7f, 0.3f, topk, hits);
    }
    double t1 = now_us();
    double avg = (t1 - t0) / iters;

    printf("=== BM3: Hybrid Search ===\n");
    printf("  chunks=%u  dim=%u  topk=%u  iters=%d\n", aivdb_chunk_count(db), dim, topk, iters);
    printf("  avg = %.1f us  (%.4f ms)\n", avg, avg / 1000.0);
    printf("  QPS = %.0f\n\n", iters / ((t1 - t0) * 1e-6));

    free(q);
    free(hits);
}

static void bm_scaling(uint32_t dim, uint32_t topk) {
    uint32_t sizes[] = {100, 500, 1000, 2000, 5000, 10000};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== BM4: Vector Search Scaling (chunks) ===\n");
    printf("  dim=%u  topk=%u\n", dim, topk);
    printf("  %-10s %-12s %-12s %s\n", "Chunks", "Avg(us)", "Avg(ms)", "Status");
    printf("  %-10s %-12s %-12s %s\n", "-----", "------", "------", "------");

    for (int s = 0; s < nsizes; s++) {
        aivdb_t *db = make_db(sizes[s], dim);
        uint32_t d = aivdb_dim(db);
        float *q = (float *)malloc(d * sizeof(float));
        aivdb_hit_t *hits = (aivdb_hit_t *)malloc(topk * sizeof(aivdb_hit_t));
        rand_vec(q, d);

        int iters = 500;
        double t0 = now_us();
        for (int i = 0; i < iters; i++) {
            q[0] += 0.00001f;
            aivdb_search_vector(db, q, topk, hits);
        }
        double t1 = now_us();
        double avg = (t1 - t0) / iters;

        printf("  %-10u %-12.1f %-12.4f %s\n", sizes[s], avg, avg / 1000.0,
               avg < 1000.0 ? "PASS" : "FAIL");

        aivdb_close(db);
        free(q);
        free(hits);
    }
    printf("\n");
}

static void bm_topk(uint32_t nchunks, uint32_t dim) {
    uint32_t topks[] = {5, 10, 20, 50, 100, 200};
    int ntopks = sizeof(topks) / sizeof(topks[0]);

    printf("=== BM5: TopK Scaling ===\n");
    printf("  chunks=%u  dim=%u\n", nchunks, dim);
    printf("  %-10s %-12s %-12s %s\n", "TopK", "Avg(us)", "Avg(ms)", "Status");
    printf("  %-10s %-12s %-12s %s\n", "----", "------", "------", "------");

    aivdb_t *db = make_db(nchunks, dim);
    uint32_t d = aivdb_dim(db);
    float *q = (float *)malloc(d * sizeof(float));
    rand_vec(q, d);

    for (int t = 0; t < ntopks; t++) {
        aivdb_hit_t *hits = (aivdb_hit_t *)malloc(topks[t] * sizeof(aivdb_hit_t));
        int iters = 500;
        double t0 = now_us();
        for (int i = 0; i < iters; i++) {
            q[0] += 0.00001f;
            aivdb_search_vector(db, q, topks[t], hits);
        }
        double t1 = now_us();
        double avg = (t1 - t0) / iters;

        printf("  %-10u %-12.1f %-12.4f %s\n", topks[t], avg, avg / 1000.0,
               avg < 1000.0 ? "PASS" : "FAIL");
        free(hits);
    }
    printf("\n");

    aivdb_close(db);
    free(q);
}

static void bm_dims(uint32_t nchunks, uint32_t topk) {
    uint32_t dims[] = {128, 256, 384, 512, 768, 1024, 1536};
    int ndims = sizeof(dims) / sizeof(dims[0]);

    printf("=== BM6: Dimension Scaling ===\n");
    printf("  chunks=%u  topk=%u\n", nchunks, topk);
    printf("  %-10s %-12s %-12s %s\n", "Dim", "Avg(us)", "Avg(ms)", "Status");
    printf("  %-10s %-12s %-12s %s\n", "---", "------", "------", "------");

    for (int d = 0; d < ndims; d++) {
        aivdb_t *db = make_db(nchunks, dims[d]);
        float *q = (float *)malloc(dims[d] * sizeof(float));
        aivdb_hit_t *hits = (aivdb_hit_t *)malloc(topk * sizeof(aivdb_hit_t));
        rand_vec(q, dims[d]);

        int iters = 500;
        double t0 = now_us();
        for (int i = 0; i < iters; i++) {
            q[0] += 0.00001f;
            aivdb_search_vector(db, q, topk, hits);
        }
        double t1 = now_us();
        double avg = (t1 - t0) / iters;

        printf("  %-10u %-12.1f %-12.4f %s\n", dims[d], avg, avg / 1000.0,
               avg < 1000.0 ? "PASS" : "FAIL");

        aivdb_close(db);
        free(q);
        free(hits);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    uint32_t dim = 768;
    uint32_t nchunks = 1000;
    uint32_t topk = 20;
    int iters = 500;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) nchunks = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iter") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    srand(42);
    printf("AIVDB Benchmark Suite\n");
    printf("=====================\n\n");
    printf("Config: chunks=%u dim=%u topk=%u iters=%d\n\n", nchunks, dim, topk, iters);

    aivdb_t *db = make_db(nchunks, dim);

    bm_vector(db, topk, iters);
    bm_keyword(db, topk, iters);
    bm_hybrid(db, topk, iters);

    aivdb_close(db);

    bm_scaling(dim, topk);
    bm_topk(nchunks, dim);
    bm_dims(nchunks, topk);

    printf("All benchmarks done.\n");
    return 0;
}
