#pragma GCC optimize("O3")
#define _GNU_SOURCE
#include <ctype.h>
#define _IMMINTRIN_H_INCLUDED
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <fmaintrin.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================== Minimal NumPy BitGenerator ABI ==================== */
typedef struct bitgen {
    void *state;
    uint64_t (*next_uint64)(void *st);
    uint32_t (*next_uint32)(void *st);
    double (*next_double)(void *st);
    uint64_t (*next_raw)(void *st);
} bitgen_t;

extern void random_standard_uniform_fill(bitgen_t *, intptr_t, double *);
extern void random_standard_normal_fill(bitgen_t *, intptr_t, double *);

/* ==================== OpenBLAS (public CBLAS, same calls numpy makes) ==================== */
/* Looked up by name so it works with numpy 1.x wheels (cblas_*64_), numpy 2.x wheels
   (scipy_cblas_*64_) and a 32-bit-int system OpenBLAS (cblas_*). */
typedef void (*dgemv64_fn)(int order, int trans, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, const double *x, int64_t incx,
                           double beta, double *y, int64_t incy);
typedef void (*dgemv32_fn)(int order, int trans, int m, int n, double alpha,
                           const double *a, int lda, const double *x, int incx,
                           double beta, double *y, int incy);
typedef double (*ddot64_fn)(int64_t n, const double *x, int64_t incx, const double *y, int64_t incy);
typedef double (*ddot32_fn)(int n, const double *x, int incx, const double *y, int incy);
typedef double (*exact_u_fn)(int64_t seed_id, int rnd, int64_t idx);

static dgemv64_fn g_dgemv64 = NULL;
static ddot64_fn g_ddot64 = NULL;
static dgemv32_fn g_dgemv32 = NULL;
static ddot32_fn g_ddot32 = NULL;

static int ensure_blas(const char *openblas_path) {
    if ((g_dgemv64 && g_ddot64) || (g_dgemv32 && g_ddot32)) return 0;
    void *h = NULL;
    if (openblas_path && openblas_path[0]) {
        h = dlopen(openblas_path, RTLD_LAZY | RTLD_NOLOAD);
        if (!h) h = dlopen(openblas_path, RTLD_LAZY);
    }
    if (!h) h = RTLD_DEFAULT;
    static const char *pre[] = {"scipy_", "", NULL};
    char name[64];
    for (int i = 0; pre[i] && !(g_dgemv64 && g_ddot64); i++) {
        snprintf(name, sizeof name, "%scblas_dgemv64_", pre[i]);
        g_dgemv64 = (dgemv64_fn)dlsym(h, name);
        snprintf(name, sizeof name, "%scblas_ddot64_", pre[i]);
        g_ddot64 = (ddot64_fn)dlsym(h, name);
    }
    if (g_dgemv64 && g_ddot64) return 0;
    g_dgemv64 = NULL; g_ddot64 = NULL;
    g_dgemv32 = (dgemv32_fn)dlsym(h, "cblas_dgemv");
    g_ddot32 = (ddot32_fn)dlsym(h, "cblas_ddot");
    return (g_dgemv32 && g_ddot32) ? 0 : -1;
}

static inline double blas_ddot(int64_t n, const double *x, const double *y) {
    return g_ddot64 ? g_ddot64(n, x, 1, y, 1) : g_ddot32((int)n, x, 1, y, 1);
}

/* y = A x for a C-contiguous rows x cols A, via cblas_dgemv(ColMajor, Trans, ...) exactly as
   numpy's matmul calls it, so results stay bit-identical to `A @ x`. */
static inline void matvec_rowmajor(int64_t rows, int64_t cols, const double *A, const double *x, double *y, double *blas_buf) {
    (void)blas_buf;
    if (rows == 1) {
        y[0] = blas_ddot(cols, A, x);
    } else if (rows > 1) {
        memset(y, 0, (size_t)rows * sizeof(double));
        if (g_dgemv64) g_dgemv64(102, 112, cols, rows, 1.0, A, cols, x, 1, 0.0, y, 1);
        else g_dgemv32(102, 112, (int)cols, (int)rows, 1.0, A, (int)cols, x, 1, 0.0, y, 1);
    }
}

/* ==================== Part 1: Fast Monte Carlo Simulator ==================== */

static inline double fast_log(double x) {
    union { double d; uint64_t u; } vx = { x };
    int64_t exp_bits = (int64_t)((vx.u >> 52) & 0x7ffULL) - 1023;
    vx.u = (vx.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m = vx.d;
    int gt = (m > 1.4142135623730951);
    if (gt) {
        m *= 0.5;
        exp_bits++;
    }
    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;
    double poly = 0.09090909090909091;
    poly = poly * t2 + 0.1111111111111111;
    poly = poly * t2 + 0.14285714285714285;
    poly = poly * t2 + 0.2;
    poly = poly * t2 + 0.3333333333333333;
    poly = poly * t2 + 1.0;
    return (double)exp_bits * 0.69314718055994530942 + 2.0 * t * poly;
}

static inline double fast_exp(double x) {
    if (x < -40.0) return 0.0;
    if (x > 40.0) return 2.3538526683701998e17;
    double kf = floor(x * 1.4426950408889634074 + 0.5);
    int64_t k = (int64_t)kf;
    double r = x - kf * 0.69314718055994530942;
    double p = 1.3888888888888889e-3;
    p = p * r + 8.3333333333333333e-3;
    p = p * r + 4.1666666666666667e-2;
    p = p * r + 0.16666666666666666;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    union { uint64_t u; double d; } scale;
    scale.u = (uint64_t)(k + 1023) << 52;
    return p * scale.d;
}


static inline __m256d fast_log_pd4(__m256d x) {
    __m256i ux = _mm256_castpd_si256(x);
    __m256i exp_i = _mm256_sub_epi64(
        _mm256_and_si256(_mm256_srli_epi64(ux, 52), _mm256_set1_epi64x(0x7ffLL)),
        _mm256_set1_epi64x(1023LL)
    );
    __m256i mant_i = _mm256_or_si256(
        _mm256_and_si256(ux, _mm256_set1_epi64x(0x000fffffffffffffLL)),
        _mm256_set1_epi64x(0x3ff0000000000000LL)
    );
    __m256d m = _mm256_castsi256_pd(mant_i);
    __m256d gt = _mm256_cmp_pd(m, _mm256_set1_pd(1.4142135623730951), _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), gt);
    exp_i = _mm256_sub_epi64(exp_i, _mm256_castpd_si256(gt));
    __m128i e32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(exp_i, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)));
    __m256d exp_d = _mm256_cvtepi32_pd(e32);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d t = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    __m256d t2 = _mm256_mul_pd(t, t);
    __m256d poly = _mm256_set1_pd(0.09090909090909091);
    poly = _mm256_fmadd_pd(poly, t2, _mm256_set1_pd(0.1111111111111111));
    poly = _mm256_fmadd_pd(poly, t2, _mm256_set1_pd(0.14285714285714285));
    poly = _mm256_fmadd_pd(poly, t2, _mm256_set1_pd(0.2));
    poly = _mm256_fmadd_pd(poly, t2, _mm256_set1_pd(0.3333333333333333));
    poly = _mm256_fmadd_pd(poly, t2, one);
    return _mm256_fmadd_pd(exp_d, _mm256_set1_pd(0.69314718055994530942),
                           _mm256_mul_pd(_mm256_add_pd(t, t), poly));
}

static inline __m256d fast_exp_pd4(__m256d x) {
    __m256d xc = _mm256_max_pd(_mm256_set1_pd(-40.0), _mm256_min_pd(_mm256_set1_pd(40.0), x));
    __m256d kf = _mm256_floor_pd(_mm256_fmadd_pd(xc, _mm256_set1_pd(1.4426950408889634074), _mm256_set1_pd(0.5)));
    __m256d r = _mm256_fnmadd_pd(kf, _mm256_set1_pd(0.69314718055994530942), xc);
    __m256d p = _mm256_set1_pd(1.3888888888888889e-3);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(8.3333333333333333e-3));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(4.1666666666666667e-2));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.16666666666666666));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.5));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    __m128i k32 = _mm256_cvtpd_epi32(kf);
    __m256i k64 = _mm256_cvtepi32_epi64(k32);
    __m256i scale_i = _mm256_slli_epi64(_mm256_add_epi64(k64, _mm256_set1_epi64x(1023LL)), 52);
    __m256d res = _mm256_mul_pd(p, _mm256_castsi256_pd(scale_i));
    __m256d under = _mm256_cmp_pd(x, _mm256_set1_pd(-40.0), _CMP_LT_OQ);
    return _mm256_andnot_pd(under, res);
}

static inline double exact_p_ct(
    double eq_c, double eq_t_val, int pistol, int ot, double s_ct, int map_idx,
    const double *rm_coef, double rm_inter
) {
    double row[13] = {0.0};
    row[0] = pistol ? 0.0 : log((eq_c + 200.0) / (eq_t_val + 200.0));
    row[1] = (double)pistol;
    row[2] = (double)ot;
    row[3] = s_ct;
    if (map_idx >= 0 && map_idx < 9) row[4 + map_idx] = 1.0;
    double dot = blas_ddot(13, row, rm_coef);
    return 1.0 / (1.0 + exp(-(dot + rm_inter)));
}

typedef struct {
    int64_t m;
    const double *xp;
    const double *fp;
    double slopes[64];
    int16_t bucket[72];
} BuyTable;

static inline void init_buy_table(BuyTable *bt, int64_t m, const double *xp, const double *fp) {
    bt->m = m;
    bt->xp = xp;
    bt->fp = fp;
    if (m > 0 && m <= 64) {
        for (int64_t j = 0; j < m - 1; ++j) {
            bt->slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        }
        for (int b = 0; b < 72; ++b) {
            double x_low = b * 250.0;
            int64_t lo = 0;
            while (lo + 1 < m - 1 && xp[lo + 1] <= x_low) lo++;
            bt->bucket[b] = (int16_t)lo;
        }
    }
}

static inline void buy_vec(int64_t m_active, const double * restrict res, const BuyTable *bt, double * restrict out) {
    int64_t m = bt->m;
    const double * restrict xp = bt->xp;
    const double * restrict fp = bt->fp;
    const double * restrict slopes = bt->slopes;
    const int16_t * restrict bucket = bt->bucket;
    double x0 = xp[0], xm = xp[m - 1], f0 = fp[0], fm = fp[m - 1];

    if (m <= 64) {
        for (int64_t i = 0; i < m_active; ++i) {
            double x = res[i];
            double val;
            if (x <= x0) {
                val = f0;
            } else if (x >= xm) {
                val = fm;
            } else {
                int b = (int)(x * 0.004);
                int64_t lo = (b >= 0 && b < 72) ? bucket[b] : 0;
                while (lo + 1 < m - 1 && xp[lo + 1] <= x) lo++;
                int64_t hi = lo + 1;
                if (x == xp[hi]) {
                    val = fp[hi];
                } else {
                    val = fp[lo] + slopes[lo] * (x - xp[lo]);
                    if (isnan(val)) val = fp[lo];
                }
            }
            out[i] = val < x ? val : x;
        }
    } else {
        for (int64_t i = 0; i < m_active; ++i) {
            double x = res[i];
            double val;
            if (x <= x0) val = f0;
            else if (x >= xm) val = fm;
            else {
                int64_t lo = 0, hi = m - 1;
                while (hi - lo > 1) {
                    int64_t mid = (lo + hi) >> 1;
                    if (xp[mid] > x) hi = mid;
                    else lo = mid;
                }
                if (x == xp[hi]) val = fp[hi];
                else {
                    double sl = (fp[hi] - fp[lo]) / (xp[hi] - xp[lo]);
                    val = fp[lo] + sl * (x - xp[lo]);
                    if (isnan(val)) val = fp[lo];
                }
            }
            out[i] = val < x ? val : x;
        }
    }
}

static inline void next_res_vec(
    int64_t m_active, double * restrict res, const double * restrict eq, const uint8_t * restrict won, const double * restrict losses,
    const double * restrict coef, double inter, double * restrict Z3, double * restrict base, double * restrict blas_buf
) {
    for (int64_t i = 0; i < m_active; ++i) {
        double w = (double)won[i];
        double r = res[i];
        double e = eq[i];
        Z3[3 * i + 0] = r - e;
        Z3[3 * i + 1] = e * w;
        Z3[3 * i + 2] = w;
    }
    matvec_rowmajor(m_active, 3, Z3, coef, base, blas_buf);
    for (int64_t i = 0; i < m_active; ++i) {
        double b = base[i] + inter;
        double ls = losses[i] < 4.0 ? losses[i] : 4.0;
        double bonus = (won[i] == 1) ? 0.0 : ((1400.0 + 500.0 * ls) - 2400.0);
        double v = b + bonus;
        if (v < 0.0) v = 0.0;
        else if (v > 22000.0) v = 22000.0;
        res[i] = v;
    }
}

static double *g_buf = NULL;
static int64_t g_cap = 0;

static double *get_scratch(int64_t n) {
    if (n > g_cap) {
        free(g_buf);
        g_buf = NULL;
        if (posix_memalign((void **)&g_buf, 64, (size_t)(24 * n + 512) * sizeof(double)) != 0) return NULL;
        g_cap = n;
    }
    return g_buf;
}

static int run_sim_engine(
    bitgen_t *bg, const double *z_pre, const double *u_pre, const double *logit_u_pre,
    exact_u_fn u_cb, int64_t seed_id,
    int map_idx, double str_a_scalar, int a_ct_first, int64_t n, double sigma,
    const double *rm_coef, double rm_inter,
    const BuyTable *bt_ct, const BuyTable *bt_t,
    const double *tc_ct, double tr_ct_inter,
    const double *tc_t, double tr_t_inter,
    int mode,
    double *out_p_win, double *out_lr_mean,
    int64_t *ext_lr, int64_t *ext_sa, int64_t *ext_sb,
    double *out_h1, int *has_round, double *win_round_p
) {
    double *buf = get_scratch(n);
    if (!buf) return -1;

    double *str_a    = buf;
    double *res_a    = str_a + n;
    double *res_b    = res_a + n;
    double *loss_a   = res_b + n;
    double *loss_b   = loss_a + n;
    double *eq_a     = loss_b + n;
    double *eq_b     = eq_a + n;
    double *u_buf    = eq_b + n;
    double *tmp      = u_buf + n;
    double *Z3       = tmp + n;
    double *blas_buf = Z3 + 3 * n;
    int64_t *sa      = (int64_t *)(blas_buf + 2 * n + 256);
    int64_t *sb      = sa + n;
    int64_t *act_idx = sb + n;
    uint8_t *a_won   = (uint8_t *)(act_idx + n);
    uint8_t *b_won   = a_won + n;

    if (z_pre) {
        for (int64_t i = 0; i < n; ++i) {
            str_a[i] = str_a_scalar + sigma * z_pre[i];
        }
    } else {
        random_standard_normal_fill(bg, (intptr_t)n, str_a);
        for (int64_t i = 0; i < n; ++i) {
            str_a[i] = str_a_scalar + sigma * str_a[i];
        }
    }
    memset(sa, 0, (size_t)n * sizeof(int64_t));
    memset(sb, 0, (size_t)n * sizeof(int64_t));
    for (int64_t i = 0; i < n; ++i) {
        act_idx[i] = i;
    }

    if (mode == 3) {
        memset(has_round, 0, 61 * sizeof(int));
        out_h1[0] = 0.0;
        out_h1[1] = 0.0;
    }

    double map_term = (map_idx >= 0 && map_idx < 9) ? rm_coef[4 + map_idx] : 0.0;
    double c0 = rm_coef[0], c1 = rm_coef[1], c2 = rm_coef[2], c3 = rm_coef[3];

    int64_t act_n = n;
    int rnd = 0;
    while (act_n > 0 && rnd < 60) {
        rnd++;
        int a_ct;
        if (rnd <= 24) {
            a_ct = (a_ct_first != 0) == (rnd <= 12);
            if (rnd == 1 || rnd == 13) {
                for (int64_t j = 0; j < act_n; ++j) {
                    res_a[j] = 800.0;
                    res_b[j] = 800.0;
                    loss_a[j] = 1.0;
                    loss_b[j] = 1.0;
                }
            }
        } else {
            int k = (rnd - 25) % 6;
            a_ct = (a_ct_first != 0) == (((rnd - 25) / 3) % 2 == 1);
            if (k == 0 || k == 3) {
                for (int64_t j = 0; j < act_n; ++j) {
                    res_a[j] = 12500.0;
                    res_b[j] = 12500.0;
                }
            }
        }
        int pistol = (rnd == 1 || rnd == 13) ? 1 : 0;
        int ot = (rnd > 24) ? 1 : 0;

        double *res_ct = a_ct ? res_a : res_b;
        double *res_t  = a_ct ? res_b : res_a;
        double *eq_ct  = a_ct ? eq_a : eq_b;
        double *eq_t   = a_ct ? eq_b : eq_a;

        if (pistol) {
            for (int64_t j = 0; j < act_n; ++j) {
                eq_ct[j] = 700.0;
                eq_t[j] = 700.0;
            }
        } else {
            buy_vec(act_n, res_ct, bt_ct, eq_ct);
            buy_vec(act_n, res_t, bt_t, eq_t);
        }

        const double *u_rnd = NULL;
        const double *lu_rnd = NULL;
        if (logit_u_pre) {
            lu_rnd = logit_u_pre + (int64_t)(rnd - 1) * n;
            if (u_pre) u_rnd = u_pre + (int64_t)(rnd - 1) * n;
        } else if (u_pre) {
            u_rnd = u_pre + (int64_t)(rnd - 1) * n;
        } else {
            random_standard_uniform_fill(bg, (intptr_t)n, u_buf);
            u_rnd = u_buf;
        }

        double base_const = (pistol ? c1 : 0.0) + (ot ? c2 : 0.0) + map_term + rm_inter;
        double c3_signed = a_ct ? c3 : -c3;

        int64_t a_won_alive_sum = 0;
        __m256d v_200 = _mm256_set1_pd(200.0);
        __m256d v_c0 = _mm256_set1_pd(c0);
        __m256d v_c3s = _mm256_set1_pd(c3_signed);
        __m256d v_bc = _mm256_set1_pd(base_const);
        __m256d v_abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));

        if (lu_rnd) {
            __m256d v_eps = _mm256_set1_pd(1e-6);
            int64_t j = 0;
            for (; j + 4 <= act_n; j += 4) {
                __m256d v_str = _mm256_loadu_pd(str_a + j);
                __m256d v_z;
                if (pistol) {
                    v_z = _mm256_fmadd_pd(v_str, v_c3s, v_bc);
                } else {
                    __m256d v_ec = _mm256_add_pd(_mm256_loadu_pd(eq_ct + j), v_200);
                    __m256d v_et = _mm256_add_pd(_mm256_loadu_pd(eq_t + j), v_200);
                    __m256d v_x0 = fast_log_pd4(_mm256_div_pd(v_ec, v_et));
                    v_z = _mm256_fmadd_pd(v_x0, v_c0, _mm256_fmadd_pd(v_str, v_c3s, v_bc));
                }
                __m256d v_lu;
                if (act_n == n) {
                    v_lu = _mm256_loadu_pd(lu_rnd + j);
                } else {
                    v_lu = _mm256_setr_pd(lu_rnd[act_idx[j]], lu_rnd[act_idx[j+1]], lu_rnd[act_idx[j+2]], lu_rnd[act_idx[j+3]]);
                }
                __m256d v_diff = _mm256_and_pd(_mm256_sub_pd(v_z, v_lu), v_abs_mask);
                int guard = _mm256_movemask_pd(_mm256_cmp_pd(v_diff, v_eps, _CMP_LT_OQ));
                int cmp_m = _mm256_movemask_pd(_mm256_cmp_pd(v_lu, v_z, _CMP_LT_OQ));
                if (__builtin_expect(guard == 0, 1)) {
                    for (int lane = 0; lane < 4; ++lane) {
                        int64_t jj = j + lane;
                        int64_t oi = (act_n == n) ? jj : act_idx[jj];
                        uint8_t ctw = (uint8_t)((cmp_m >> lane) & 1);
                        uint8_t aw = a_ct ? ctw : (uint8_t)(1 - ctw);
                        a_won[jj] = aw;
                        b_won[jj] = (uint8_t)(1 - aw);
                        a_won_alive_sum += aw;
                        sa[oi] += aw;
                        sb[oi] += (1 - aw);
                    }
                } else {
                    for (int lane = 0; lane < 4; ++lane) {
                        int64_t jj = j + lane;
                        int64_t oi = (act_n == n) ? jj : act_idx[jj];
                        uint8_t ctw;
                        if ((guard >> lane) & 1) {
                            double p_ex = exact_p_ct(eq_ct[jj], eq_t[jj], pistol, ot, a_ct ? str_a[jj] : -str_a[jj], map_idx, rm_coef, rm_inter);
                            double u_ex = u_rnd ? u_rnd[oi] : u_cb(seed_id, rnd, oi);
                            ctw = (u_ex < p_ex) ? 1 : 0;
                        } else {
                            ctw = (uint8_t)((cmp_m >> lane) & 1);
                        }
                        uint8_t aw = a_ct ? ctw : (uint8_t)(1 - ctw);
                        a_won[jj] = aw;
                        b_won[jj] = (uint8_t)(1 - aw);
                        a_won_alive_sum += aw;
                        sa[oi] += aw;
                        sb[oi] += (1 - aw);
                    }
                }
            }
            for (; j < act_n; ++j) {
                int64_t oi = (act_n == n) ? j : act_idx[j];
                double x0 = pistol ? 0.0 : fast_log((eq_ct[j] + 200.0) / (eq_t[j] + 200.0));
                double z = x0 * c0 + str_a[j] * c3_signed + base_const;
                double lu = lu_rnd[oi];
                uint8_t ctw;
                if (__builtin_expect(fabs(z - lu) < 1e-6, 0)) {
                    double p_ex = exact_p_ct(eq_ct[j], eq_t[j], pistol, ot, a_ct ? str_a[j] : -str_a[j], map_idx, rm_coef, rm_inter);
                    double u_ex = u_rnd ? u_rnd[oi] : u_cb(seed_id, rnd, oi);
                    ctw = (u_ex < p_ex) ? 1 : 0;
                } else {
                    ctw = (lu < z) ? 1 : 0;
                }
                uint8_t aw = a_ct ? ctw : (uint8_t)(1 - ctw);
                a_won[j] = aw;
                b_won[j] = (uint8_t)(1 - aw);
                a_won_alive_sum += aw;
                sa[oi] += aw;
                sb[oi] += (1 - aw);
            }
        } else {
            __m256d v_eps = _mm256_set1_pd(1e-7);
            __m256d v_one = _mm256_set1_pd(1.0);
            __m256d v_neg = _mm256_set1_pd(-0.0);
            int64_t j = 0;
            for (; j + 4 <= act_n; j += 4) {
                __m256d v_str = _mm256_loadu_pd(str_a + j);
                __m256d v_z;
                if (pistol) {
                    v_z = _mm256_fmadd_pd(v_str, v_c3s, v_bc);
                } else {
                    __m256d v_ec = _mm256_add_pd(_mm256_loadu_pd(eq_ct + j), v_200);
                    __m256d v_et = _mm256_add_pd(_mm256_loadu_pd(eq_t + j), v_200);
                    __m256d v_x0 = fast_log_pd4(_mm256_div_pd(v_ec, v_et));
                    v_z = _mm256_fmadd_pd(v_x0, v_c0, _mm256_fmadd_pd(v_str, v_c3s, v_bc));
                }
                __m256d v_p = _mm256_div_pd(v_one, _mm256_add_pd(v_one, fast_exp_pd4(_mm256_xor_pd(v_z, v_neg))));
                __m256d v_u;
                if (act_n == n) {
                    v_u = _mm256_loadu_pd(u_rnd + j);
                } else {
                    v_u = _mm256_setr_pd(u_rnd[act_idx[j]], u_rnd[act_idx[j+1]], u_rnd[act_idx[j+2]], u_rnd[act_idx[j+3]]);
                }
                __m256d v_diff = _mm256_and_pd(_mm256_sub_pd(v_u, v_p), v_abs_mask);
                int guard = _mm256_movemask_pd(_mm256_cmp_pd(v_diff, v_eps, _CMP_LT_OQ));
                int cmp_m = _mm256_movemask_pd(_mm256_cmp_pd(v_u, v_p, _CMP_LT_OQ));
                if (__builtin_expect(guard == 0, 1)) {
                    for (int lane = 0; lane < 4; ++lane) {
                        int64_t jj = j + lane;
                        int64_t oi = (act_n == n) ? jj : act_idx[jj];
                        uint8_t ctw = (uint8_t)((cmp_m >> lane) & 1);
                        uint8_t aw = a_ct ? ctw : (uint8_t)(1 - ctw);
                        a_won[jj] = aw;
                        b_won[jj] = (uint8_t)(1 - aw);
                        a_won_alive_sum += aw;
                        sa[oi] += aw;
                        sb[oi] += (1 - aw);
                    }
                } else {
                    for (int lane = 0; lane < 4; ++lane) {
                        int64_t jj = j + lane;
                        int64_t oi = (act_n == n) ? jj : act_idx[jj];
                        uint8_t ctw;
                        if ((guard >> lane) & 1) {
                            double p_ex = exact_p_ct(eq_ct[jj], eq_t[jj], pistol, ot, a_ct ? str_a[jj] : -str_a[jj], map_idx, rm_coef, rm_inter);
                            ctw = (u_rnd[oi] < p_ex) ? 1 : 0;
                        } else {
                            ctw = (uint8_t)((cmp_m >> lane) & 1);
                        }
                        uint8_t aw = a_ct ? ctw : (uint8_t)(1 - ctw);
                        a_won[jj] = aw;
                        b_won[jj] = (uint8_t)(1 - aw);
                        a_won_alive_sum += aw;
                        sa[oi] += aw;
                        sb[oi] += (1 - aw);
                    }
                }
            }
            for (; j < act_n; ++j) {
                int64_t oi = (act_n == n) ? j : act_idx[j];
                double x0 = pistol ? 0.0 : fast_log((eq_ct[j] + 200.0) / (eq_t[j] + 200.0));
                double z = x0 * c0 + str_a[j] * c3_signed + base_const;
                double p_approx = 1.0 / (1.0 + fast_exp(-z));
                double u = u_rnd[oi];
                uint8_t ctw;
                if (__builtin_expect(fabs(u - p_approx) < 1e-7, 0)) {
                    double p_ex = exact_p_ct(eq_ct[j], eq_t[j], pistol, ot, a_ct ? str_a[j] : -str_a[j], map_idx, rm_coef, rm_inter);
                    ctw = (u < p_ex) ? 1 : 0;
                } else {
                    ctw = (u < p_approx) ? 1 : 0;
                }
                uint8_t aw = a_ct ? ctw : (uint8_t)(1 - ctw);
                a_won[j] = aw;
                b_won[j] = (uint8_t)(1 - aw);
                a_won_alive_sum += aw;
                sa[oi] += aw;
                sb[oi] += (1 - aw);
            }
        }

        if (mode == 3) {
            has_round[rnd] = 1;
            win_round_p[rnd] = (double)a_won_alive_sum / (double)act_n;
        }

        if (a_ct) {
            next_res_vec(act_n, res_a, eq_a, a_won, loss_a, tc_ct, tr_ct_inter, Z3, tmp, blas_buf);
            next_res_vec(act_n, res_b, eq_b, b_won, loss_b, tc_t, tr_t_inter, Z3, tmp, blas_buf);
        } else {
            next_res_vec(act_n, res_a, eq_a, a_won, loss_a, tc_t, tr_t_inter, Z3, tmp, blas_buf);
            next_res_vec(act_n, res_b, eq_b, b_won, loss_b, tc_ct, tr_ct_inter, Z3, tmp, blas_buf);
        }
        for (int64_t j = 0; j < act_n; ++j) {
            if (a_won[j]) {
                double v = loss_a[j] - 1.0;
                loss_a[j] = v > 0.0 ? v : 0.0;
                double vb = loss_b[j] + 1.0;
                loss_b[j] = vb < 4.0 ? vb : 4.0;
            } else {
                double v = loss_a[j] + 1.0;
                loss_a[j] = v < 4.0 ? v : 4.0;
                double vb = loss_b[j] - 1.0;
                loss_b[j] = vb > 0.0 ? vb : 0.0;
            }
        }

        if (mode == 3 && rnd == 12) {
            int64_t s_a_tot = 0, s_b_tot = 0;
            for (int64_t i = 0; i < n; ++i) {
                s_a_tot += sa[i];
                s_b_tot += sb[i];
            }
            out_h1[0] = (double)s_a_tot / (double)n;
            out_h1[1] = (double)s_b_tot / (double)n;
        }

        if (rnd >= 13) {
            int64_t out_j = 0;
            if (rnd <= 24) {
                for (int64_t j = 0; j < act_n; ++j) {
                    int64_t orig_i = act_idx[j];
                    if (sa[orig_i] < 13 && sb[orig_i] < 13) {
                        if (out_j != j) {
                            act_idx[out_j] = orig_i;
                            str_a[out_j] = str_a[j];
                            res_a[out_j] = res_a[j];
                            res_b[out_j] = res_b[j];
                            loss_a[out_j] = loss_a[j];
                            loss_b[out_j] = loss_b[j];
                        }
                        out_j++;
                    }
                }
            } else {
                int64_t base = 12 + 3 * ((rnd - 25) / 6);
                for (int64_t j = 0; j < act_n; ++j) {
                    int64_t orig_i = act_idx[j];
                    if ((sa[orig_i] - base < 4) && (sb[orig_i] - base < 4)) {
                        if (out_j != j) {
                            act_idx[out_j] = orig_i;
                            str_a[out_j] = str_a[j];
                            res_a[out_j] = res_a[j];
                            res_b[out_j] = res_b[j];
                            loss_a[out_j] = loss_a[j];
                            loss_b[out_j] = loss_b[j];
                        }
                        out_j++;
                    }
                }
            }
            act_n = out_j;
        }
    }

    int64_t wins = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (sa[i] > sb[i]) wins++;
    }
    *out_p_win = (double)wins / (double)n;
    if (mode == 1) {
        int64_t lr_sum = 0;
        for (int64_t i = 0; i < n; ++i) {
            lr_sum += (sa[i] < sb[i]) ? sa[i] : sb[i];
        }
        *out_lr_mean = (double)lr_sum / (double)n;
    } else if (mode >= 2) {
        for (int64_t i = 0; i < n; ++i) {
            ext_lr[i] = (sa[i] < sb[i]) ? sa[i] : sb[i];
        }
        if (mode == 3) {
            memcpy(ext_sa, sa, (size_t)n * sizeof(int64_t));
            memcpy(ext_sb, sb, (size_t)n * sizeof(int64_t));
        }
    }
    return 0;
}

int c_sim_core(
    const char *openblas_path,
    void *bitgen_ptr, int map_idx, double str_a_scalar, int a_ct_first, int64_t n, double sigma,
    const double *rm_coef, double rm_inter,
    int64_t m_ct, const double *bx_ct, const double *by_ct,
    int64_t m_t, const double *bx_t, const double *by_t,
    const double *tc_ct, double tr_ct_inter,
    const double *tc_t, double tr_t_inter,
    int mode,
    double *out_p_win, double *out_lr_mean,
    int64_t *ext_lr, int64_t *ext_sa, int64_t *ext_sb,
    double *out_h1, int *has_round, double *win_round_p
) {
    if (ensure_blas(openblas_path) != 0) return -2;
    BuyTable bt_ct, bt_t;
    init_buy_table(&bt_ct, m_ct, bx_ct, by_ct);
    init_buy_table(&bt_t, m_t, bx_t, by_t);
    return run_sim_engine(
        (bitgen_t *)bitgen_ptr, NULL, NULL, NULL, NULL, 0,
        map_idx, str_a_scalar, a_ct_first, n, sigma,
        rm_coef, rm_inter, &bt_ct, &bt_t, tc_ct, tr_ct_inter, tc_t, tr_t_inter,
        mode, out_p_win, out_lr_mean, ext_lr, ext_sa, ext_sb, out_h1, has_round, win_round_p
    );
}

int c_anchor_only(
    const char *openblas_path, double target,
    const double *rm_coef_in, double rm_inter, double sigma,
    int64_t m_ct, const double *bx_ct, const double *by_ct,
    int64_t m_t, const double *bx_t, const double *by_t,
    const double *tc_ct, double tr_ct_inter,
    const double *tc_t, double tr_t_inter,
    const double *z7, const double *u7, const double *lu7,
    double *out_c3
) {
    if (ensure_blas(openblas_path) != 0) return -2;
    double coef[13];
    memcpy(coef, rm_coef_in, 13 * sizeof(double));
    BuyTable bt_ct, bt_t;
    init_buy_table(&bt_ct, m_ct, bx_ct, by_ct);
    init_buy_table(&bt_t, m_t, bx_t, by_t);
    const int anchor_maps[3] = {5, 0, 6};
    double want = 1.0 / (1.0 + exp(-target));
    double lo = 0.0, hi = 3.0;
    double p_win = 0.0, lr_m = 0.0;
    for (int step = 0; step < 9; ++step) {
        double mid = (lo + hi) * 0.5;
        coef[3] = mid;
        double sum_p = 0.0;
        for (int mi = 0; mi < 3; ++mi) {
            for (int ct_i = 0; ct_i < 2; ++ct_i) {
                int a_ct = (ct_i == 0) ? 1 : 0;
                if (run_sim_engine(NULL, z7, u7, lu7, NULL, 7, anchor_maps[mi], target, a_ct, 2000, sigma,
                                   coef, rm_inter, &bt_ct, &bt_t, tc_ct, tr_ct_inter, tc_t, tr_t_inter,
                                   0, &p_win, &lr_m, NULL, NULL, NULL, NULL, NULL, NULL) != 0) return -1;
                sum_p += p_win;
            }
        }
        double got = sum_p / 6.0;
        if (got < want) lo = mid;
        else hi = mid;
    }
    *out_c3 = (lo + hi) * 0.5;
    return 0;
}

int c_calibrate_sigma(
    const char *openblas_path,
    const double *rm_coef_in, double rm_inter,
    int64_t m_ct, const double *bx_ct, const double *by_ct,
    int64_t m_t, const double *bx_t, const double *by_t,
    const double *tc_ct, double tr_ct_inter,
    const double *tc_t, double tr_t_inter,
    int64_t n_str, const double *strengths, double real_lr,
    const double *z7, const double *u7, const double *lu7,
    const double *z400, const double *lu400, exact_u_fn u_cb,
    double *out_best_sigma, double *out_best_c3
) {
    if (ensure_blas(openblas_path) != 0) return -2;
    double coef[13];
    memcpy(coef, rm_coef_in, 13 * sizeof(double));
    BuyTable bt_ct, bt_t;
    init_buy_table(&bt_ct, m_ct, bx_ct, by_ct);
    init_buy_table(&bt_t, m_t, bx_t, by_t);

    if (n_str > 60) n_str = 60;
    const double sigmas[5] = {0.0, 0.75, 1.5, 2.25, 3.0};
    const int anchor_maps[3] = {5, 0, 6};
    double want = 1.0 / (1.0 + exp(-1.0));

    int has_best = 0;
    double best_diff = 0.0, best_sigma = 0.0, best_c3 = 0.0;
    double p_win = 0.0, lr_m = 0.0;

    for (int si = 0; si < 5; ++si) {
        double sigma = sigmas[si];
        double lo = 0.0, hi = 3.0;
        for (int step = 0; step < 9; ++step) {
            double mid = (lo + hi) * 0.5;
            coef[3] = mid;
            double sum_p = 0.0;
            for (int mi = 0; mi < 3; ++mi) {
                for (int ct_i = 0; ct_i < 2; ++ct_i) {
                    int a_ct = (ct_i == 0) ? 1 : 0;
                    if (run_sim_engine(NULL, z7, u7, lu7, NULL, 7, anchor_maps[mi], 1.0, a_ct, 2000, sigma,
                                       coef, rm_inter, &bt_ct, &bt_t, tc_ct, tr_ct_inter, tc_t, tr_t_inter,
                                       0, &p_win, &lr_m, NULL, NULL, NULL, NULL, NULL, NULL) != 0) return -1;
                    sum_p += p_win;
                }
            }
            double got = sum_p / 6.0;
            if (got < want) lo = mid;
            else hi = mid;
        }
        coef[3] = (lo + hi) * 0.5;

        double sum_lr = 0.0;
        for (int64_t i = 0; i < n_str; ++i) {
            if (run_sim_engine(NULL, z400 + i * 400, NULL, lu400 + i * 60 * 400, u_cb, i,
                               5, strengths[i], 1, 400, sigma,
                               coef, rm_inter, &bt_ct, &bt_t, tc_ct, tr_ct_inter, tc_t, tr_t_inter,
                               1, &p_win, &lr_m, NULL, NULL, NULL, NULL, NULL, NULL) != 0) return -1;
            sum_lr += lr_m;
        }
        double sim_lr = (n_str > 0) ? (sum_lr / (double)n_str) : NAN;
        double diff = fabs(sim_lr - real_lr);
        if (!has_best || diff < best_diff) {
            has_best = 1;
            best_diff = diff;
            best_sigma = sigma;
            best_c3 = coef[3];
        }
    }
    *out_best_sigma = best_sigma;
    *out_best_c3 = best_c3;
    return 0;
}

/* ==================== Part 2: Pure-C Prematch Strength Engine ==================== */
#pragma GCC optimize("O2")

static uint64_t fnv1a_str(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= (uint64_t)(*p);
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}

static uint64_t mix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k ? k : 1ULL;
}

typedef struct {
    char *buf;
    uint32_t len;
    uint32_t cap;
} StrPool;

static void sp_init(StrPool *sp, uint32_t cap) {
    sp->buf = (char *)malloc(cap);
    sp->buf[0] = '\0';
    sp->len = 1;
    sp->cap = cap;
}

static uint32_t sp_add(StrPool *sp, const char *s, size_t slen) {
    while (sp->len + slen + 1 > sp->cap) {
        sp->cap *= 2;
        sp->buf = (char *)realloc(sp->buf, sp->cap);
    }
    uint32_t off = sp->len;
    memcpy(sp->buf + off, s, slen);
    sp->buf[off + slen] = '\0';
    sp->len += (uint32_t)(slen + 1);
    return off;
}

/* String -> int32 intern table */
typedef struct {
    uint64_t *hashes;
    uint32_t *str_offs;
    int32_t *ids;
    uint32_t cap; /* power of 2 */
    int32_t count;
} InternTab;

static void it_init(InternTab *it, uint32_t cap) {
    it->hashes = (uint64_t *)calloc(cap, sizeof(uint64_t));
    it->str_offs = (uint32_t *)calloc(cap, sizeof(uint32_t));
    it->ids = (int32_t *)malloc(cap * sizeof(int32_t));
    it->cap = cap;
    it->count = 0;
}

static void it_free(InternTab *it) {
    free(it->hashes);
    free(it->str_offs);
    free(it->ids);
}

static void it_grow(InternTab *it) {
    uint32_t ncap = it->cap * 2;
    uint64_t *nh = (uint64_t *)calloc(ncap, sizeof(uint64_t));
    uint32_t *no = (uint32_t *)calloc(ncap, sizeof(uint32_t));
    int32_t *ni = (int32_t *)malloc(ncap * sizeof(int32_t));
    uint32_t mask = ncap - 1;
    for (uint32_t i = 0; i < it->cap; ++i) {
        if (it->hashes[i]) {
            uint32_t slot = (uint32_t)it->hashes[i] & mask;
            while (nh[slot]) slot = (slot + 1) & mask;
            nh[slot] = it->hashes[i];
            no[slot] = it->str_offs[i];
            ni[slot] = it->ids[i];
        }
    }
    free(it->hashes); free(it->str_offs); free(it->ids);
    it->hashes = nh; it->str_offs = no; it->ids = ni; it->cap = ncap;
}

static int32_t it_intern(InternTab *it, StrPool *sp, const char *s, uint32_t *out_off_by_id_arr_unused) {
    (void)out_off_by_id_arr_unused;
    if ((uint32_t)(it->count + 1) * 10 >= it->cap * 7) it_grow(it);
    uint64_t h = fnv1a_str(s);
    uint32_t mask = it->cap - 1;
    uint32_t slot = (uint32_t)h & mask;
    while (it->hashes[slot]) {
        if (it->hashes[slot] == h && strcmp(sp->buf + it->str_offs[slot], s) == 0) {
            return it->ids[slot];
        }
        slot = (slot + 1) & mask;
    }
    uint32_t off = sp_add(sp, s, strlen(s));
    int32_t id = it->count++;
    it->hashes[slot] = h;
    it->str_offs[slot] = off;
    it->ids[slot] = id;
    return id;
}

/* uint64 -> int64 hash map */
typedef struct {
    uint64_t *keys;
    uint8_t *used;
    int64_t *vals;
    uint32_t cap;
    uint32_t count;
} U64Map;

static void u64m_init(U64Map *m, uint32_t cap) {
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->used = (uint8_t *)calloc(cap, 1);
    m->vals = (int64_t *)malloc(cap * sizeof(int64_t));
    m->cap = cap;
    m->count = 0;
}

static void u64m_free(U64Map *m) {
    free(m->keys); free(m->used); free(m->vals);
}

static void u64m_grow(U64Map *m) {
    uint32_t ncap = m->cap * 2;
    U64Map nm;
    u64m_init(&nm, ncap);
    uint32_t mask = ncap - 1;
    for (uint32_t i = 0; i < m->cap; ++i) {
        if (m->used[i]) {
            uint32_t slot = (uint32_t)mix64(m->keys[i]) & mask;
            while (nm.used[slot]) slot = (slot + 1) & mask;
            nm.used[slot] = 1;
            nm.keys[slot] = m->keys[i];
            nm.vals[slot] = m->vals[i];
            nm.count++;
        }
    }
    u64m_free(m);
    *m = nm;
}

static int u64m_get(const U64Map *m, uint64_t k, int64_t *out_v) {
    uint32_t mask = m->cap - 1;
    uint32_t slot = (uint32_t)mix64(k) & mask;
    while (m->used[slot]) {
        if (m->keys[slot] == k) {
            *out_v = m->vals[slot];
            return 1;
        }
        slot = (slot + 1) & mask;
    }
    return 0;
}

/* setdefault: inserts only if missing, returns existing or inserted value */
static int64_t u64m_setdefault(U64Map *m, uint64_t k, int64_t v) {
    if ((m->count + 1) * 10 >= m->cap * 7) u64m_grow(m);
    uint32_t mask = m->cap - 1;
    uint32_t slot = (uint32_t)mix64(k) & mask;
    while (m->used[slot]) {
        if (m->keys[slot] == k) return m->vals[slot];
        slot = (slot + 1) & mask;
    }
    m->used[slot] = 1;
    m->keys[slot] = k;
    m->vals[slot] = v;
    m->count++;
    return v;
}

static void compute_key_str(const char *in, char *out, size_t cap) {
    char s[512];
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p && n + 1 < sizeof(s); ++p) {
        unsigned char c = (unsigned char)tolower(*p);
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) s[n++] = (char)c;
    }
    s[n] = '\0';
    size_t start = 0, end = n;
    if (n >= 4 && memcmp(s, "team", 4) == 0) start = 4;
    static const char * const suffixes[] = {"esports", "esport", "gaming", "club", "team", "csgo", "cs2", "cs"};
    static const size_t suflen[] = {7, 6, 6, 4, 4, 4, 3, 2};
    size_t rem = end - start;
    for (int k = 0; k < 8; ++k) {
        size_t L = suflen[k];
        if (rem >= L && memcmp(s + end - L, suffixes[k], L) == 0) {
            end -= L;
            break;
        }
    }
    if (end > start) {
        size_t len = end - start < cap - 1 ? end - start : cap - 1;
        memcpy(out, s + start, len);
        out[len] = '\0';
    } else {
        size_t len = n < cap - 1 ? n : cap - 1;
        memcpy(out, s, len);
        out[len] = '\0';
    }
}

static void compute_norm_str(const char *in, char *out, size_t cap) {
    char low[512];
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p && n + 1 < sizeof(low); ++p) {
        low[n++] = (char)tolower(*p);
    }
    low[n] = '\0';
    char stripped[512];
    size_t m = 0, i = 0;
    while (i < n) {
        if (low[i] == '(') {
            size_t j = i + 1;
            int found = 0;
            while (j < n && low[j] != '\n') {
                if (low[j] == ')') { found = 1; break; }
                j++;
            }
            if (found) { i = j + 1; continue; }
        }
        if (i + 7 <= n && memcmp(low + i, "globant", 7) == 0) { i += 7; continue; }
        if (i + 4 <= n && memcmp(low + i, "clan", 4) == 0) { i += 4; continue; }
        if (m + 1 < sizeof(stripped)) stripped[m++] = low[i];
        i++;
    }
    stripped[m] = '\0';
    compute_key_str(stripped, out, cap);
}

/* Civil date (y, m, d) -> days since 1970-01-01 (Howard Hinnant's exact civil_from_days inverse) */
static inline int32_t days_from_civil(int y, int m, int d) {
    y -= (m <= 2);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (unsigned)(m + (m > 2 ? -3 : 9)) + 2) / 5 + (unsigned)d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return (int32_t)(era * 146097 + (int)doe - 719468);
}

static inline int parse_2dig(const char *p) {
    return (p[0] - '0') * 10 + (p[1] - '0');
}

static void parse_datetime_str(const char *s, int64_t *out_sec, int32_t *out_day, char *norm_iso19, size_t iso_cap) {
    size_t len = strlen(s);
    int y = (s[0]-'0')*1000 + (s[1]-'0')*100 + (s[2]-'0')*10 + (s[3]-'0');
    int m = parse_2dig(s + 5);
    int d = parse_2dig(s + 8);
    int hr = (len >= 13) ? parse_2dig(s + 11) : 0;
    int mi = (len >= 16) ? parse_2dig(s + 14) : 0;
    int sc = (len >= 19) ? parse_2dig(s + 17) : 0;
    int32_t days = days_from_civil(y, m, d);
    *out_day = days;
    *out_sec = (int64_t)days * 86400LL + (int64_t)hr * 3600LL + (int64_t)mi * 60LL + (int64_t)sc;
    if (norm_iso19) {
        if (len == 19) {
            memcpy(norm_iso19, s, 20);
        } else {
            snprintf(norm_iso19, iso_cap, "%04d-%02d-%02d %02d:%02d:%02d", y & 0x3fff, m & 0x1f, d & 0x1f, hr & 0x3f, mi & 0x3f, sc & 0x3f);
        }
    }
}

/* Read entire file into memory buffer (freed right after parsing that single file!) */
static char *slurp_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return NULL; }
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 2);
    if (!buf) { fclose(fp); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[rd] = '\n';
    buf[rd + 1] = '\0';
    *out_len = rd;
    return buf;
}

/* Parse one RFC-4180 CSV record in-place from *cur (up to end), writing field pointers into cols[0..max_cols-1].
   Returns number of fields parsed, or 0 at EOF. */
static int parse_csv_row(char **cur_p, char *end, char **cols, int max_cols) {
    char *p = *cur_p;
    while (p < end && (*p == '\r' || *p == '\n')) p++;
    if (p >= end) { *cur_p = p; return 0; }
    int ncol = 0;
    while (p < end) {
        char *val;
        if (*p == '"') {
            p++;
            val = p;
            char *w = p;
            while (p < end) {
                if (*p == '"') {
                    if (p + 1 < end && p[1] == '"') {
                        *w++ = '"';
                        p += 2;
                    } else {
                        p++;
                        break;
                    }
                } else {
                    *w++ = *p++;
                }
            }
            while (p < end && *p != ',' && *p != '\n' && *p != '\r') p++;
            char term = (p < end) ? *p : '\n';
            if (p < end) p++;
            if (term == '\r' && p < end && *p == '\n') p++;
            *w = '\0';
            if (ncol < max_cols) cols[ncol++] = val;
            if (term != ',') break;
        } else {
            val = p;
            while (p < end && *p != ',' && *p != '\n' && *p != '\r') p++;
            char term = (p < end) ? *p : '\n';
            *p = '\0';
            if (p < end) p++;
            if (term == '\r' && p < end && *p == '\n') p++;
            if (ncol < max_cols) cols[ncol++] = val;
            if (term != ',') break;
        }
    }
    *cur_p = p;
    return ncol;
}

static inline int is_na_str(const char *s) {
    if (!s || !s[0]) return 1;
    if ((s[0] == 'n' || s[0] == 'N') && (s[1] == 'a' || s[1] == 'A') && (s[2] == 'n' || s[2] == 'N') && s[3] == '\0') return 1;
    return 0;
}

typedef struct {
    int32_t raw_off;
    int32_t key_id;
    int32_t norm_id;
} TeamMeta;

typedef struct {
    InternTab raw_tab;
    InternTab key_tab;
    InternTab norm_tab;
    TeamMeta *meta;
    int32_t cap;
} TeamRegistry;

static void tr_init(TeamRegistry *tr) {
    it_init(&tr->raw_tab, 8192);
    it_init(&tr->key_tab, 8192);
    it_init(&tr->norm_tab, 8192);
    tr->cap = 8192;
    tr->meta = (TeamMeta *)malloc((size_t)tr->cap * sizeof(TeamMeta));
}

static void tr_free(TeamRegistry *tr) {
    it_free(&tr->raw_tab);
    it_free(&tr->key_tab);
    it_free(&tr->norm_tab);
    free(tr->meta);
}

static int32_t tr_get_team(TeamRegistry *tr, StrPool *sp, const char *name) {
    int32_t prev_cnt = tr->raw_tab.count;
    int32_t tid = it_intern(&tr->raw_tab, sp, name, NULL);
    if (tid == prev_cnt) {
        if (tid >= tr->cap) {
            tr->cap *= 2;
            tr->meta = (TeamMeta *)realloc(tr->meta, (size_t)tr->cap * sizeof(TeamMeta));
        }
        uint32_t r_off = sp_add(sp, name, strlen(name));
        char kbuf[256], nbuf[256];
        compute_key_str(name, kbuf, sizeof(kbuf));
        compute_norm_str(name, nbuf, sizeof(nbuf));
        int32_t kid = it_intern(&tr->key_tab, sp, kbuf, NULL);
        int32_t nid = it_intern(&tr->norm_tab, sp, nbuf, NULL);
        tr->meta[tid].raw_off = (int32_t)r_off;
        tr->meta[tid].key_id = kid;
        tr->meta[tid].norm_id = nid;
    }
    return tid;
}

typedef struct {
    int32_t la[5];
    uint8_t n_la;
    uint8_t valid_a[5];
    double stats_a[5][4];
    int32_t lb[5];
    uint8_t n_lb;
    uint8_t valid_b[5];
    double stats_b[5][4];
} KgLineupStats;

typedef struct {
    int64_t dt_sec;
    int32_t day_ord;
    uint8_t is_lp;
    int64_t mid_int;
    uint32_t mid_str_off;
    double gid;
    double mno;
    uint32_t orig_idx;
    int32_t t1_id;
    int32_t t2_id;
    int32_t map_id;
    double s1;
    double s2;
    double bo;
    double tier;
    int32_t kg_idx; /* -1 for LP */
} HistRow;

static const char *g_sort_sp_buf = NULL;

static int cmp_kg_rows(const void *pa, const void *pb) {
    const HistRow *a = (const HistRow *)pa;
    const HistRow *b = (const HistRow *)pb;
    if (a->dt_sec != b->dt_sec) return (a->dt_sec < b->dt_sec) ? -1 : 1;
    if (a->mid_int != b->mid_int) return (a->mid_int < b->mid_int) ? -1 : 1;
    if (a->gid != b->gid) return (a->gid < b->gid) ? -1 : 1;
    return (a->orig_idx < b->orig_idx) ? -1 : (a->orig_idx > b->orig_idx ? 1 : 0);
}

static int cmp_hist_rows(const void *pa, const void *pb) {
    const HistRow *a = (const HistRow *)pa;
    const HistRow *b = (const HistRow *)pb;
    if (a->dt_sec != b->dt_sec) return (a->dt_sec < b->dt_sec) ? -1 : 1;
    if (a->is_lp != b->is_lp) return (a->is_lp < b->is_lp) ? -1 : 1;
    if (a->is_lp == 0) {
        if (a->mid_int != b->mid_int) return (a->mid_int < b->mid_int) ? -1 : 1;
    } else {
        int sc = strcmp(g_sort_sp_buf + a->mid_str_off, g_sort_sp_buf + b->mid_str_off);
        if (sc != 0) return sc;
    }
    if (a->mno != b->mno) return (a->mno < b->mno) ? -1 : 1;
    return (a->orig_idx < b->orig_idx) ? -1 : (a->orig_idx > b->orig_idx ? 1 : 0);
}

/* Exact NumPy 1D pairwise sum */
static double np_sum(const double *a, int n) {
    if (n <= 8) {
        if (n <= 0) return 0.0;
        double res = a[0];
        for (int i = 1; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        double r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        int i = 8;
        for (; i <= n - 8; i += 8) {
            r0 += a[i+0]; r1 += a[i+1]; r2 += a[i+2]; r3 += a[i+3];
            r4 += a[i+4]; r5 += a[i+5]; r6 += a[i+6]; r7 += a[i+7];
        }
        double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int half = n / 2;
    half -= half % 8;
    return np_sum(a, half) + np_sum(a + half, n - half);
}

typedef struct {
    double elo;
    double form_w[10];
    double form_r[10];
    uint8_t form_len;
    uint8_t form_pos;
    int32_t lineup[5];
    uint8_t lineup_len;
    int64_t last_sec;
    uint8_t has_last;
    int32_t n_maps;
} TeamState;

typedef struct {
    double stats[20][4];
    uint8_t len;
    uint8_t pos;
} PlayerState;

typedef struct {
    int32_t *winners;
    int32_t len;
    int32_t cap;
} H2HList;

static inline uint64_t pack_pair_map_day(int32_t a, int32_t b, int32_t m, int32_t day) {
    uint64_t lo = (uint64_t)(uint32_t)(a < b ? a : b);
    uint64_t hi = (uint64_t)(uint32_t)(a < b ? b : a);
    uint64_t h = mix64((lo << 32) | hi);
    return mix64(h ^ (((uint64_t)(uint32_t)m << 32) | (uint64_t)(uint32_t)day));
}

static inline uint64_t pack_u32_pair(int32_t a, int32_t b) {
    uint64_t lo = (uint64_t)(uint32_t)(a < b ? a : b);
    uint64_t hi = (uint64_t)(uint32_t)(a < b ? b : a);
    return (lo << 32) | hi;
}

static inline uint64_t pack_norm_pair_map(int32_t na, int32_t nb, int32_t m) {
    uint64_t lo = (uint64_t)(uint32_t)(na < nb ? na : nb);
    uint64_t hi = (uint64_t)(uint32_t)(na < nb ? nb : na);
    return (lo << 42) | (hi << 21) | (uint64_t)(uint32_t)m;
}

static void eval_side_feat(
    int32_t tid, int32_t mid, const int32_t *lu, int n_lu, int64_t when_sec,
    const TeamState *ts_arr, int32_t n_maps_vocab, const double *map_elo_arr, const int32_t *n_map_arr,
    const PlayerState *ps_arr, double out[11]
) {
    const TeamState *ts = &ts_arr[tid];
    out[0] = ts->elo;
    out[1] = map_elo_arr[(size_t)tid * (size_t)n_maps_vocab + (size_t)mid];

    if (ts->form_len > 0) {
        double fw[10], fr[10];
        int fl = ts->form_len;
        int start = (fl == 10) ? ts->form_pos : 0;
        for (int i = 0; i < fl; ++i) {
            int idx = (start + i) % 10;
            fw[i] = ts->form_w[idx];
            fr[i] = ts->form_r[idx];
        }
        out[2] = np_sum(fw, fl) / (double)fl;
        out[3] = np_sum(fr, fl) / (double)fl;
    } else {
        out[2] = NAN;
        out[3] = NAN;
    }

    double c0[100], c1[100], c2[100], c3[100];
    int np_rows = 0;
    for (int i = 0; i < n_lu; ++i) {
        int32_t pid = lu[i];
        if (pid >= 0 && ps_arr[pid].len > 0) {
            const PlayerState *ps = &ps_arr[pid];
            int pl = ps->len;
            int start = (pl == 20) ? ps->pos : 0;
            for (int k = 0; k < pl; ++k) {
                int idx = (start + k) % 20;
                c0[np_rows] = ps->stats[idx][0];
                c1[np_rows] = ps->stats[idx][1];
                c2[np_rows] = ps->stats[idx][2];
                c3[np_rows] = ps->stats[idx][3];
                np_rows++;
            }
        }
    }
    if (np_rows > 0) {
        double s0 = np_sum(c0, np_rows);
        double s1 = np_sum(c1, np_rows);
        double denom = s1 > 1.0 ? s1 : 1.0;
        out[4] = s0 / denom;
        out[5] = np_sum(c2, np_rows) / (double)np_rows;
        out[6] = np_sum(c3, np_rows) / (double)np_rows;
    } else {
        out[4] = NAN;
        out[5] = NAN;
        out[6] = NAN;
    }

    if (ts->has_last) {
        int64_t diff_s = when_sec - ts->last_sec;
        int64_t days = (diff_s >= 0) ? (diff_s / 86400LL) : ((diff_s - 86399LL) / 86400LL);
        out[7] = (double)days;
    } else {
        out[7] = NAN;
    }

    out[8] = log1p((double)ts->n_maps);
    out[9] = log1p((double)n_map_arr[(size_t)tid * (size_t)n_maps_vocab + (size_t)mid]);

    if (ts->lineup_len > 0) {
        /* Set intersection size len(set(lu) & set(prev)) */
        int32_t uniq_prev[5];
        int n_up = 0;
        for (int i = 0; i < ts->lineup_len; ++i) {
            int32_t x = ts->lineup[i];
            int dup = 0;
            for (int j = 0; j < n_up; ++j) if (uniq_prev[j] == x) { dup = 1; break; }
            if (!dup) uniq_prev[n_up++] = x;
        }
        int inter = 0;
        for (int j = 0; j < n_up; ++j) {
            for (int i = 0; i < n_lu; ++i) {
                if (lu[i] == uniq_prev[j]) { inter++; break; }
            }
        }
        out[10] = (double)inter;
    } else {
        out[10] = NAN;
    }
}

static int find_col(char **hdr, int nhdr, const char *name) {
    for (int i = 0; i < nhdr; ++i) {
        if (strcmp(hdr[i], name) == 0) return i;
    }
    return -1;
}

int c_prematch_features(
    const char *v5_dir, const char *lp_csv,
    int n_files,
    const char **ct_teams, const char **t_teams, const char **demo_maps,
    const int *ct_wins, const int *t_wins,
    double *out_X_fortran, /* shape (n_files, 15) Fortran-contiguous: col * n_files + f */
    int *out_found,        /* 0 or 1 per file */
    int *out_ct_first,     /* 1 if hist[pick].norm(a) == norm(ct) else 0 */
    int *out_kg_cnt,
    int *out_lp_cnt
) {
    StrPool sp;
    sp_init(&sp, 1 << 20);
    TeamRegistry tr;
    tr_init(&tr);
    InternTab map_tab;
    it_init(&map_tab, 128);
    U64Map tier_map;
    u64m_init(&tier_map, 32768);
    U64Map player_map;
    u64m_init(&player_map, 16384);
    int32_t n_players = 0;

    char path[2048];
    char *cols[256];

    /* 1. Read cs2_tier{1,2,3}_games.csv */
    for (int t = 1; t <= 3; ++t) {
        snprintf(path, sizeof(path), "%s/cs2_tier%d_games.csv", v5_dir, t);
        size_t flen = 0;
        char *fbuf = slurp_file(path, &flen);
        if (!fbuf) continue;
        char *cur = fbuf, *end = fbuf + flen;
        int nh = parse_csv_row(&cur, end, cols, 256);
        int i_gid = find_col(cols, nh, "game_id");
        int i_tot = find_col(cols, nh, "is_total");
        if (i_gid >= 0 && i_tot >= 0) {
            int nc;
            while ((nc = parse_csv_row(&cur, end, cols, 256)) > 0) {
                if (i_tot < nc && strcmp(cols[i_tot], "True") == 0) continue;
                if (i_gid < nc && !is_na_str(cols[i_gid])) {
                    double gid = strtod(cols[i_gid], NULL);
                    union { double d; uint64_t u; } ug = { gid };
                    u64m_setdefault(&tier_map, ug.u, (int64_t)t);
                }
            }
        }
        free(fbuf);
    }

    /* 2. Read cs2_all_tiers_games.csv */
    size_t kg_cap = 16384, kg_len = 0;
    HistRow *kg_rows = (HistRow *)malloc(kg_cap * sizeof(HistRow));
    KgLineupStats *kg_stats = (KgLineupStats *)malloc(kg_cap * sizeof(KgLineupStats));

    snprintf(path, sizeof(path), "%s/cs2_all_tiers_games.csv", v5_dir);
    {
        size_t flen = 0;
        char *fbuf = slurp_file(path, &flen);
        if (!fbuf) {
            free(kg_rows); free(kg_stats); u64m_free(&tier_map); u64m_free(&player_map);
            it_free(&map_tab); tr_free(&tr); free(sp.buf);
            return -1;
        }
        char *cur = fbuf, *end = fbuf + flen;
        int nh = parse_csv_row(&cur, end, cols, 256);
        int i_mid = find_col(cols, nh, "match_id");
        int i_gid = find_col(cols, nh, "game_id");
        int i_tot = find_col(cols, nh, "is_total");
        int i_bo  = find_col(cols, nh, "bestOf");
        int i_s1  = find_col(cols, nh, "score1_game");
        int i_s2  = find_col(cols, nh, "score2_game");
        int i_map = find_col(cols, nh, "map_name");
        int i_dt  = find_col(cols, nh, "datetime");
        int i_t1  = find_col(cols, nh, "team1");
        int i_t2  = find_col(cols, nh, "team2");

        int p1_id[5], p2_id[5], p1_st[5][4], p2_st[5][4];
        const char *st_names[4] = {"kills", "deaths", "adr", "kast"};
        char cname[64];
        for (int j = 0; j < 5; ++j) {
            snprintf(cname, sizeof(cname), "team1_player%d_id", j + 1);
            p1_id[j] = find_col(cols, nh, cname);
            snprintf(cname, sizeof(cname), "team2_player%d_id", j + 1);
            p2_id[j] = find_col(cols, nh, cname);
            for (int x = 0; x < 4; ++x) {
                snprintf(cname, sizeof(cname), "team1_player%d_%s", j + 1, st_names[x]);
                p1_st[j][x] = find_col(cols, nh, cname);
                snprintf(cname, sizeof(cname), "team2_player%d_%s", j + 1, st_names[x]);
                p2_st[j][x] = find_col(cols, nh, cname);
            }
        }

        int nc;
        while ((nc = parse_csv_row(&cur, end, cols, 256)) > 0) {
            if (i_tot < nc && strcmp(cols[i_tot], "True") == 0) continue;
            if (i_map >= nc || i_s1 >= nc || i_s2 >= nc) continue;
            if (is_na_str(cols[i_map]) || is_na_str(cols[i_s1]) || is_na_str(cols[i_s2])) continue;
            double s1 = strtod(cols[i_s1], NULL);
            double s2 = strtod(cols[i_s2], NULL);
            double mx = s1 > s2 ? s1 : s2;
            if (mx < 13.0 || s1 == s2) continue;

            if (kg_len >= kg_cap) {
                kg_cap *= 2;
                kg_rows = (HistRow *)realloc(kg_rows, kg_cap * sizeof(HistRow));
                kg_stats = (KgLineupStats *)realloc(kg_stats, kg_cap * sizeof(KgLineupStats));
            }
            HistRow *hr = &kg_rows[kg_len];
            KgLineupStats *ks = &kg_stats[kg_len];

            parse_datetime_str(cols[i_dt], &hr->dt_sec, &hr->day_ord, NULL, 0);
            hr->is_lp = 0;
            hr->mid_int = strtoll(cols[i_mid], NULL, 10);
            hr->mid_str_off = 0;
            hr->gid = (!is_na_str(cols[i_gid])) ? strtod(cols[i_gid], NULL) : INFINITY;
            hr->mno = INFINITY;
            hr->orig_idx = (uint32_t)kg_len;
            hr->t1_id = tr_get_team(&tr, &sp, cols[i_t1]);
            hr->t2_id = tr_get_team(&tr, &sp, cols[i_t2]);
            hr->map_id = it_intern(&map_tab, &sp, cols[i_map], NULL);
            hr->s1 = s1;
            hr->s2 = s2;
            hr->bo = (i_bo >= 0 && i_bo < nc && !is_na_str(cols[i_bo])) ? strtod(cols[i_bo], NULL) : 0.0;
            int64_t tv = 2;
            if (!isinf(hr->gid)) {
                union { double d; uint64_t u; } ug = { hr->gid };
                u64m_get(&tier_map, ug.u, &tv);
            }
            hr->tier = (double)tv;
            hr->kg_idx = (int32_t)kg_len;

            ks->n_la = 0;
            for (int j = 0; j < 5; ++j) {
                if (p1_id[j] >= 0 && p1_id[j] < nc && !is_na_str(cols[p1_id[j]])) {
                    double pid_d = strtod(cols[p1_id[j]], NULL);
                    union { double d; uint64_t u; } up = { pid_d };
                    int32_t pid = (int32_t)u64m_setdefault(&player_map, up.u, (int64_t)n_players);
                    if (pid == n_players) n_players++;
                    int slot = ks->n_la++;
                    ks->la[slot] = pid;
                    int all_ok = 1;
                    for (int x = 0; x < 4; ++x) {
                        int cidx = p1_st[slot][x];
                        if (cidx < 0 || cidx >= nc || is_na_str(cols[cidx])) { all_ok = 0; break; }
                        ks->stats_a[slot][x] = strtod(cols[cidx], NULL);
                    }
                    ks->valid_a[slot] = (uint8_t)all_ok;
                }
            }
            ks->n_lb = 0;
            for (int j = 0; j < 5; ++j) {
                if (p2_id[j] >= 0 && p2_id[j] < nc && !is_na_str(cols[p2_id[j]])) {
                    double pid_d = strtod(cols[p2_id[j]], NULL);
                    union { double d; uint64_t u; } up = { pid_d };
                    int32_t pid = (int32_t)u64m_setdefault(&player_map, up.u, (int64_t)n_players);
                    if (pid == n_players) n_players++;
                    int slot = ks->n_lb++;
                    ks->lb[slot] = pid;
                    int all_ok = 1;
                    for (int x = 0; x < 4; ++x) {
                        int cidx = p2_st[slot][x];
                        if (cidx < 0 || cidx >= nc || is_na_str(cols[cidx])) { all_ok = 0; break; }
                        ks->stats_b[slot][x] = strtod(cols[cidx], NULL);
                    }
                    ks->valid_b[slot] = (uint8_t)all_ok;
                }
            }
            kg_len++;
        }
        free(fbuf);
    }

    qsort(kg_rows, kg_len, sizeof(HistRow), cmp_kg_rows);

    /* Build known: key_id -> first seen team_id in pd.concat([d.team1, d.team2]) */
    U64Map known_map;
    u64m_init(&known_map, 8192);
    for (size_t i = 0; i < kg_len; ++i) {
        int32_t tid = kg_rows[i].t1_id;
        u64m_setdefault(&known_map, (uint64_t)(uint32_t)(tr.meta[tid].key_id + 1), (int64_t)tid);
    }
    for (size_t i = 0; i < kg_len; ++i) {
        int32_t tid = kg_rows[i].t2_id;
        u64m_setdefault(&known_map, (uint64_t)(uint32_t)(tr.meta[tid].key_id + 1), (int64_t)tid);
    }

    /* Build kk set for duplicate filtering: (unordered_pair(t1_id, t2_id), map_id, day_ord) */
    U64Map kk_set;
    u64m_init(&kk_set, 32768);
    int64_t min_kg_sec = (kg_len > 0) ? kg_rows[0].dt_sec : INT64_MIN;
    for (size_t i = 0; i < kg_len; ++i) {
        uint64_t kh = pack_pair_map_day(kg_rows[i].t1_id, kg_rows[i].t2_id, kg_rows[i].map_id, kg_rows[i].day_ord);
        u64m_setdefault(&kk_set, kh, 1);
    }

    /* 3. Read Liquipedia CSV */
    size_t hist_cap = kg_len + 40960;
    HistRow *hist = (HistRow *)malloc(hist_cap * sizeof(HistRow));
    for (size_t i = 0; i < kg_len; ++i) {
        hist[i] = kg_rows[i];
        hist[i].orig_idx = (uint32_t)i;
    }
    free(kg_rows);
    size_t hist_len = kg_len;

    {
        size_t flen = 0;
        char *fbuf = slurp_file(lp_csv, &flen);
        if (!fbuf) {
            free(hist); free(kg_stats); u64m_free(&kk_set); u64m_free(&known_map);
            u64m_free(&tier_map); u64m_free(&player_map); it_free(&map_tab); tr_free(&tr); free(sp.buf);
            return -2;
        }
        char *cur = fbuf, *end = fbuf + flen;
        int nh = parse_csv_row(&cur, end, cols, 256);
        int i_dt   = find_col(cols, nh, "datetime");
        int i_map  = find_col(cols, nh, "map_name");
        int i_s1   = find_col(cols, nh, "score1_game");
        int i_s2   = find_col(cols, nh, "score2_game");
        int i_mno  = find_col(cols, nh, "map_no");
        int i_bo   = find_col(cols, nh, "bestOf");
        int i_t1   = find_col(cols, nh, "team1");
        int i_t2   = find_col(cols, nh, "team2");
        int i_tier = find_col(cols, nh, "tier");

        char iso19[64];
        char mid_buf[1024];
        int nc;
        while ((nc = parse_csv_row(&cur, end, cols, 256)) > 0) {
            if (i_s1 >= nc || i_s2 >= nc || is_na_str(cols[i_s1]) || is_na_str(cols[i_s2])) continue;
            double s1 = strtod(cols[i_s1], NULL);
            double s2 = strtod(cols[i_s2], NULL);
            double mx = s1 > s2 ? s1 : s2;
            if (mx < 13.0 || s1 == s2) continue;

            int64_t dt_sec;
            int32_t day_ord;
            parse_datetime_str(cols[i_dt], &dt_sec, &day_ord, iso19, sizeof(iso19));

            int32_t raw_t1 = tr_get_team(&tr, &sp, cols[i_t1]);
            int32_t raw_t2 = tr_get_team(&tr, &sp, cols[i_t2]);
            int64_t kt1 = raw_t1, kt2 = raw_t2;
            u64m_get(&known_map, (uint64_t)(uint32_t)(tr.meta[raw_t1].key_id + 1), &kt1);
            u64m_get(&known_map, (uint64_t)(uint32_t)(tr.meta[raw_t2].key_id + 1), &kt2);
            int32_t t1_id = (int32_t)kt1;
            int32_t t2_id = (int32_t)kt2;
            int32_t map_id = it_intern(&map_tab, &sp, cols[i_map], NULL);

            if (dt_sec < min_kg_sec) continue;
            int64_t dummy;
            if (u64m_get(&kk_set, pack_pair_map_day(t1_id, t2_id, map_id, day_ord - 1), &dummy) ||
                u64m_get(&kk_set, pack_pair_map_day(t1_id, t2_id, map_id, day_ord), &dummy) ||
                u64m_get(&kk_set, pack_pair_map_day(t1_id, t2_id, map_id, day_ord + 1), &dummy)) {
                continue;
            }

            if (hist_len >= hist_cap) {
                hist_cap *= 2;
                hist = (HistRow *)realloc(hist, hist_cap * sizeof(HistRow));
            }
            const char *t1_s = sp.buf + tr.meta[t1_id].raw_off;
            const char *t2_s = sp.buf + tr.meta[t2_id].raw_off;
            int mlen = snprintf(mid_buf, sizeof(mid_buf), "%s%s%s", iso19, t1_s, t2_s);
            if (mlen < 0) mlen = 0;
            uint32_t m_off = sp_add(&sp, mid_buf, (size_t)mlen);

            HistRow *hr = &hist[hist_len];
            hr->dt_sec = dt_sec;
            hr->day_ord = day_ord;
            hr->is_lp = 1;
            hr->mid_int = 0;
            hr->mid_str_off = m_off;
            hr->gid = 0.0;
            hr->mno = (i_mno >= 0 && i_mno < nc && !is_na_str(cols[i_mno])) ? strtod(cols[i_mno], NULL) : INFINITY;
            hr->orig_idx = (uint32_t)hist_len;
            hr->t1_id = t1_id;
            hr->t2_id = t2_id;
            hr->map_id = map_id;
            hr->s1 = s1;
            hr->s2 = s2;
            hr->bo = (i_bo >= 0 && i_bo < nc && !is_na_str(cols[i_bo])) ? strtod(cols[i_bo], NULL) : 0.0;
            hr->tier = (i_tier >= 0 && i_tier < nc && !is_na_str(cols[i_tier])) ? strtod(cols[i_tier], NULL) : 2.0;
            hr->kg_idx = -1;
            hist_len++;
        }
        free(fbuf);
    }

    *out_kg_cnt = (int)kg_len;
    *out_lp_cnt = (int)(hist_len - kg_len);

    g_sort_sp_buf = sp.buf;
    qsort(hist, hist_len, sizeof(HistRow), cmp_hist_rows);

    /* Intern demo teams & maps BEFORE building state arrays so vocab sizes are final */
    int32_t *demo_nct = (int32_t *)malloc((size_t)(n_files + 1) * sizeof(int32_t));
    int32_t *demo_nt  = (int32_t *)malloc((size_t)(n_files + 1) * sizeof(int32_t));
    int32_t *demo_mid = (int32_t *)malloc((size_t)(n_files + 1) * sizeof(int32_t));
    for (int f = 0; f < n_files; ++f) {
        int32_t t_ct = tr_get_team(&tr, &sp, ct_teams[f]);
        int32_t t_t  = tr_get_team(&tr, &sp, t_teams[f]);
        demo_nct[f] = tr.meta[t_ct].norm_id;
        demo_nt[f]  = tr.meta[t_t].norm_id;
        demo_mid[f] = it_intern(&map_tab, &sp, demo_maps[f], NULL);
    }

    /* Build idx chained list over hist: head_map[pack_norm_pair_map] -> last seen, prev_link[i] -> earlier seen */
    U64Map idx_head;
    u64m_init(&idx_head, 65536);
    int32_t *prev_cand = (int32_t *)malloc((hist_len + 1) * sizeof(int32_t));
    for (size_t i = 0; i < hist_len; ++i) {
        int32_t na = tr.meta[hist[i].t1_id].norm_id;
        int32_t nb = tr.meta[hist[i].t2_id].norm_id;
        uint64_t key = pack_norm_pair_map(na, nb, hist[i].map_id) + 1ULL;
        int64_t old_head = -1;
        if (u64m_get(&idx_head, key, &old_head)) {
            prev_cand[i] = (int32_t)old_head;
            /* overwrite head in idx_head */
            uint32_t mask = idx_head.cap - 1;
            uint32_t slot = (uint32_t)mix64(key) & mask;
            while (idx_head.used[slot]) {
                if (idx_head.keys[slot] == key) { idx_head.vals[slot] = (int64_t)i; break; }
                slot = (slot + 1) & mask;
            }
        } else {
            prev_cand[i] = -1;
            u64m_setdefault(&idx_head, key, (int64_t)i);
        }
    }

    int32_t *file_pick = (int32_t *)malloc((size_t)(n_files + 1) * sizeof(int32_t));
    uint8_t *needed = (uint8_t *)calloc(hist_len + 1, 1);
    int64_t max_need = -1;

    for (int f = 0; f < n_files; ++f) {
        int32_t nct = demo_nct[f], nt = demo_nt[f], m = demo_mid[f];
        uint64_t key = pack_norm_pair_map(nct, nt, m) + 1ULL;
        int64_t head = -1;
        if (!u64m_get(&idx_head, key, &head) || head < 0) {
            file_pick[f] = -1;
            out_found[f] = 0;
            out_ct_first[f] = 0;
            continue;
        }
        /* Walk backwards from latest candidate: first exact match wins, else head (latest overall)! */
        double w_ct = (double)ct_wins[f], w_t = (double)t_wins[f];
        int32_t chosen = (int32_t)head;
        for (int32_t cur_i = (int32_t)head; cur_i >= 0; cur_i = prev_cand[cur_i]) {
            int32_t na = tr.meta[hist[cur_i].t1_id].norm_id;
            int32_t nb = tr.meta[hist[cur_i].t2_id].norm_id;
            double s1 = hist[cur_i].s1, s2 = hist[cur_i].s2;
            int exact = 0;
            if (na != nb && nct != nt) {
                exact = ((na == nct && nb == nt && s1 == w_ct && s2 == w_t) ||
                         (na == nt && nb == nct && s1 == w_t && s2 == w_ct));
            } else if (na == nb && nct == nt && na == nct) {
                exact = (s2 == w_t);
            }
            if (exact) {
                chosen = cur_i;
                break;
            }
        }
        file_pick[f] = chosen;
        out_found[f] = 1;
        out_ct_first[f] = (tr.meta[hist[chosen].t1_id].norm_id == nct) ? 1 : 0;
        needed[chosen] = 1;
        if (chosen > max_need) max_need = chosen;
    }

    /* 4. Replay State up to group containing max_need */
    int32_t n_teams = tr.raw_tab.count;
    int32_t n_maps = map_tab.count;
    TeamState *ts_arr = (TeamState *)calloc((size_t)(n_teams + 1), sizeof(TeamState));
    for (int32_t t = 0; t < n_teams; ++t) ts_arr[t].elo = 1500.0;

    size_t tm_sz = (size_t)(n_teams + 1) * (size_t)(n_maps + 1);
    double *map_elo_arr = (double *)malloc(tm_sz * sizeof(double));
    for (size_t i = 0; i < tm_sz; ++i) map_elo_arr[i] = 1500.0;
    int32_t *n_map_arr = (int32_t *)calloc(tm_sz, sizeof(int32_t));
    PlayerState *ps_arr = (PlayerState *)calloc((size_t)(n_players + 1), sizeof(PlayerState));

    U64Map h2h_map;
    u64m_init(&h2h_map, 65536);
    size_t h2h_cap = 16384, h2h_cnt = 0;
    H2HList *h2h_lists = (H2HList *)calloc(h2h_cap, sizeof(H2HList));

    /* Store 15 features for needed hist indices via U64Map hist_idx -> feat_slot */
    U64Map feat_slot_map;
    u64m_init(&feat_slot_map, 1024);
    double *needed_feats = (double *)malloc((size_t)(n_files + 16) * 15 * sizeof(double));
    int32_t n_feat_slots = 0;

    size_t pos = 0;
    while (pos < hist_len && (int64_t)pos <= max_need) {
        size_t end = pos + 1;
        while (end < hist_len &&
               hist[end].dt_sec == hist[pos].dt_sec &&
               hist[end].is_lp == hist[pos].is_lp &&
               ((hist[pos].is_lp == 0)
                ? (hist[end].mid_int == hist[pos].mid_int)
                : (strcmp(sp.buf + hist[end].mid_str_off, sp.buf + hist[pos].mid_str_off) == 0))) {
            end++;
        }

        /* Pass 1: Features for needed rows in this match */
        for (size_t i = pos; i < end; ++i) {
            if (!needed[i]) continue;
            const HistRow *r = &hist[i];
            int32_t a = r->t1_id, b = r->t2_id, m = r->map_id;
            const int32_t *la = ts_arr[a].lineup;
            int n_la = ts_arr[a].lineup_len;
            const int32_t *lb = ts_arr[b].lineup;
            int n_lb = ts_arr[b].lineup_len;
            if (r->kg_idx >= 0) {
                const KgLineupStats *ks = &kg_stats[r->kg_idx];
                if (ks->n_la > 0) { la = ks->la; n_la = ks->n_la; }
                if (ks->n_lb > 0) { lb = ks->lb; n_lb = ks->n_lb; }
            }
            double sa[11], sb[11];
            eval_side_feat(a, m, la, n_la, r->dt_sec, ts_arr, n_maps, map_elo_arr, n_map_arr, ps_arr, sa);
            eval_side_feat(b, m, lb, n_lb, r->dt_sec, ts_arr, n_maps, map_elo_arr, n_map_arr, ps_arr, sb);

            int32_t fslot = n_feat_slots++;
            u64m_setdefault(&feat_slot_map, (uint64_t)(i + 1), (int64_t)fslot);
            double *dst = needed_feats + (size_t)fslot * 15;
            for (int k = 0; k < 11; ++k) {
                double d = sa[k] - sb[k];
                dst[k] = isnan(d) ? 0.0 : d;
            }
            uint64_t hk = pack_u32_pair(a, b) + 1ULL;
            int64_t hidx = -1;
            if (u64m_get(&h2h_map, hk, &hidx) && hidx >= 0 && h2h_lists[hidx].len > 0) {
                const H2HList *hl = &h2h_lists[hidx];
                double *w_tmp = (double *)malloc((size_t)hl->len * sizeof(double));
                for (int32_t q = 0; q < hl->len; ++q) w_tmp[q] = (hl->winners[q] == a) ? 1.0 : 0.0;
                dst[11] = (np_sum(w_tmp, hl->len) / (double)hl->len) - 0.5;
                dst[12] = log1p((double)hl->len);
                free(w_tmp);
            } else {
                dst[11] = 0.0;
                dst[12] = 0.0;
            }
            dst[13] = isnan(r->tier) ? 2.0 : r->tier;
            dst[14] = isnan(r->bo) ? 0.0 : r->bo;
        }

        /* Pass 2: Update State for all rows in this match */
        for (size_t i = pos; i < end; ++i) {
            const HistRow *r = &hist[i];
            int32_t a = r->t1_id, b = r->t2_id, m = r->map_id;
            int y = (r->s1 > r->s2) ? 1 : 0;

            double de = 32.0 * ((double)y - 1.0 / (1.0 + pow(10.0, -(ts_arr[a].elo - ts_arr[b].elo) / 400.0)));
            ts_arr[a].elo += de;
            ts_arr[b].elo -= de;

            size_t ka = (size_t)a * (size_t)n_maps + (size_t)m;
            size_t kb = (size_t)b * (size_t)n_maps + (size_t)m;
            double dm = 32.0 * ((double)y - 1.0 / (1.0 + pow(10.0, -(map_elo_arr[ka] - map_elo_arr[kb]) / 400.0)));
            map_elo_arr[ka] += dm;
            map_elo_arr[kb] -= dm;

            double rd = r->s1 - r->s2;
            {
                TeamState *ta = &ts_arr[a];
                ta->form_w[ta->form_pos] = (double)y;
                ta->form_r[ta->form_pos] = rd;
                ta->form_pos = (uint8_t)((ta->form_pos + 1) % 10);
                if (ta->form_len < 10) ta->form_len++;
            }
            {
                TeamState *tb = &ts_arr[b];
                tb->form_w[tb->form_pos] = (double)(1 - y);
                tb->form_r[tb->form_pos] = -rd;
                tb->form_pos = (uint8_t)((tb->form_pos + 1) % 10);
                if (tb->form_len < 10) tb->form_len++;
            }

            uint64_t hk = pack_u32_pair(a, b) + 1ULL;
            int64_t hidx = u64m_setdefault(&h2h_map, hk, (int64_t)h2h_cnt);
            if ((size_t)hidx == h2h_cnt) {
                if (h2h_cnt >= h2h_cap) {
                    size_t ncap = h2h_cap * 2;
                    h2h_lists = (H2HList *)realloc(h2h_lists, ncap * sizeof(H2HList));
                    memset(h2h_lists + h2h_cap, 0, (ncap - h2h_cap) * sizeof(H2HList));
                    h2h_cap = ncap;
                }
                h2h_cnt++;
            }
            H2HList *hl = &h2h_lists[hidx];
            if (hl->len >= hl->cap) {
                hl->cap = hl->cap ? hl->cap * 2 : 4;
                hl->winners = (int32_t *)realloc(hl->winners, (size_t)hl->cap * sizeof(int32_t));
            }
            hl->winners[hl->len++] = y ? a : b;

            ts_arr[a].n_maps++;
            ts_arr[b].n_maps++;
            n_map_arr[ka]++;
            n_map_arr[kb]++;
            ts_arr[a].last_sec = r->dt_sec;
            ts_arr[a].has_last = 1;
            ts_arr[b].last_sec = r->dt_sec;
            ts_arr[b].has_last = 1;

            if (r->kg_idx >= 0) {
                const KgLineupStats *ks = &kg_stats[r->kg_idx];
                for (int j = 0; j < ks->n_la; ++j) {
                    if (ks->valid_a[j]) {
                        PlayerState *ps = &ps_arr[ks->la[j]];
                        memcpy(ps->stats[ps->pos], ks->stats_a[j], 4 * sizeof(double));
                        ps->pos = (uint8_t)((ps->pos + 1) % 20);
                        if (ps->len < 20) ps->len++;
                    }
                }
                for (int j = 0; j < ks->n_lb; ++j) {
                    if (ks->valid_b[j]) {
                        PlayerState *ps = &ps_arr[ks->lb[j]];
                        memcpy(ps->stats[ps->pos], ks->stats_b[j], 4 * sizeof(double));
                        ps->pos = (uint8_t)((ps->pos + 1) % 20);
                        if (ps->len < 20) ps->len++;
                    }
                }
                if (ks->n_la > 0) {
                    memcpy(ts_arr[a].lineup, ks->la, (size_t)ks->n_la * sizeof(int32_t));
                    ts_arr[a].lineup_len = ks->n_la;
                }
                if (ks->n_lb > 0) {
                    memcpy(ts_arr[b].lineup, ks->lb, (size_t)ks->n_lb * sizeof(int32_t));
                    ts_arr[b].lineup_len = ks->n_lb;
                }
            }
        }
        pos = end;
    }

    /* Populate Fortran-contiguous out_X_fortran (n_files x 15) */
    for (int f = 0; f < n_files; ++f) {
        if (!out_found[f]) {
            for (int col = 0; col < 15; ++col) out_X_fortran[(size_t)col * (size_t)n_files + (size_t)f] = 0.0;
        } else {
            int64_t fslot = 0;
            u64m_get(&feat_slot_map, (uint64_t)(file_pick[f] + 1), &fslot);
            const double *src = needed_feats + (size_t)fslot * 15;
            for (int col = 0; col < 15; ++col) {
                out_X_fortran[(size_t)col * (size_t)n_files + (size_t)f] = src[col];
            }
        }
    }

    /* Clean up ALL temporary heap memory! */
    free(needed_feats);
    u64m_free(&feat_slot_map);
    for (size_t i = 0; i < h2h_cnt; ++i) free(h2h_lists[i].winners);
    free(h2h_lists);
    u64m_free(&h2h_map);
    free(ps_arr);
    free(n_map_arr);
    free(map_elo_arr);
    free(ts_arr);
    free(needed);
    free(file_pick);
    free(prev_cand);
    u64m_free(&idx_head);
    free(demo_mid);
    free(demo_nt);
    free(demo_nct);
    free(hist);
    free(kg_stats);
    u64m_free(&kk_set);
    u64m_free(&known_map);
    u64m_free(&player_map);
    u64m_free(&tier_map);
    it_free(&map_tab);
    tr_free(&tr);
    free(sp.buf);
    return 0;
}
