#ifndef SYSTOLIC_DRV_H
#define SYSTOLIC_DRV_H

#include <stdint.h>
#include <pulp.h>

/*
 * MatMulFIB HWPE driver for PULPissimo.
 *
 * This replaces the old AXI-Lite/local-bank programming model with the native
 * HWPE protocol:
 *
 *   CPU writes A/B tiles to HWPE-visible TCDM scratch
 *       -> READ(ACQUIRE)
 *       -> write context registers
 *       -> WRITE(TRIGGER, 1)
 *       -> poll READ(RUNNING_JOB) until completion counter advances
 *       -> CPU reads C tile from TCDM scratch
 *
 * Hardware tile shape:
 *   A[8 x K] * B[K x 8] -> C[8 x 8]
 *
 * Input format : signed Q8.8  (int16_t)
 * Output format: signed Q16.16 (int32_t)
 */

/* -------------------------------------------------------------------------- */
/* HWPE base address                                                          */
/* -------------------------------------------------------------------------- */

#ifndef SYSTOLIC_HWPE_BASE_ADDR
# ifdef ARCHI_HWPE_ADDR
#  define SYSTOLIC_HWPE_BASE_ADDR ((uint32_t)ARCHI_HWPE_ADDR)
# elif defined(ARCHI_FC_HWPE_ADDR)
#  define SYSTOLIC_HWPE_BASE_ADDR ((uint32_t)ARCHI_FC_HWPE_ADDR)
# elif defined(ARCHI_HWACC_ADDR)
#  define SYSTOLIC_HWPE_BASE_ADDR ((uint32_t)ARCHI_HWACC_ADDR)
# elif defined(ARCHI_FC_HWACC_ADDR)
#  define SYSTOLIC_HWPE_BASE_ADDR ((uint32_t)ARCHI_FC_HWACC_ADDR)
# else
#  define SYSTOLIC_HWPE_BASE_ADDR (0x1A10C000u)
# endif
#endif

/* Standard hwpe_ctrl mandatory registers. */
#define SYSTOLIC_HWPE_REG_TRIGGER      0x000u
#define SYSTOLIC_HWPE_REG_ACQUIRE      0x004u
#define SYSTOLIC_HWPE_REG_FINISHED     0x008u
#define SYSTOLIC_HWPE_REG_JOB_STATUS   0x00Cu
#define SYSTOLIC_HWPE_REG_RUNNING_JOB  0x010u
#define SYSTOLIC_HWPE_REG_SOFT_CLEAR   0x014u

/* MatMulFIB context registers. */
#define SYSTOLIC_HWPE_REG_A_ADDR       0x040u
#define SYSTOLIC_HWPE_REG_B_ADDR       0x044u
#define SYSTOLIC_HWPE_REG_C_ADDR       0x048u
#define SYSTOLIC_HWPE_REG_INNER_SIZE   0x04Cu
#define SYSTOLIC_HWPE_REG_FLAGS        0x050u

#define SYSTOLIC_HWPE_FLAG_EVENT_ENABLE (1u << 0)

/*
 * Reuse the TCDM region already proven by the EQN HWPE flow.
 *
 * For first bring-up we intentionally keep conservative fixed scratch
 * addresses instead of passing arbitrary L2 pointers to the HWPE.
 *
 * The fixed scratch spacing can physically hold more data, but this K128
 * hardware revision intentionally caps one HWPE job at K <= 128 to match the
 * reduced 128-entry local A/B buffers.
 */
#ifndef SYSTOLIC_HWPE_TCDM_BASE
#define SYSTOLIC_HWPE_TCDM_BASE 0x1C010000u
#endif

#define SYSTOLIC_HWPE_TCDM_A_ADDR (SYSTOLIC_HWPE_TCDM_BASE + 0x0000u)
#define SYSTOLIC_HWPE_TCDM_B_ADDR (SYSTOLIC_HWPE_TCDM_BASE + 0x1000u)
#define SYSTOLIC_HWPE_TCDM_C_ADDR (SYSTOLIC_HWPE_TCDM_BASE + 0x3000u)

#define SYSTOLIC_HWPE_ROWS          8
#define SYSTOLIC_HWPE_COLS          8
#define SYSTOLIC_HWPE_RTL_MAX_K     128
#define SYSTOLIC_HWPE_SCRATCH_MAX_K 128

/* Driver return codes. */
#define SYSTOLIC_HWPE_OK             0
#define SYSTOLIC_HWPE_ERR_ACQUIRE   -1
#define SYSTOLIC_HWPE_ERR_TIMEOUT   -2
#define SYSTOLIC_HWPE_ERR_BAD_K     -3
#define SYSTOLIC_HWPE_ERR_BAD_ARG   -4

/*
 * Small state used only to preserve the legacy low-level API
 * (set_size/start/wait/read_result).
 */
static uint32_t systolic_current_inner_size = 0u;
static int      systolic_last_start_rc      = SYSTOLIC_HWPE_OK;
static uint32_t systolic_last_acquire       = 0xffffffffu;
static uint32_t systolic_running_job_before = 0u;
static uint32_t systolic_running_job_expect = 0u;

/* -------------------------------------------------------------------------- */
/* MMIO / memory helpers                                                      */
/* -------------------------------------------------------------------------- */

static inline void systolic_io_barrier(void)
{
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
}

static inline void systolic_hwpe_write32(uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)(uintptr_t)(SYSTOLIC_HWPE_BASE_ADDR + offset) = value;
}

static inline uint32_t systolic_hwpe_read32(uint32_t offset)
{
    return *(volatile uint32_t *)(uintptr_t)(SYSTOLIC_HWPE_BASE_ADDR + offset);
}

static inline void systolic_mem_write32(uint32_t address, uint32_t value)
{
    *(volatile uint32_t *)(uintptr_t)address = value;
}

static inline uint32_t systolic_mem_read32(uint32_t address)
{
    return *(volatile uint32_t *)(uintptr_t)address;
}

/*
 * Write one 16-bit element into a 32-bit-packed TCDM stream.
 *
 * Streamer convention:
 *   word[15:0]  = earlier row-major element
 *   word[31:16] = next row-major element
 */
static inline void systolic_mem_write16_packed(uint32_t base,
                                               uint32_t elem_index,
                                               uint16_t value)
{
    const uint32_t word_addr = base + ((elem_index >> 1) << 2);
    uint32_t word = systolic_mem_read32(word_addr);

    if ((elem_index & 1u) == 0u) {
        word = (word & 0xffff0000u) | (uint32_t)value;
    } else {
        word = (word & 0x0000ffffu) | ((uint32_t)value << 16);
    }

    systolic_mem_write32(word_addr, word);
}

/*
 * Compatibility helper for legacy source code.
 *
 * The old AXI wrapper used WR_REG(0x900, 0) only to force an AXI write-data
 * value. That operation has no meaning in the HWPE/TCDM architecture, so the
 * legacy 0x900 write is intentionally a no-op.
 */
#define WR_REG(off, val)                                                   \
    do {                                                                   \
        if ((uint32_t)(off) != 0x900u) {                                  \
            systolic_hwpe_write32((uint32_t)(off), (uint32_t)(val));       \
        }                                                                  \
    } while (0)

#define RD_REG(off) systolic_hwpe_read32((uint32_t)(off))

static inline void io_barrier(void)
{
    systolic_io_barrier();
}

/* -------------------------------------------------------------------------- */
/* Native HWPE helpers                                                        */
/* -------------------------------------------------------------------------- */

static inline int systolic_hwpe_acquire(uint32_t *job_id)
{
    const uint32_t acquire = systolic_hwpe_read32(SYSTOLIC_HWPE_REG_ACQUIRE);

    systolic_last_acquire = acquire;

    /* hwpe_ctrl_regfile: -1 = all contexts busy, -2 = another PE offloading. */
    if ((acquire == 0xffffffffu) || (acquire == 0xfffffffeu)) {
        return SYSTOLIC_HWPE_ERR_ACQUIRE;
    }

    if (job_id != 0) {
        *job_id = acquire;
    }

    return SYSTOLIC_HWPE_OK;
}

static inline int systolic_hwpe_wait_finished(int timeout_limit)
{
    /*
     * IMPORTANT:
     * Do NOT poll FINISHED (0x008) through apb2per.
     *
     * In hwpe_ctrl_regfile.sv, FINISHED is a read-to-clear counter
     * ("mainly used by SW interrupt handler").  The counter is cleared as soon
     * as the mandatory-register read is observed.  With the registered
     * APB/peripheral response path, the value can therefore be 1 internally
     * one cycle before PREADY/RVALID, then become 0 by the cycle in which the
     * CPU actually accepts the read response.
     *
     * RUNNING_JOB (0x010) has no read side effect.  hwpe_ctrl increments it
     * after true_done, so it is a robust completion indicator for polling.
     */
    while (timeout_limit-- > 0) {
        const uint32_t running =
            systolic_hwpe_read32(SYSTOLIC_HWPE_REG_RUNNING_JOB) & 0xffu;

        if (running == systolic_running_job_expect) {
            systolic_io_barrier();
            return SYSTOLIC_HWPE_OK;
        }
    }

    return SYSTOLIC_HWPE_ERR_TIMEOUT;
}

static inline int systolic_hwpe_start_job(uint32_t inner_size)
{
    uint32_t job_id = 0u;

    if ((inner_size == 0u) ||
        (inner_size > (uint32_t)SYSTOLIC_HWPE_RTL_MAX_K) ||
        (inner_size > (uint32_t)SYSTOLIC_HWPE_SCRATCH_MAX_K)) {
        return SYSTOLIC_HWPE_ERR_BAD_K;
    }

    int rc = systolic_hwpe_acquire(&job_id);
    if (rc != SYSTOLIC_HWPE_OK) {
        return rc;
    }

    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_A_ADDR,
                          SYSTOLIC_HWPE_TCDM_A_ADDR);
    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_B_ADDR,
                          SYSTOLIC_HWPE_TCDM_B_ADDR);
    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_C_ADDR,
                          SYSTOLIC_HWPE_TCDM_C_ADDR);
    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_INNER_SIZE, inner_size);

    /* Polling is used for first bring-up, therefore event generation is off. */
    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_FLAGS, 0u);

    /*
     * Snapshot the non-destructive completion counter before TRIGGER.
     * running_job_id is 8-bit in hwpe_ctrl_regfile.
     */
    systolic_running_job_before =
        systolic_hwpe_read32(SYSTOLIC_HWPE_REG_RUNNING_JOB) & 0xffu;
    systolic_running_job_expect =
        (systolic_running_job_before + 1u) & 0xffu;

    systolic_io_barrier();

    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_TRIGGER, 1u);
    systolic_io_barrier();

    return SYSTOLIC_HWPE_OK;
}

/* -------------------------------------------------------------------------- */
/* Legacy driver API, now backed by HWPE + TCDM                               */
/* -------------------------------------------------------------------------- */

static inline void systolic_reset(void)
{
    systolic_hwpe_write32(SYSTOLIC_HWPE_REG_SOFT_CLEAR, 1u);
    systolic_io_barrier();

    systolic_current_inner_size = 0u;
    systolic_last_start_rc      = SYSTOLIC_HWPE_OK;
    systolic_last_acquire       = 0xffffffffu;
    systolic_running_job_before = 0u;
    systolic_running_job_expect = 0u;
}

static inline void set_size(uint32_t inner_size)
{
    /*
     * Do not write the context register here.
     * HWPE context registers belong to the context selected by ACQUIRE, so
     * INNER_SIZE is programmed in systolic_start(), after ACQUIRE succeeds.
     */
    systolic_current_inner_size = inner_size;
}

static inline void systolic_load_west(int addr, uint16_t data)
{
    if (addr < 0) {
        return;
    }

    systolic_mem_write16_packed(SYSTOLIC_HWPE_TCDM_A_ADDR,
                                (uint32_t)addr,
                                data);
}

static inline void systolic_load_north(int addr, uint16_t data)
{
    if (addr < 0) {
        return;
    }

    systolic_mem_write16_packed(SYSTOLIC_HWPE_TCDM_B_ADDR,
                                (uint32_t)addr,
                                data);
}

static inline void systolic_start(void)
{
    systolic_last_start_rc =
        systolic_hwpe_start_job(systolic_current_inner_size);
}

static inline int systolic_wait_done(int timeout_limit)
{
    if (systolic_last_start_rc != SYSTOLIC_HWPE_OK) {
        return systolic_last_start_rc;
    }

    return systolic_hwpe_wait_finished(timeout_limit);
}

static inline uint32_t systolic_read_result(int row, int col, int total_cols)
{
    (void)total_cols;

    if ((row < 0) || (row >= SYSTOLIC_HWPE_ROWS) ||
        (col < 0) || (col >= SYSTOLIC_HWPE_COLS)) {
        return 0u;
    }

    const uint32_t index =
        (uint32_t)row * (uint32_t)SYSTOLIC_HWPE_COLS + (uint32_t)col;

    return systolic_mem_read32(SYSTOLIC_HWPE_TCDM_C_ADDR + (index << 2));
}

static inline uint32_t systolic_get_last_acquire(void)
{
    return systolic_last_acquire;
}

static inline uint32_t systolic_get_running_job_before(void)
{
    return systolic_running_job_before;
}

static inline uint32_t systolic_get_running_job_expected(void)
{
    return systolic_running_job_expect;
}

static inline uint32_t systolic_get_running_job_now(void)
{
    return systolic_hwpe_read32(SYSTOLIC_HWPE_REG_RUNNING_JOB) & 0xffu;
}

static inline uint32_t systolic_get_job_status(void)
{
    return systolic_hwpe_read32(SYSTOLIC_HWPE_REG_JOB_STATUS);
}

/* -------------------------------------------------------------------------- */
/* Tiled high-level API                                                       */
/* -------------------------------------------------------------------------- */

/*
 * Return-value version used by new HWPE software.
 *
 * A is MxK, B is KxN, Res is MxN.
 * M/N are tiled in 8x8 output blocks.
 */
static inline int systolic_multiply_large_hwpe(int M,
                                               int K,
                                               int N,
                                               int16_t *A,
                                               int16_t *B,
                                               int32_t *Res)
{
    if ((M <= 0) || (N <= 0) || (K <= 0) ||
        (A == 0) || (B == 0) || (Res == 0)) {
        return SYSTOLIC_HWPE_ERR_BAD_ARG;
    }

    if ((K > SYSTOLIC_HWPE_RTL_MAX_K) ||
        (K > SYSTOLIC_HWPE_SCRATCH_MAX_K)) {
        return SYSTOLIC_HWPE_ERR_BAD_K;
    }

    for (int i = 0; i < M; i += SYSTOLIC_HWPE_ROWS) {
        for (int j = 0; j < N; j += SYSTOLIC_HWPE_COLS) {

            systolic_reset();
            set_size((uint32_t)K);

            /*
             * Prefill the 8x8 C scratch tile with a deterministic sentinel.
             * This is a bring-up aid: if the HWPE performs no/partial stores,
             * the CPU sees 0xDEADBEEF instead of uninitialized X data.
             */
            for (int c_init = 0;
                 c_init < (SYSTOLIC_HWPE_ROWS * SYSTOLIC_HWPE_COLS);
                 c_init++) {
                systolic_mem_write32(
                    SYSTOLIC_HWPE_TCDM_C_ADDR + ((uint32_t)c_init << 2),
                    0xDEADBEEFu);
            }

            systolic_io_barrier();

            /* A tile: 8 rows x K columns, row-major. */
            for (int r = 0; r < SYSTOLIC_HWPE_ROWS; r++) {
                for (int k_idx = 0; k_idx < K; k_idx++) {
                    int16_t value = 0;

                    if ((i + r) < M) {
                        value = A[(i + r) * K + k_idx];
                    }

                    systolic_load_west(r * K + k_idx, (uint16_t)value);
                }
            }

            /* B tile: K rows x 8 columns, row-major. */
            for (int k_idx = 0; k_idx < K; k_idx++) {
                for (int c = 0; c < SYSTOLIC_HWPE_COLS; c++) {
                    int16_t value = 0;

                    if ((j + c) < N) {
                        value = B[k_idx * N + (j + c)];
                    }

                    systolic_load_north(k_idx * SYSTOLIC_HWPE_COLS + c,
                                        (uint16_t)value);
                }
            }

            systolic_io_barrier();

            systolic_start();

            int rc = systolic_wait_done(1000000);
            if (rc != SYSTOLIC_HWPE_OK) {
                return rc;
            }

            /* Copy the complete/valid portion of the 8x8 result tile. */
            for (int r = 0; r < SYSTOLIC_HWPE_ROWS; r++) {
                if ((i + r) >= M) {
                    break;
                }

                for (int c = 0; c < SYSTOLIC_HWPE_COLS; c++) {
                    if ((j + c) >= N) {
                        break;
                    }

                    Res[(i + r) * N + (j + c)] =
                        (int32_t)systolic_read_result(r, c,
                                                      SYSTOLIC_HWPE_COLS);
                }
            }
        }
    }

    return SYSTOLIC_HWPE_OK;
}

/*
 * Preserve the original public API so identity.c and later model software can
 * continue to call systolic_multiply_large(...).
 *
 * New bring-up tests should prefer systolic_multiply_large_hwpe(...) because
 * it returns an explicit error code.
 */
static inline void systolic_multiply_large(int M,
                                           int K,
                                           int N,
                                           int16_t *A,
                                           int16_t *B,
                                           int32_t *Res)
{
    (void)systolic_multiply_large_hwpe(M, K, N, A, B, Res);
}

#endif /* SYSTOLIC_DRV_H */
