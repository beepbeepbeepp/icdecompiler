#ifndef SYSTOLIC_DRV_H
#define SYSTOLIC_DRV_H

#include <stdint.h>
#include <pulp.h>

/* Hardware Base Address */
#define SYSTOLIC_BASE  0x1A400000

/* Register Offsets */
#define REG_START      0x000
#define REG_STATUS     0x004
#define MTRX_SIZE      0x008
#define NORTH_ADDR     0x100
#define NORTH_DATA     0x104
#define NORTH_STRB     0x108
#define WEST_ADDR      0x200
#define WEST_DATA      0x204
#define WEST_STRB      0x208
#define OUTPUT_ADDR    0x400
#define OUTPUT_DATA    0x404

/* Driver Macros */
#define WR_REG(off, val) (*(volatile uint32_t*)(SYSTOLIC_BASE + (off)) = (val))
#define RD_REG(off)      (*(volatile uint32_t*)(SYSTOLIC_BASE + (off)))

static inline void io_barrier() {
    // This assembly instruction forces the processor to wait until 
    // all memory/IO transactions are completed.
    __asm__ volatile ("fence" ::: "memory");
}

/* --- Driver API --- */

static inline void systolic_reset() {
    WR_REG(REG_START, 1);
    WR_REG(REG_START, 0);
    WR_REG(REG_START, 1);
}

// static inline void systolic_load_west(int row, int col, uint16_t data, int total_cols) {
//     WR_REG(WEST_ADDR, row * total_cols + col);
//     WR_REG(WEST_DATA, data);
//     WR_REG(WEST_STRB, 1);
// }

static inline void systolic_load_west(int addr, uint16_t data) {
    WR_REG(WEST_ADDR, addr);
    WR_REG(WEST_DATA, data);
    WR_REG(WEST_STRB, 1);
}

// static inline void systolic_load_north(int row, int col, uint16_t data, int total_cols) {
//     WR_REG(NORTH_ADDR, row * total_cols + col);
//     WR_REG(NORTH_DATA, data);
//     WR_REG(NORTH_STRB, 1);
// }

static inline void systolic_load_north(int addr, uint16_t data) {
    WR_REG(NORTH_ADDR, addr);
    WR_REG(NORTH_DATA, data);
    WR_REG(NORTH_STRB, 1);
}

static inline void systolic_start() {
    WR_REG(REG_START, 3);
}

static inline void set_size(uint32_t inner_size) {
		WR_REG(MTRX_SIZE, inner_size);
}

static inline uint32_t systolic_read_result(int row, int col, int total_cols) {
    WR_REG(OUTPUT_ADDR, row * total_cols + col);
    return RD_REG(OUTPUT_DATA);
}

static inline int systolic_wait_done(int timeout_limit) {
    // while (timeout_limit-- > 0) {
    //     if (RD_REG(REG_STATUS) & 0x2) return 0; // Success
    //     // rt_time_wait_us(10);
    // }
    // return -1; // Timeout

    while (!(RD_REG(REG_STATUS) & 0x2)) {
        if (timeout_limit > 0) timeout_limit--;
        else return -1;
    }
    return 0;
}

/**
 * Tiled Matrix Multiplication for Large M and N
 * M, K, N: Dimensions where A is [M x K] and B is [K x N]
 * The hardware processes 8xK blocks of output at a time.
 */
static inline void systolic_multiply_large(int M, int K, int N, int16_t *A, int16_t *B, int32_t *Res) {
    // Loop through Rows of A in steps of 8
    for (int i = 0; i < M; i += 8) {
        // Loop through Cols of B in steps of 8
        for (int j = 0; j < N; j += 8) {
            
            systolic_reset();
            set_size(K); // The hardware handles the full inner dimension

            // 1. Load Strip from A (West) - 8 rows, K columns
            for (int r = 0; r < 8; r++) {
                for (int k_idx = 0; k_idx < K; k_idx++) {
                    int16_t val = 0;
                    // If row exists in A, load it; otherwise pad with 0
                    if (i + r < M) {
                        val = A[(i + r) * K + k_idx];
                    }
                    systolic_load_west(r * K + k_idx, val);
                }
            }

            // 2. Load Strip from B (North) - K rows, 8 columns
            for (int k_idx = 0; k_idx < K; k_idx++) {
                for (int c = 0; c < 8; c++) {
                    int16_t val = 0;
                    // If column exists in B, load it; otherwise pad with 0
                    if (j + c < N) {
                        val = B[k_idx * N + (j + c)];
                    }
                    systolic_load_north(k_idx * 8 + c, val);
                }
            }
            WR_REG(0x900, 0); // Just for zeroing axi.w_data

            // 3. Execute HW
            systolic_start();
            if (systolic_wait_done(10000) != 0) return; // Timeout safety

            // 4. Read back 8x8 result tile into the correct location in Res
            for (int r = 0; r < 8; r++) {
                if (i + r >= M) break; // Don't write outside Res rows
                for (int c = 0; c < 8; c++) {
                    if (j + c >= N) break; // Don't write outside Res cols
                    
                    int32_t hw_val = systolic_read_result(r, c, 8);
                    Res[(i + r) * N + (j + c)] = hw_val;
                }
            }
        }
    }
}

// static inline void systolic_multiply_large(int A_ROWS, int INNER_SIZE, int B_COLS, int16_t *A, int16_t *B, int32_t *Res) {
//     for (int out_row=0; out_row<A_ROWS; out_row+=8) {
//         for (int out_col=0; out_col<B_COLS; out_col+=8) {
//             systolic_reset();
//             set_size(INNER_SIZE);
//             WR_REG(0x900, 0); // Just for zeroing axi.w_data

//             for (int row=0; row<8; row++) {
//                 for (int col=0; col<INNER_SIZE; col++) {
//                     int16_t val = 0;
//                     if((out_row + row) < A_ROWS) val = A[(out_row+row)*INNER_SIZE+col];
//                     systolic_load_west(row*INNER_SIZE+col, val);
//                 }
//             }

//             for (int row=0; row<INNER_SIZE; row++) {
//                 for (int col=0; col<8; col++) {
//                     int16_t val = 0;
//                     if((out_col + col) < B_COLS) val = B[row*B_COLS +col+out_col];
//                     systolic_load_north(row*8 +col, val);
//                 }
//             }
//             WR_REG(0x900, 0); // Just for zeroing axi.w_data

//             systolic_start();
//             if (systolic_wait_done(10000) != 0) return; // Timeout safety

//             for (int row=0; row<8; row++) {
//                 if ((out_row+row) >= A_ROWS) break;
//                 for (int col=0; col<8; col++) {
//                     if ((out_col+col) >= B_COLS) break;

//                     Res[(out_row+row)*B_COLS + (out_col+col)] = systolic_read_result(row,col,8);
//                 }
//             }
//         }
//     }
// }

#endif