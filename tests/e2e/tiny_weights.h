/* Q8.8 weights generated from examples/tiny-cnn/savedmodel_tiny. Do not edit by hand. */
#ifndef TINY_Q88_H
#define TINY_Q88_H
#include <stdint.h>

/* Weights and activations are Q8.8; biases are Q16.16 so they add
   directly into the accumulator. */

extern const int16_t tiny_conv1_w[4][1][5][3];
extern const int32_t tiny_conv1_b[4];
extern const int16_t tiny_conv2_w[8][1][3][4];
extern const int32_t tiny_conv2_b[8];
extern const int16_t tiny_conv3_w[8][1][3][8];
extern const int32_t tiny_conv3_b[8];
extern const int16_t tiny_fc1_w[1][8][8];
extern const int32_t tiny_fc1_b[8];
extern const int16_t tiny_fc2_w[1][8][4];
extern const int32_t tiny_fc2_b[4];

#endif