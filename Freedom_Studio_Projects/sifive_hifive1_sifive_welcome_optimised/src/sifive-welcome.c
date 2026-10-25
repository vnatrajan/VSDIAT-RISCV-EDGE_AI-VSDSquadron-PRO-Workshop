/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <stdint.h>
#include <metal/cpu.h>
#include <metal/led.h>
#include <metal/button.h>
#include <metal/switch.h>

/* All tables are const -> placed in flash (.rodata), not in the 16 KB RAM */
#include "svm_model_mnist_optimised.h"   /* weights (int8), weight_scale, bias       */
#include "scaler_mnist_optimised.h"      /* mean, inv_scale (float)                  */
#include "mnist_images_optimised.h"      /* test_images (uint8), test_labels         */

#define RTC_FREQ    32768

/*
 * Linear SVM inference: score[c] = bias[c] + sum_i W[c][i] * x_scaled[i]
 *   x_scaled[i] = (pixel[i] - mean[i]) * inv_scale[i]     (StandardScaler)
 *   W[c][i]     = weights[c][i] * weight_scale            (int8 de-quantize)
 * Feature-outer loop: each pixel is scaled once and added to all 10 classes.
 * Nothing is written back into the image, so it can stay in flash.
 */
int predict(const uint8_t *img)
{
    float score[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; ++c)
        score[c] = bias[c];

    for (int i = 0; i < NUM_FEATURES; ++i) {
        float x = ((float)img[i] - mean[i]) * inv_scale[i] * weight_scale;
        for (int c = 0; c < NUM_CLASSES; ++c)
            score[c] += (float)weights[c][i] * x;
    }

    int best = 0;
    for (int c = 1; c < NUM_CLASSES; ++c)
        if (score[c] > score[best]) best = c;
    return best;
}

int main(void)
{
    int correct = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        int predicted = predict(test_images[i]);
        int actual    = test_labels[i];
        correct += (predicted == actual);
        printf("Image %d: Predicted = %d, Actual = %d\n", i, predicted, actual);
    }
    printf("Accuracy: %d/%d\n", correct, NUM_TEST_IMAGES);
    return 0;
}
