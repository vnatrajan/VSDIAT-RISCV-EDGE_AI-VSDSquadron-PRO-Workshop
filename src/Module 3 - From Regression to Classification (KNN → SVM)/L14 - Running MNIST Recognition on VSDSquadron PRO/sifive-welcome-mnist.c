/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <metal/cpu.h>
#include <metal/led.h>
#include <metal/button.h>
#include <metal/switch.h>

#include <math.h>
#include "svm_model_mnist.h"
#include "scaler_mnist.h"
#include "mnist_images.h"

#define RTC_FREQ    32768

void scale_input(float *x) {
	for (int i=0; i<NUM_FEATURES; ++i) {
		x[i] = (x[i] - mean[i]) / scale[i];
	}
}

int predict(float *x) {
    int best_class = 0;
    float max_score = -INFINITY;
    for (int c = 0; c < NUM_CLASSES; ++c){
        float score = bias[0];
        for (int i = 0; i < NUM_FEATURES; i++){
            score+= weights[0][i] * x[i];
        }
        if (score > max_score) {
            max_score = score;
            best_class = c;
        }
    }
    return best_class;
}

void print_float(float val)
{
    int int_part = (int)val;
    int frac_part = (int)((val - int_part) * 100);
    if (frac_part < 0) frac_part *= -1;
    printf("%d.%02d \n", int_part, frac_part);
}

int main() {
    for (int i = 0; i < NUM_TEST_IMAGES; i++){
        int predicted = predict(test_images[i]);
        int actual = test_labels[i];
        printf("Image %d: Predicted = %d, Actual = %d\n", i, predicted, actual);
    }
    return 0;
}

