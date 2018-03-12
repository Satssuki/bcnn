/*
* Copyright (c) 2016-2018 Jean-Noel Braun.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef BCNN_MNIST_ITERATOR_H
#define BCNN_MNIST_ITERATOR_H

#include "base_iterator.h"
#include "bcnn/bcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

bcnn_iterator_type mnist_iterator;

typedef struct mnist_param {
    int n_samples;
    FILE *f_input;
    FILE *f_label;
    int n_iter;
    int input_width;
    int input_height;
    int input_depth;
    unsigned char *input_uchar;
    int label_width;
    int *label_int;
} mnist_param;

void bcnn_iterator_mnist_initialize(bcnn_iterator *layer,
                                    bcnn_iterator_param *param);
void bcnn_iterator_mnist_next(bcnn_iterator *layer);
void bcnn_iterator_mnist_terminate(bcnn_iterator *layer);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_MNIST_ITERATOR_H