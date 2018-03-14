/*
* Copyright (c) 2016 Jean-Noel Braun.
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

#include "cifar10_iterator.h"

#include "bcnn_utils.h"

bcnn_iterator_type cifar10_iterator = {
    .param_size = sizeof(cifar10_param),
    .initialize = bcnn_iterator_cifar10_initialize,
    .next = bcnn_iterator_cifar10_next,
    .terminate = bcnn_iterator_cifar10_terminate};

void bcnn_iterator_cifar10_initialize(bcnn_iterator *iterator,
                                      bcnn_iterator_param *input_param) {
    FILE *f_bin = NULL;
    f_bin = bh_fopen(input_param->data_path, "rb");
    cifar10_param *iter_param = (cifar10_param *)iterator->param;
    iter_param->label_width = 1;

    iter_param->label_int = (int *)calloc(1, sizeof(int));
    iter_param->input_width = 32;
    iter_param->input_height = 32;
    iter_param->input_depth = 3;
    iter_param->input_uchar = (unsigned char *)calloc(
        iter_param->input_width * iter_param->input_height *
            iter_param->input_depth,
        sizeof(unsigned char));
    iter_param->f_input = f_bin;
}

void bcnn_iterator_cifar10_next(bcnn_iterator *iterator) {
    unsigned char l;
    unsigned int n_img = 0, n_labels = 0;
    size_t n = 0;
    int x, y, k, i;
    char tmp[3072];
    cifar10_param *iter_param = (cifar10_param *)iterator->param;

    if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0) {
        rewind(iter->f_input);
    } else {
        fseek(iter->f_input, -1, SEEK_CUR);
    }

    // Read label
    n = fread((char *)&l, 1, sizeof(char), iter->f_input);
    iter->label_int[0] = (int)l;
    // Read img
    n = fread(tmp, 1,
              iter->input_width * iter->input_height * iter->input_depth,
              iter->f_input);
    // Swap depth <-> spatial dim arrangement
    for (k = 0; k < iter->input_depth; ++k) {
        for (y = 0; y < iter->input_height; ++y) {
            for (x = 0; x < iter->input_width; ++x) {
                iter->input_uchar[(x + iter->input_width * y) *
                                      iter->input_depth +
                                  k] =
                    tmp[iter->input_width * (iter->input_height * k + y) + x];
            }
        }
    }
}

void bcnn_iterator_cifar10_terminate(bcnn_iterator *iterator) {
    cifar10_param *iter_param = (cifar10_param *)iterator->param;
    bh_fclose(iter_param->f_input);
    bh_free(iter->input_uchar);
    bh_free(iter->label_int);
}