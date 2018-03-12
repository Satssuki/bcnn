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

#include "mnist_iterator.h"

#include "bh_log.h"

bcnn_iterator_type mnist_iterator = {
    .param_size = sizeof(mnist_param),
    .initialize = bcnn_iterator_mnist_initialize,
    .next = bcnn_iterator_mnist_next,
    .terminate = bcnn_iterator_mnist_terminate};

static unsigned int read_uint(char *v) {
    unsigned int ret = 0;
    for (int i = 0; i < 4; ++i) {
        ret <<= 8;
        ret |= (unsigned char)v[i];
    }
    return ret;
}

void bcnn_iterator_mnist_initialize(bcnn_iterator *iterator,
                                    bcnn_iterator_param *input_param) {
    FILE *f_img = NULL, *f_label = NULL;
    char tmp[16] = {0};
    int n_img = 0, n_lab = 0, nr = 0;

    mnist_param *iter_param = (mnist_param *)iterator->param;

    f_img = fopen(input_param->data_path, "rb");
    if (f_img == NULL) {
        fprintf(stderr, "[ERROR] Cound not open file %s\n",
                input_param->data_path);
        return -1;
    }
    f_label = fopen(input_param->label_path, "rb");
    if (f_label == NULL) {
        fprintf(stderr, "[ERROR] Cound not open file %s\n",
                input_param->label_path);
        return -1;
    }

    iter_param->f_input = f_img;
    iter_param->f_label = f_label;
    iter_param->n_iter = 0;
    // Read header
    nr = fread(tmp, 1, 16, iter_param->f_input);
    n_img = read_uint(tmp + 4);
    iter_param->input_height = read_uint(tmp + 8);
    iter_param->input_width = read_uint(tmp + 12);
    iter_param->input_depth = 1;
    nr = fread(tmp, 1, 8, iter_param->f_label);
    n_lab = read_uint(tmp + 4);
    bh_check(n_img == n_lab,
             "Inconsistent MNIST data: number of images and labels must be the "
             "same");

    iter_param->input_uchar = (unsigned char *)calloc(
        iter_param->input_width * iter_param->input_height,
        sizeof(unsigned char));
    iter_param->label_int = (int *)calloc(1, sizeof(int));
    rewind(iter_param->f_input);
    rewind(iter_param->f_label);
}

void bcnn_iterator_mnist_next(bcnn_iterator *iterator) {
    char tmp[16];
    unsigned char l;
    unsigned int n_img = 0, n_labels = 0;
    size_t n = 0;
    mnist_param *iter_param = (mnist_param *)iterator->param;

    if (fread((char *)&l, 1, sizeof(char), iter_param->f_input) == 0) {
        rewind(iter_param->f_input);
    } else {
        fseek(iter_param->f_input, -1, SEEK_CUR);
    }
    if (fread((char *)&l, 1, sizeof(char), iter_param->f_label) == 0) {
        rewind(iter_param->f_label);
    } else {
        fseek(iter_param->f_label, -1, SEEK_CUR);
    }

    if (ftell(iter_param->f_input) == 0 && ftell(iter_param->f_label) == 0) {
        n = fread(tmp, 1, 16, iter_param->f_input);
        n_img = _read_int(tmp + 4);
        iter_param->input_height = _read_int(tmp + 8);
        iter_param->input_width = _read_int(tmp + 12);
        n = fread(tmp, 1, 8, iter_param->f_label);
        n_labels = _read_int(tmp + 4);
        bh_check(n_img == n_labels,
                 "MNIST data: number of images and labels must be the same");
        iter_param->n_samples = n_img;
    }

    // Read label
    n = fread((char *)&l, 1, sizeof(char), iter_param->f_label);
    iter_param->label_int[0] = (int)l;
    // Read img
    n = fread(iter_param->input_uchar, 1,
              iter_param->input_width * iter_param->input_height,
              iter_param->f_input);
}

void bcnn_iterator_mnist_terminate(bcnn_iterator *iterator) {
    mnist_param *iter_param = (mnist_param *)iter_param->param;
    if (iter_param->f_input != NULL) {
        fclose(iter_param->f_input);
    }
    if (iter_param->f_label != NULL) {
        fclose(iter_param->f_label);
    }
    bh_free(iter->input_uchar);
    bh_free(iter->label_int);
}