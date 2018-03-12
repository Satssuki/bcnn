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

#ifndef BCNN_ITERATOR_H
#define BCNN_ITERATOR_H

#include "bcnn/bcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* data_path;
    const char* label_path;
} bcnn_iterator_param;

typedef struct bcnn_iterator {
    void* param;
    struct bcnn_iterator_type* type;
} bcnn_iterator;

typedef struct bcnn_iterator_type {
    int param_size;
    void (*initialize)(bcnn_iterator* iterator, bcnn_iterator_param* param);
    void (*next)(bcnn_iterator* iterator);
    void (*terminate)(bcnn_iterator* iterator);
} bcnn_iterator_type;

bcnn_iterator* bcnn_iterator_new(bcnn_iterator_type* type,
                                 bcnn_iterator_param* param);
void bcnn_iterator_delete(bcnn_iterator* iterator);

void bcnn_iterator_initialize(bcnn_iterator* iterator,
                              bcnn_iterator_param* param);
void bcnn_iterator_next(bcnn_iterator* iterator);
void bcnn_iterator_terminate(bcnn_iterator* iterator);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_ITERATOR_H