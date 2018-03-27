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

#ifndef BCNN_LAYER_H
#define BCNN_LAYER_H

#include "bcnn/bcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcnn_layer_param {
    int num;
    int size;
    int stride;
    int pad;
    int quantize;
    int net_state;
    bcnn_activation activation;
    bcnn_loss loss;
    bcnn_loss_metric loss_metric;
    float dropout_rate;
    float scale;
    int concat_index;
    float num_constraints;
    int input_size;
    int output_size;
    bcnn_filler_type init;
} bcnn_layer_param;

typedef struct bcnn_layer_base {
    void* param;
    struct bcnn_layer_instance* type;
} bcnn_layer_base;

typedef struct bcnn_layer_instance {
    int param_size;
    void (*initialize)(bcnn_layer_base* layer, bcnn_layer_param* param,
                       bcnn_net* net);
    void (*terminate)(bcnn_layer_base* layer);
    void (*forward)(bcnn_layer_base* layer, bcnn_net* net,
                    bcnn_connection* conn);
    void (*backward)(bcnn_layer_base* layer, bcnn_net* net,
                     bcnn_connection* conn);
    void (*update)(bcnn_layer_base* layer, bcnn_net* net);
} bcnn_layer_instance;

/* Generic constructor */
bcnn_layer_base* bcnn_layer_new(bcnn_layer_instance* type,
                                bcnn_layer_param* param, bcnn_net* net);

/* Generic destructor */
void bcnn_layer_delete(bcnn_layer_base* layer);

void bcnn_layer_initialize(bcnn_layer_base* layer, bcnn_layer_param* param,
                           bcnn_net* net);

void bcnn_layer_udpate(bcnn_layer_base* layer, bcnn_net* net);

void bcnn_layer_terminate(bcnn_layer_base* layer);

void bcnn_layer_forward(bcnn_layer_base* layer, bcnn_net* net,
                        bcnn_connection* conn);

void bcnn_layer_backward(bcnn_layer_base* layer, bcnn_net* net,
                         bcnn_connection* conn);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_LAYER_H
