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

#ifndef BCNN_FULL_CONNECTED_H
#define BCNN_FULL_CONNECTED_H

#include <bcnn/bcnn.h>
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

bcnn_layer_instance full_connected;

void bcnn_full_connected_initialize(bcnn_layer_base *layer,
                                    bcnn_layer_param *param, bcnn_net *net);
void bcnn_full_connected_terminate(bcnn_layer_base *layer);
void bcnn_full_connected_update(bcnn_layer_base *layer, bcnn_net *net);
void bcnn_full_connected_forward(bcnn_layer_base *layer, bcnn_net *net,
                                 bcnn_connection *conn);
void bcnn_full_connected_backward(bcnn_layer_base *layer, bcnn_net *net,
                                  bcnn_connection *conn);

typedef struct full_connected_param {
    int output_size;
    int input_size;
    bcnn_filler_type init;
    bcnn_activation activation;
    int quantize;
    bcnn_tensor weights;
    bcnn_tensor biases;
    float *adam_m;  // Adam optimizer: first moment gradient
    float *adam_v;  // Adam optimizer: second moment gradient
#ifdef BCNN_USE_CUDA
    float *adam_m_gpu;
    float *adam_v_gpu;
#endif
} full_connected_param;

#ifdef __cplusplus
}
#endif

#endif  // BCNN_FULL_CONNECTED_H