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
#include "bcnn_fc_layer.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn_activation_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"
#include "bh_log.h"

int bcnn_add_fullc_layer(bcnn_net *net, int output_size, bcnn_filler_type init,
                         bcnn_activation activation, int quantize, char *src_id,
                         char *dst_id) {
    int i;
    bcnn_connection conn = {0};
    int input_size = 0;
    bcnn_node dst_node = {0};

    if (net->nb_connections > 0) {
        int is_src_node_found = 0;
        for (i = net->num_nodes - 1; i >= 0; --i) {
            if (strcmp(net->nodes[i].id, src_id) == 0) {
                bcnn_connection_add_src_node(&conn, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found,
                 "Full-connected layer: invalid input node name %s", src_id);
    } else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = FULL_CONNECTED;

    // Setup output node
    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor,
                          net->nodes[conn.src[0]].tensor.n,  // batch size
                          output_size,                       // depth
                          1,                                 // height
                          1,                                 // width
                          1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    input_size = bcnn_tensor_get_size3d(&net->nodes[conn.src[0]].tensor);
    // Setup layer weights
    bcnn_tensor_create(&conn.layer->weights, 1, 1, 1, input_size * output_size,
                       1);
    bcnn_tensor_filler w_filler = {.range = input_size, .type = init};
    bcnn_tensor_fill(&conn.layer->weights, w_filler);
    // Setup layer biases
    bcnn_tensor_create(&conn.layer->biases, 1, 1, 1, output_size, 1);

    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_get_size(&conn.layer->weights);
        conn.layer->adam_m = (float *)calloc(weights_size, sizeof(float));
        conn.layer->adam_v = (float *)calloc(weights_size, sizeof(float));
    }

#ifdef BCNN_USE_CUDA
    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_get_size(&conn.layer->weights);
        conn.layer->adam_m_gpu =
            bcnn_cuda_memcpy_f32(conn.layer->adam_m, weights_size);
        conn.layer->adam_v_gpu =
            bcnn_cuda_memcpy_f32(conn.layer->adam_v, weights_size);
    }
#endif
    conn.layer->activation = activation;

    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[Connected] input_shape= %dx%dx%d output_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, net->nodes[conn.dst[0]].tensor.w,
        net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return 0;
}

int bcnn_forward_fullc_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                 bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, batch_size = dst.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    int dst_size = bcnn_tensor_get_size3d(&dst);
    int sz = bcnn_tensor_get_size(&dst);

    memset(dst.data, 0, dst_size * batch_size * sizeof(float));

#ifdef BCNN_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, dst_size,
                src_size, 1.0f, src.data, src_size, layer->weights.data,
                src_size, 1.0f, dst.data, dst_size);
#else
    // Original
    bcnn_gemm(0, 1, batch_size, dst_size, src_size, 1.0f, src.data, src_size,
              layer->weights.data, src_size, 1.0f, dst.data, dst_size);
#endif

    for (i = 0; i < batch_size; ++i)
        bcnn_axpy(dst_size, 1, layer->biases.data, dst.data + i * dst_size);

    bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_fullc_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                  bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, batch_size = dst.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    int dst_size = bcnn_tensor_get_size3d(&dst);
    int sz = bcnn_tensor_get_size(&dst);

    bcnn_backward_activation_cpu(dst.data, dst.grad_data, sz,
                                 layer->activation);

    for (i = 0; i < batch_size; ++i) {
        bcnn_axpy(dst_size, 1, dst.grad_data + i * dst_size,
                  layer->biases.grad_data);
    }

#ifdef BCNN_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dst_size, src_size,
                batch_size, 1.0f, dst.grad_data, dst_size, src.data, src_size,
                1.0f, layer->weights.grad_data, src_size);
#else
    // Original
    bcnn_gemm(1, 0, dst_size, src_size, batch_size, 1.0f, dst.grad_data,
              dst_size, src.data, src_size, 1.0f, layer->weights.grad_data,
              src_size);
#endif

    if (src.grad_data) {
#ifdef BCNN_USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size,
                    src_size, dst_size, 1.0f, dst.grad_data, dst_size,
                    layer->weights.data, src_size, 1.0f, src.grad_data,
                    src_size);
#else
        // Original
        bcnn_gemm(0, 0, batch_size, src_size, dst_size, 1.0f, dst.grad_data,
                  dst_size, layer->weights.data, src_size, 1.0f, src.grad_data,
                  src_size);
#endif
    }

    return BCNN_SUCCESS;
}

#ifdef BCNN_USE_CUDA

int bcnn_forward_fullc_layer_gpu(bcnn_layer *layer, bcnn_node *src_node,
                                 bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, batch_size = dst.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    int dst_size = bcnn_tensor_get_size3d(&dst);
    int sz = bcnn_tensor_get_size(&dst);

    bcnn_cuda_fill_f32(dst_size * batch_size, 0.0f, dst.data_gpu, 1);

    bcnn_cuda_gemm(0, 1, batch_size, dst_size, src_size, 1, src.data_gpu,
                   src_size, layer->weights.data_gpu, src_size, 1, dst.data_gpu,
                   dst_size);

    for (i = 0; i < batch_size; ++i) {
        bcnn_cuda_axpy(dst_size, 1, layer->biases.data_gpu, 1,
                       dst.data_gpu + i * dst_size, 1);
    }
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_fullc_layer_gpu(bcnn_layer *layer, bcnn_node *src_node,
                                  bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, batch_size = dst.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    int dst_size = bcnn_tensor_get_size3d(&dst);
    int sz = bcnn_tensor_get_size(&dst);

    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu, sz,
                                 layer->activation);

    for (i = 0; i < batch_size; ++i) {
        bcnn_cuda_axpy(dst_size, 1, dst.grad_data_gpu + i * dst_size, 1,
                       layer->biases.grad_data_gpu, 1);
    }

    bcnn_cuda_gemm(1, 0, dst_size, src_size, batch_size, 1, dst.grad_data_gpu,
                   dst_size, src.data_gpu, src_size, 1,
                   layer->weights.grad_data_gpu, src_size);
    if (src.grad_data_gpu) {
        bcnn_cuda_gemm(0, 0, batch_size, src_size, dst_size, 1,
                       dst.grad_data_gpu, dst_size, layer->weights.data_gpu,
                       src_size, 1, src.grad_data_gpu, src_size);
    }

    return BCNN_SUCCESS;
}
#endif

int bcnn_forward_fullc_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_fullc_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_fullc_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_fullc_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_fullc_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_fullc_layer_cpu(conn->layer, src, dst);
#endif
}