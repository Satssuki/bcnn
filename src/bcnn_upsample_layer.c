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
#include "bcnn_upsample_layer.h"

#include <bh/bh_string.h>
#include "bcnn_utils.h"

/* Adapted from darknet for yolo v3 port */

bcnn_status bcnn_add_upsample_layer(bcnn_net *net, int size, char *src_id,
                                    char *dst_id) {
    int sz, i;
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        BCNN_CHECK_AND_LOG(
            net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
            "Upsample layer: invalid input node name %s", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }
    // Compute output size according to padding option
    bcnn_tensor_set_shape(&dst_tensor,
                          net->tensors[node.src[0]].n,         // n size
                          net->tensors[node.src[0]].c,         // depth
                          net->tensors[node.src[0]].h * size,  // height
                          net->tensors[node.src[0]].w * size,  // width
                          1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = UPSAMPLE;
    node.layer->size = size;

    sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);

    bcnn_net_add_node(net, node);

    BCNN_INFO(net->log_ctx,
              "[Upsample] input_shape= %dx%dx%d size= %d ouput_shape= "
              "%dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, size, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);
    return 0;
}

int bcnn_forward_upsample_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                    bcnn_tensor *dst_tensor) {
    int dst_sz = bcnn_tensor_size(dst_tensor);
    memset(dst_tensor->data, 0, sizeof(float));
    for (int b = 0; b < src_tensor->n; ++b) {
        for (int k = 0; k < src_tensor->c; ++k) {
            for (int j = 0; j < src_tensor->h * layer->size; ++j) {
                for (int i = 0; i < src_tensor->w * layer->size; ++i) {
                    int src_id =
                        b * src_tensor->w * src_tensor->h * src_tensor->c +
                        k * src_tensor->w * src_tensor->h +
                        (j / layer->size) * src_tensor->w + i / layer->size;
                    int dst_id = b * src_tensor->w * src_tensor->h *
                                     src_tensor->c * layer->size * layer->size +
                                 k * src_tensor->w * src_tensor->h *
                                     layer->size * layer->size +
                                 j * src_tensor->w * layer->size + i;
                    dst_tensor->data[dst_id] = src_tensor->data[src_id];
                }
            }
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_forward_upsample_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_upsample_layer_gpu(node->layer, src, dst);
#else
    return bcnn_forward_upsample_layer_cpu(node->layer, src, dst);
#endif
}

int bcnn_backward_upsample_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                     bcnn_tensor *dst_tensor) {
    for (int b = 0; b < src_tensor->n; ++b) {
        for (int k = 0; k < src_tensor->c; ++k) {
            for (int j = 0; j < src_tensor->h * layer->size; ++j) {
                for (int i = 0; i < src_tensor->w * layer->size; ++i) {
                    int src_id =
                        b * src_tensor->w * src_tensor->h * src_tensor->c +
                        k * src_tensor->w * src_tensor->h +
                        (j / layer->size) * src_tensor->w + i / layer->size;
                    int dst_id = b * src_tensor->w * src_tensor->h *
                                     src_tensor->c * layer->size * layer->size +
                                 k * src_tensor->w * src_tensor->h *
                                     layer->size * layer->size +
                                 j * src_tensor->w * layer->size + i;
                    src_tensor->grad_data[src_id] +=
                        dst_tensor->grad_data[dst_id];
                }
            }
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_backward_upsample_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_upsample_layer_gpu(node->layer, src, dst);
#else
    return bcnn_backward_upsample_layer_cpu(node->layer, src, dst);
#endif
    return 0;
}