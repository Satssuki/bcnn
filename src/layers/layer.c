#include "layer.h"

// bcnn_layer_create implentation
bcnn_layer_base* bcnn_layer_create(bcnn_layer_instance* layer_type,
                                   bcnn_layer_param* param, bcnn_net* net) {
    bcnn_layer_base* layer =
        (bcnn_layer_base*)calloc(1, sizeof(bcnn_layer_base));
    layer->type = layer_type;
    bcnn_layer_initialize(layer, param, net);
    return layer;
}

void bcnn_layer_initialize(bcnn_layer_base* layer, bcnn_layer_param* param,
                           bcnn_net* net) {
    layer->param = calloc(1, layer->type->param_size);
    layer->type->initialize(layer, param, net);
}

void bcnn_layer_udpate(bcnn_layer_base* layer, bcnn_net* net) {
    layer->type->update(layer, net);
}

void bcnn_layer_terminate(bcnn_layer_base* layer) {
    layer->type->terminate(layer);
    bh_free(layer->param);
}

void bcnn_layer_forward(bcnn_layer_base* layer, bcnn_net* net,
                        bcnn_connection* conn) {
    layer->type->forward(layer, net, conn);
}

void bcnn_layer_backward(bcnn_layer_base* layer, bcnn_net* net,
                         bcnn_connection* conn) {
    layer->type->backward(layer, net, conn);
}

void bcnn_layer_free(bcnn_layer_base* layer) {
    layer->type->terminate(layer);
    bh_free(layer);
}