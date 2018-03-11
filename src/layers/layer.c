#include "layer.h"

// bcnn_layer_create implentation
bcnn_layer_base* bcnn_layer_create(bcnn_layer_instance* layer_type,
                                   bcnn_layer_param* param, bcnn_net* net) {
    bcnn_layer_base* layer =
        (bcnn_layer_base*)calloc(1, sizeof(bcnn_layer_base));

    layer->type = layer_type;
    /*layer->initialize = layer_type->initialize;
    layer->terminate = layer_type->terminate;
    layer->forward = layer_type->forward;
    layer->backward = layer_type->backward;
    layer->update = layer_type->update;*/

    bcnn_layer_initialize(layer, param, net);
}

void bcnn_layer_initialize(bcnn_layer_base* layer, bcnn_layer_param* param,
                           bcnn_net* net) {
    if (layer->type->initialize == NULL) {
        return;
    }
    layer->type->initialize(layer, param, net);
}

void bcnn_layer_udpate(bcnn_layer_base* layer, bcnn_net* net) {
    if (layer->type->update == NULL) {
        return;
    }
    layer->type->update(layer, net);
}

void bcnn_layer_terminate(bcnn_layer_base* layer) {
    if (layer->type->terminate == NULL) {
        return;
    }
    layer->type->terminate(layer);
}

void bcnn_layer_forward(bcnn_layer_base* layer, bcnn_net* net,
                        bcnn_connection* conn) {
    if (layer->type->forward == NULL) {
        return;
    }
    layer->type->forward(layer, net, conn);
}

/**
 *
 */
void bcnn_layer_backward(bcnn_layer_base* layer, bcnn_net* net,
                         bcnn_connection* conn) {
    if (layer->type->backward == NULL) {
        return;
    }
    layer->type->backward(layer, net, conn);
}

void bcnn_layer_free(bcnn_layer_base* layer) {
    if (layer->type->terminate == NULL) {
        return;
    }
    layer->type->terminate(layer);
    bh_free(layer);
    // Note: layer undefined
}