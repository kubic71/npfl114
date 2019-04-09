# 7f0a197b-bc00-11e7-a937-00505601122b
# 7cf40fc2-b294-11e7-a937-00505601122b
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.regularizers import l2


def rnpa_bottleneck_layer(input_tensor, nb_filters, filter_sz, stage,
                          #init='glorot_normal',
                          reg=0.0, use_shortcuts=True):
    nb_in_filters, nb_bottleneck_filters = nb_filters

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = '+' + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage > 1:  # first activation is just after conv1
        x = BatchNormalization(#axis=1,
                               name=bn_name + 'a')(input_tensor)
        x = Activation('relu', name=relu_name + 'a')(x)
    else:
        x = input_tensor

    x = Convolution2D(
        filters=nb_bottleneck_filters, kernel_size=1,
        #init=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=(conv_name + 'a')
    )(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(
        #axis=1,
        name=bn_name + 'b')(x)
    x = Activation('relu', name=relu_name + 'b')(x)
    x = Convolution2D(
        filters=nb_bottleneck_filters, kernel_size=filter_sz,
        padding='same',
        #init=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=conv_name + 'b'
    )(x)




    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(
        #axis=1,
        name=bn_name + 'c')(x)
    x = Activation('relu', name=relu_name + 'c')(x)
    x = Convolution2D(filters=nb_in_filters, kernel_size=1,
                      #init=init,
                      kernel_regularizer=l2(reg),
                      name=conv_name + 'c'
                      )(x)

    # merge
    if use_shortcuts:
        x = Add()([x, input_tensor])

    return x


# def buildResnet(input_shape=(32, 32, 3), nb_classes=10, layer1_params=(5, 64, 2), res_layer_params=(3, 16, 3), final_layer_params=None, #init='glorot_normal',
#                 reg=0.0001, use_shortcuts=True):
def buildResnet(input_shape=(32, 32, 3), nb_classes=10, layer1_params=(3,128,2), res_layer_params=(3,32,25),
                final_layer_params=None,  # init='glorot_normal',
                reg=0.001, use_shortcuts=True):


    first_l_filter_size, first_l_filters, first_l_stride = layer1_params
    filter_size, filters, stages = res_layer_params

    use_final_conv = (final_layer_params is not None)
    if use_final_conv:
        sz_fin_filters, nb_fin_filters, stride_fin = final_layer_params
        sz_pool_fin = input_shape[1] / (first_l_stride * stride_fin)
    else:
        sz_pool_fin = input_shape[1] / (first_l_stride)

    img_input = Input(shape=input_shape, name='cifar')

    x = Convolution2D(
        filters=first_l_filters, kernel_size=first_l_filter_size,
        padding='same',
        strides=first_l_stride,
        #init=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name='conv0'
    )(img_input)
    x = BatchNormalization(
        #axis=1,
        name='bn0')(x)
    x = Activation('relu', name='relu0')(x)

    for stage in range(1, stages + 1):
        x = rnpa_bottleneck_layer(
            x,
            (first_l_filters, filters),
            filter_size,
            stage,
            #init=init,
            reg=reg,
            use_shortcuts=use_shortcuts
        )

    x = BatchNormalization(#axis=1,
                           name='bnF')(x)
    x = Activation('relu', name='reluF')(x)

    if use_final_conv:
        x = Convolution2D(
            filters=nb_fin_filters, kernel_size=sz_fin_filters,
            padding='same',
            strides=stride_fin,
            #init=init,
            kernel_regularizer=l2(reg),
            name='convF'
        )(x)

    x = AveragePooling2D((sz_pool_fin, sz_pool_fin), name='avg_pool')(x)

    x = Flatten(name='flat')(x)
    x = Dense(nb_classes, activation='softmax', name='fc10')(x)

    return Model(img_input, x, name='rnpa')