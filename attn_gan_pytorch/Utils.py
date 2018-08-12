""" module contains small utils for parsing configurations """

import torch as th


def get_act_fn(fn_name):
    """
    helper for creating the activation function
    :param fn_name: string containing act_fn name
                    currently supports: [tanh, sigmoid, relu, lrelu]
    :return: fn => PyTorch activation function
    """
    fn_name = fn_name.lower()

    if fn_name == "tanh":
        fn = th.nn.Tanh()

    elif fn_name == "sigmoid":
        fn = th.nn.Sigmoid()

    elif fn_name == "relu":
        fn = th.nn.ReLU()

    elif "lrelu" in fn_name:
        negative_slope = float(fn_name.split("(")[-1][:-1])
        fn = th.nn.LeakyReLU(negative_slope=negative_slope)

    else:
        raise NotImplementedError("requested activation function is not implemented")

    return fn


def get_layer(layer):
    """
    static private helper for creating a layer from the given conf
    :param layer: dict containing info
    :return: lay => PyTorch layer
    """
    from attn_gan_pytorch.CustomLayers import SelfAttention, \
        SpectralNorm, IgnoreAttentionMap, FullAttention
    from torch.nn import Sequential, Conv2d, Dropout2d, ConvTranspose2d, BatchNorm2d
    from attn_gan_pytorch.ConfigManagement import parse2tuple

    # lowercase the name
    name = layer.name.lower()

    if name == "conv":
        in_channels, out_channels = parse2tuple(layer.channels)
        kernel_size = parse2tuple(layer.kernel_dims)
        stride = parse2tuple(layer.stride)
        padding = parse2tuple(layer.padding)
        bias = layer.bias
        act_fn = get_act_fn(layer.activation)

        if hasattr(layer, "spectral_norm") and layer.spectral_norm:
            if layer.batch_norm:
                mod_layer = Sequential(
                    SpectralNorm(Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, bias=bias)),
                    BatchNorm2d(out_channels),
                    act_fn
                )
            else:
                mod_layer = Sequential(
                    SpectralNorm(Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, bias=bias)),
                    act_fn
                )
        else:
            if layer.batch_norm:
                mod_layer = Sequential(
                    Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=bias),
                    BatchNorm2d(out_channels),
                    act_fn
                )
            else:
                mod_layer = Sequential(
                    Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=bias),
                    act_fn
                )

    elif name == "conv_transpose":
        in_channels, out_channels = parse2tuple(layer.channels)
        kernel_size = parse2tuple(layer.kernel_dims)
        stride = parse2tuple(layer.stride)
        padding = parse2tuple(layer.padding)
        bias = layer.bias
        act_fn = get_act_fn(layer.activation)

        if hasattr(layer, "spectral_norm") and layer.spectral_norm:
            if layer.batch_norm:
                mod_layer = Sequential(
                    SpectralNorm(ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride, padding, bias=bias)),
                    BatchNorm2d(out_channels),
                    act_fn
                )
            else:
                mod_layer = Sequential(
                    SpectralNorm(ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride, padding, bias=bias)),
                    act_fn
                )
        else:
            if layer.batch_norm:
                mod_layer = Sequential(
                    ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride, padding, bias=bias),
                    BatchNorm2d(out_channels),
                    act_fn
                )
            else:
                mod_layer = Sequential(
                    ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride, padding, bias=bias),
                    act_fn
                )

    elif name == "dropout":
        drop_probability = layer.drop_prob
        mod_layer = Dropout2d(p=drop_probability, inplace=False)

    elif name == "batch_norm":
        channel_num = layer.num_channels
        mod_layer = BatchNorm2d(channel_num)

    elif name == "ignore_attn_maps":
        mod_layer = IgnoreAttentionMap()

    elif name == "self_attention":
        channels = layer.channels
        squeeze_factor = layer.squeeze_factor
        bias = layer.bias

        if hasattr(layer, "activation"):
            act_fn = get_act_fn(layer.activation)
            mod_layer = SelfAttention(channels, act_fn, squeeze_factor, bias)
        else:
            mod_layer = SelfAttention(channels, None, squeeze_factor, bias)

    elif name == "full_attention":
        in_channels, out_channels = parse2tuple(layer.channels)
        kernel_size = parse2tuple(layer.kernel_dims)
        squeeze_factor = layer.squeeze_factor
        stride = parse2tuple(layer.stride)
        use_batch_norm = layer.use_batch_norm
        use_spectral_norm = layer.use_spectral_norm
        padding = parse2tuple(layer.padding)
        transpose_conv = layer.transpose_conv
        bias = layer.bias

        if hasattr(layer, "activation"):
            act_fn = get_act_fn(layer.activation)
            mod_layer = FullAttention(in_channels, out_channels, act_fn,
                                      kernel_size, transpose_conv,
                                      use_spectral_norm, use_batch_norm,
                                      squeeze_factor, stride, padding, bias)
        else:
            mod_layer = FullAttention(in_channels, out_channels, None,
                                      kernel_size, transpose_conv,
                                      use_spectral_norm, use_batch_norm,
                                      squeeze_factor, stride, padding, bias)
    else:
        raise ValueError("unknown layer type requested")

    return mod_layer
