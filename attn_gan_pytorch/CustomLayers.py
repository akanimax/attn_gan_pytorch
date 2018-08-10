""" Module implements the custom layers """

import torch as th


class SelfAttention(th.nn.Module):
    """
    Layer implements the self-attention module
    which is the main logic behind this architecture.

    args:
        in_channels: number of input channels
        out_channels: number of output channels
        activation: activation function to be applied (default: lrelu(0.2))
        kernel_size: kernel size for convolution (default: (1 x 1))
        squeeze_factor: squeeze factor for query and keys (default: 8)
        stride: stride for the convolutions (default: 1)
        padding: padding for the applied convolutions (default: 1)
        bias: whether to apply bias or not (default: True)
    """

    def __init__(self, in_channels, out_channels,
                 activation=None, kernel_size=(1, 1),
                 squeeze_factor=8, stride=1, padding=0, bias=True):
        """ constructor for the layer """

        from torch.nn import Conv2d, Parameter, Softmax

        # base constructor call
        super().__init__()

        # state of the layer
        self.activation = activation
        self.gamma = Parameter(th.zeros(1))

        # Modules required for computations
        self.query_conv = Conv2d(  # query convolution
            in_channels=in_channels,
            out_channels=in_channels // squeeze_factor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.key_conv = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // squeeze_factor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.value_conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # softmax module for applying attention
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        forward computations of the layer
        :param x: input feature maps (B x C x H x W)
        :return:
            out: self attention value + input feature (B x O x H x W)
            attention: attention map (B x C x H x W)
        """

        # extract the shape of the input tensor
        m_batchsize, c, height, width = x.size()

        # create the query projection
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (N) x C

        # create the key projection
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B x C x (N)

        # calculate the attention maps
        energy = th.bmm(proj_query, proj_key)  # energy
        attention = self.softmax(energy)  # attention (B x (N) x (N))

        # create the value projection
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        # calculate the output
        out = th.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, height, width)

        attention = attention.view(m_batchsize, -1, height, width)

        if self.activation is not None:
            out = self.activation(out)

        out = self.gamma * out + x
        return out, attention


class SpectralNorm(th.nn.Module):
    """
    Wrapper around a Torch module which applies spectral Normalization
    """

    # TODO complete the documentation for this Layer

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    @staticmethod
    def l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(th.mv(th.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(th.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        from torch.nn import Parameter

        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class IgnoreAttentionMap(th.nn.Module):
    """
    A petty module to ignore the attention
    map output by the self_attention layer
    """

    def __init__(self):
        """ has nothing and does nothing apart from super calls """
        super().__init__()

    def forward(self, inp):
        """
        ignores the attention_map the obtained input. and returns the features
        :param inp: (features, attention_maps)
        :return: output => features
        """
        return inp[0]
