""" Module implementing various loss functions """

import torch as th


# TODO Major rewrite: change the interface to use only predictions for real and fake samples
class GANLoss:
    """
    Base class for all losses
    Note that the gen_loss also has
    """

    def __init__(self, device, dis):
        self.device = device
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        raise NotImplementedError("gen_loss method has not been implemented")


class LSGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (((th.mean(self.dis(real_samps)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((th.mean(self.dis(fake_samps)) - 1) ** 2)


class HingeGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        return (th.mean(th.nn.ReLU()(1 - self.dis(real_samps))) +
                th.mean(th.nn.ReLU()(1 + self.dis(fake_samps))))

    def gen_loss(self, real_samps, fake_samps):
        return -th.mean(self.dis(fake_samps))
