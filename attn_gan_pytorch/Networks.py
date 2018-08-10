""" module implements the networks functionality """

import torch as th
import timeit
import os


class Network(th.nn.Module):
    """ General module that creates a Network from the configuration provided
        Extends a PyTorch Module
        args:
            modules: list of PyTorch layers (nn.Modules)
    """

    def __init__(self, modules):
        """ derived constructor """

        # make a call to Module constructor for allowing
        # us to attach required modules
        super().__init__()

        self.model = th.nn.Sequential(*modules)

    def forward(self, x):
        """
        forward computations
        :param x: input
        :return: y => output features volume
        """
        return self.model(x)


class Generator(Network):
    """
    Generator is an extension of a Generic Network

    args:
        modules: same as for Network
        latent_size: latent size of the Generator (GAN)
    """

    def __init__(self, modules, latent_size):
        super().__init__(modules)

        # attach the latent size for the GAN here
        self.latent_size = latent_size


class Discriminator(Network):
    pass


class ConditionalGenerator(Network):
    pass


class ConditionalDiscriminator(Network):
    pass


class GAN:
    """
    Unconditional GAN
    """

    def __init__(self, gen, dis,
                 device=th.device("cpu")):
        """ constructor for the class """
        assert isinstance(gen, Generator), "gen is not an Unconditional Generator"
        assert isinstance(dis, Discriminator), "dis is not an Unconditional Discriminator"

        # define the state of the object
        self.generator = gen.to(device)
        self.discriminator = dis.to(device)
        self.device = device

        # by default the generator and discriminator are in eval mode
        self.generator.eval()
        self.discriminator.eval()

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: (B x H x W x C)
        """
        noise = th.randn(num_samples, self.generator.latent_size).to(self.device)
        generated_images = self.generator(noise).detach()

        # reshape the generated images
        generated_images = generated_images.permute(0, 2, 3, 1)

        return generated_images

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.generator(noise).detach()

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.generator(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        return loss.item()

    @staticmethod
    def create_grid(samples, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from numpy import sqrt

        samples = th.clamp((samples / 2) + 0.5, min=0, max=1)

        # save the images:
        save_image(samples, img_file, nrow=int(sqrt(samples.shape[0])))

    def train(self, data, gen_optim, dis_optim, loss_fn,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=36,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):

        # TODO write the documentation for this method

        # turn the generator and discriminator into train mode
        self.generator.train()
        self.discriminator.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        fixed_input = th.randn(num_samples,
                               self.generator.latent_size, 1, 1).to(self.device)

        for epoch in range(start, num_epochs):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                images = batch.to(self.device)

                gan_input = th.randn(images.shape[0],
                                     self.generator.latent_size, 1, 1).to(self.device)

                # optimize the discriminator:
                gen_optim.zero_grad()
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)

                # optimize the generator:
                # resample from the latent noise
                gan_input = th.randn(images.shape[0],
                                     self.generator.latent_size, 1, 1).to(self.device)
                dis_optim.zero_grad()
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % int(limit / feedback_factor) == 0 or i == 1:
                    print("batch: %d  d_loss: %f  g_loss: %f" % (i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(dis_loss) + "\t" + str(gen_loss) + "\n")

                    # create a grid of samples and save it
                    os.makedirs(sample_dir, exist_ok=True)
                    gen_img_file = os.path.join(sample_dir, "gen_" +
                                                str(epoch) + "_" +
                                                str(i) + ".png")
                    self.create_grid(self.generator(fixed_input).detach(), gen_img_file)

                if i > limit:
                    break

            # calculate the time required for the epoch
            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 1:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")

                th.save(self.generator.state_dict(), gen_save_file)
                th.save(self.discriminator.state_dict(), dis_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.generator.eval()
        self.discriminator.eval()

# TODO implement conditional gan variant of this
