""" module implements the networks functionality """

import torch as th
import numpy as np
import timeit
import datetime
import time
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


class ConditionalGenerator(Generator):
    """ Conditional Generator is a special case of a generator
        well nothing special more than just the name. Nevertheless,
        something does lie in name. (Niki is the name
        I can't stop thinking about :blush:)

        args:
            modules: same as for Network
            latent_size: latent size of the Generator (GAN)
                         Note that latent_size also includes the size of the
                         conditional labels
    """
    pass


class Discriminator(Network):
    pass


class ConditionalDiscriminator(Discriminator):
    """
    The conditional variant of the Discriminator which (discriminator)
    is just further down the Network class tree.

    args:
        modules: Note that this list of modules must not contain the final prediction
            layer. This only reduces the spatial dimension to
            (reduced_height x reduced_width) specifically.
        embedding_size: size of the conditional embedding
        last_module: th.nn.Module which makes the conditional prediction
    """

    def __init__(self, modules, last_module):
        super().__init__(modules)

        # attach the last module separately here:
        self.last_module = last_module

        # adding the last projector conv layer which
        # concatenates the text embedding prior to prediction
        # calculation.

    def forward(self, x, embedding):
        """
        The forward pass of the Conditional Discriminator.
        :param x: input images tensor
        :param embedding: conditional vector
        :return: predictions => scores for the inputs
        """
        # obtain the reduced volume:
        reduced_volume = super().forward(x)

        # concatenate the embeddings to reduced_volume here:
        cat = th.unsqueeze(th.unsqueeze(embedding, -1), -1)
        # spatial replication
        cat = cat.expand(cat.shape[0], cat.shape[1],
                         reduced_volume.shape[2], reduced_volume.shape[3])
        final_input = th.cat((reduced_volume, cat), dim=1)

        # apply the last module to obtain the predictions:
        prediction_scores = self.last_module(final_input)

        # return the prediction scores:
        return prediction_scores


class GAN:
    """
    Unconditional GAN

    args:
        gen: Generator object
        dis: Discriminator object
        device: torch.device() for running on GPU or CPU
                default = torch.device("cpu")
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

        # create a global time counter
        global_time = time.time()

        for epoch in range(start, num_epochs + 1):
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
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)

                # optimize the generator:
                # resample from the latent noise
                gan_input = th.randn(images.shape[0],
                                     self.generator.latent_size, 1, 1).to(self.device)
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % int(limit / feedback_factor) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

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

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")

                th.save(self.generator.state_dict(), gen_save_file)
                th.save(self.discriminator.state_dict(), dis_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.generator.eval()
        self.discriminator.eval()


# TODOcomplete implement conditional gan variant of this
# conditional gan implemented

class ConditionalGAN(GAN):
    """
    Conditional GAN. Actually modifies the calls
    for optimize discriminator, optimize generator and train

    args:
        gen: ConditionalGenerator object
        dis: ConditionalDiscriminator object
        device: torch.device() for running on GPU or CPU
                default = torch.device("cpu")
    """

    def __init__(self, gen, dis, device=th.device("cpu")):
        """ constructor for this derived class """

        # some more specific checks here
        assert isinstance(gen, ConditionalGenerator), \
            "gen is not a Conditional Generator"
        assert isinstance(dis, ConditionalDiscriminator), \
            "dis is not a Conditional Discriminator"

        super().__init__(gen, dis, device)

    @staticmethod
    def randomize(correct_labels):
        """
        static helper for mismatching the given labels
        :param correct_labels: input correct labels
        :return: shuffled labels
                 (Note, that this behaviour is not
                 guaranteed to create a mismatch for every sample)
        """
        return correct_labels[np.random.permutation(correct_labels[0]), :]

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn,
                               conditional_vectors, matching_aware=False,
                               randomizer=None):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param loss_fn: loss function to be used (object of GANLoss)
        :param conditional_vectors: for conditional discrimination
        :param matching_aware: boolean for whether to use matching aware discriminator
        :param randomizer: function object for randomizing the conditional vectors.
                           i.e. to mismatch conditional vectors
                           uses the default randomize function here
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.generator(noise).detach()

        loss = loss_fn.conditional_dis_loss(real_batch, fake_samples,
                                            conditional_vectors)

        # if matching aware discrimination is to be used:
        if matching_aware:
            loss += loss_fn.conditional_dis_loss(
                real_batch, real_batch,
                randomizer(conditional_vectors)
                if randomizer is not None
                else self.randomize(conditional_vectors)
            )
            loss = loss / 2

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn,
                           conditional_vectors):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param loss_fn: loss function to be used (object of GANLoss)
        :param conditional_vectors: for conditional discrimination
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.generator(noise)

        loss = loss_fn.conditional_gen_loss(real_batch, fake_samples,
                                            conditional_vectors)

        # optimize discriminator
        gen_optim.zero_grad()
        # retain graph is true for applying regularization on the
        # conditional input
        loss.backward(retain_graph=True)
        gen_optim.step()

        return loss.item()

    def train(self, data, gen_optim, dis_optim, loss_fn,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=36,
              matching_aware=False, mismatcher=None,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):

        # TODO write the documentation for this method
        # This is the limit of procrastination now :D
        # Just note that data here gives image, label (one-hot encoded)
        # in every batch

        # turn the generator and discriminator into train mode
        self.generator.train()
        self.discriminator.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        _, debug_labels = iter(data).next()
        debug_labels = th.unsqueeze(th.unsqueeze(debug_labels, -1), -1).to(self.device)
        fixed_latent_vectors = th.randn(
            num_samples,
            self.generator.latent_size - debug_labels.shape[1],
            1, 1
        ).to(self.device)

        fixed_input = th.cat((fixed_latent_vectors, debug_labels), dim=1)

        # create a global time counter
        global_time = time.time()

        for epoch in range(start, num_epochs + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                expanded_labels = th.unsqueeze(th.unsqueeze(labels, -1), -1)

                latent_input = th.randn(
                    images.shape[0],
                    self.generator.latent_size - expanded_labels.shape[1],
                    1, 1
                ).to(self.device)

                gan_input = th.cat((latent_input, expanded_labels), dim=1)

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn, labels,
                                                       matching_aware, mismatcher)

                # optimize the generator:
                # resample from the latent noise
                latent_input = th.randn(
                    images.shape[0],
                    self.generator.latent_size - expanded_labels.shape[1],
                    1, 1
                ).to(self.device)
                gan_input = th.cat((latent_input, expanded_labels), dim=1)
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn, labels)

                # provide a loss feedback
                if i % int(limit / feedback_factor) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

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

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")

                th.save(self.generator.state_dict(), gen_save_file)
                th.save(self.discriminator.state_dict(), dis_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.generator.eval()
        self.discriminator.eval()
