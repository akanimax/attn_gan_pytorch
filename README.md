# attn_gan_pytorch
python package for self-attention gan implemented as 
extension of PyTorch nn.Module. 
paper -> https://arxiv.org/abs/1805.08318 <br>

Also includes generic layers for image based attention mechanism.
Includes a **`Full-Attention`** layer as proposed by in another 
project of mine [here](https://github.com/akanimax/fagan)

## Installation:
This is a python package availbale at the 
[**pypi.org**](https://pypi.org/project/attn-gan-pytorch/#description).
So, installation is fairly straightforward. This package depends on 
a suitable GPU version of **`torch`** and **`torch-vision`** for your
architecture. So, please download suitable pytorch prior to installing
this package. Follow the instructions at 
[pytorch.org](https://pytorch.org/) to install your version of PyTorch.
<br><br>
Install with following commands:

    $ workon [your virtual environment] 
    $ pip install attn-gan-pytorch

## Celeba Samples:
some celeba samples generated using this code for the 
fagan architecture:
<p align="center">
<img alt="generated samples" src="https://github.com/akanimax/fagan/blob/master/samples/video_gif/relativistic.gif"/>
</p>

### Head over to the [**Fagan project**](https://github.com/akanimax/fagan) repo for more info!
Also, this repo contains the code for using this package 
to build the `SAGAN` architecture as mentioned in the paper.
Please refer the `samples/` directory for this.

## Thanks
Please feel free to open PRs here if you train on other datasets 
using this package. Suggestions / Issues / Contributions are most 
welcome.

Best regards, <br>
@akanimax :)
