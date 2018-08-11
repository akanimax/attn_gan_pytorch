from setuptools import setup, find_packages

setup(
    name='attn_gan_pytorch',
    version='0.3',
    packages=find_packages(exclude=("samples/", "literature")),
    url='https://github.com/akanimax/attn_gan_pytorch',
    license='MIT',
    author='animesh karnewar',
    author_email='animeshsk3@gmail.com',
    description='python package for self-attention gan implemented as extension of ' +
                'PyTorch nn.Module. paper -> https://arxiv.org/abs/1805.08318',
    install_requires=['torch', 'torchvision', 'numpy', 'PyYAML']
)
