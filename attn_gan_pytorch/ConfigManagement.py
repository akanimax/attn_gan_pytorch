""" Module for reading and parsing configuration files """

import yaml


def get_config(conf_file):
    """
    parse and load the provided configuration
    :param conf_file: configuration file
    :return: conf => parsed configuration
    """
    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)


def parse2tuple(inp_str):
    """
    function for parsing a 2 tuple of integers
    :param inp_str: string of the form: '(3, 3)'
    :return: tuple => parsed tuple
    """
    inp_str = inp_str[1: -1]  # remove the parenthesis
    args = inp_str.split(',')
    args = tuple(map(int, args))

    return args