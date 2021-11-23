# Copyright 2021 (c) Dmitry Pasechnyuk--Vilensky

import torch
import numpy
from typing import List, Callable, Iterable
from functools import reduce


class AlternatingEnvelope:
    def __init__(self, *blocks, period: int = None, verbose: bool = True):
        self.param_blocks = []
        param_blocks = list(blocks)

        # stores all `the param_group`s for every block as in Optimizer
        for params in param_blocks:
            block_param_groups = []
            param_groups = list(params)

            if len(param_groups) == 0:
                raise ValueError("alternating envelope got an empty block")
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]

            for param_group in param_groups:
                self.add_param_block(block_param_groups, param_group)

            self.param_blocks.append(block_param_groups)

        self.verbose = verbose
        self.current_block = 0
        self.iteration = 0
        self.period = period
        self._define_mask()

    def swap(self, closure: Callable[[], float] = None):
        raise NotImplementedError

    def _define_mask(self):
        length = 0
        for block_params in self.param_blocks:
            for group in block_params:
                for p in group['params']:
                    length += reduce(lambda x,y: x*y, list(p.size()))
        self.masks = [numpy.ones(length)]
    
    def mask_size(self):
        return self.masks[0].shape
    
    def set_masks(self, masks):
        self.masks = masks
    
    def _set_mask(self) -> None:
        """
        freezes gradient backpropagation for all the blocks except the chosen one
        """
        current = 0
        for block, block_params in enumerate(self.param_blocks):
            for i, group in enumerate(block_params):
                for j, p in enumerate(group['params']):
                    length = reduce(lambda x,y: x*y, list(p.size()))
                    if p.grad is not None:
                        p.grad.mul_(torch.from_numpy(
                            self.masks[self.current_block][current:current+length]
                        ).reshape(list(p.size())))
                    current += length

    def step(self, closure: Callable[[], float] = None) -> None:
        # iterations counter for the periodic response
        self.iteration += 1

        if self.period is None or self.period == 0:
            raise ValueError("the period of alternation is not specified!")

    def add_param_block(self, block_param_groups: List[dict], param_group: dict) -> None:
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('alternating parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        block_param_groups.append(param_group)
