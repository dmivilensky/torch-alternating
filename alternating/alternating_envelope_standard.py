# Copyright 2021 (c) Dmitry Pasechnyuk--Vilensky

from .alternating_envelope import AlternatingEnvelope
from typing import Callable


class AlternatingEnvelopeStandard(AlternatingEnvelope):
    def __init__(self, *blocks, period: int = None, verbose: bool = True):
        """
        :param `*blocks`: non-keyword variadic argument for parameters iterators,
        one per block. example: `model.layer1.parameters()`. you can also
        join parameters for different layers in one block using `itertools.chain`

        :param period: number of iterations before next blocks switching

        :param verbose: flag to turn on/off the logging     
        """
        super().__init__(*blocks, period=period, verbose=verbose)
        # starts the first round
        self.swap()

    def swap(self, closure: Callable[[], float] = None) -> None:
        self._set_mask()
        # chooses the next block circularly
        self.current_block = (self.current_block + 1) % len(self.param_blocks)

    def step(self, closure: Callable[[], float] = None, blank: bool = False) -> None:
        """
        procedure to call at every iteration of the training cycle
        to perform alternating: combines all the neccesary updates
        and calculations and maintains event loop of auxiliary problems

        :param blank: flag to turn of the updates but continue counting iterations
        """
        super().step(closure=closure)

        if (self.iteration + 1) % self.period == 0 and not blank:
            self.swap()
