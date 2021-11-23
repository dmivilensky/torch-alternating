# Copyright 2021 (c) Dmitry Pasechnyuk--Vilensky

from .alternating_envelope import AlternatingEnvelope
import enum
import numpy
import random
import scipy.optimize
import torch
from typing import List, Callable

class SwitchingStrategy(enum.Enum):
    SUBSEQUENT = enum.auto()
    RANDOM = enum.auto()
    MAX_GRAD = enum.auto()
    MIN_SCALAR = enum.auto()
    BEST_CONVEX = enum.auto()


def norm(v: numpy.ndarray, z: float = 1) -> numpy.ndarray:
    """
    projection onto the simplex from
    https://gist.github.com/mblondel/6f3b7aaad90606b98f71

    :param v: point to be projected
    :param z: scale of simplex (sum of components values)
    """
    n_features = v.shape[0]
    u = numpy.sort(v)[::-1]
    cssv = numpy.cumsum(u) - z
    ind = numpy.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = numpy.maximum(v - theta, 0)
    return w


def random_from_simplex(n: int) -> numpy.ndarray:
    """
    random sample drawn from uniform
    distribution on the unit simplex

    :param n: dimensionality of random vector to draw
    """
    k = numpy.random.exponential(scale=1.0, size=n)
    return k / numpy.sum(k)


class AlternatingEnvelopeAccelerated(AlternatingEnvelope):
    """
    Implementation of the 'Accelerated Alternating Minimization' algorithm
    from https://arxiv.org/pdf/1906.03622.pdf (Algorithm 1)
    with some modificaitions on block choosing policies
    """

    def __init__(
        self, *blocks, period: int = None, closure: Callable[[], float] = None,
        switching_strategy: SwitchingStrategy = SwitchingStrategy.MAX_GRAD,
        verbose: bool = True
    ):
        """
        :param `*blocks`: non-keyword variadic argument for parameters iterators,
        one per block. example: `model.layer1.parameters()`. you can also
        join parameters for different layers in one block using `itertools.chain`

        :param period: number of iterations before next blocks switching

        :param closure: function without arguments returning the loss,
        evaluated on all the data points; having boolean
        keyword-argument-flag `backward` to turn on/off backpropagation;
        necessarly zeroing gradients before all

        :param switching_strategy: one from the corresponding enum specifing
        the policy of blocks switching and recalculation processing

        :param verbose: flag to turn on/off the logging     
        """
        super().__init__(*blocks, period=period, verbose=verbose)
        self.switching_strategy = switching_strategy

        self.A = 0.0
        self.y = dict()
        self.v = dict()
        # initializing `v_0` and `y_0` with `x_0`
        # storing them in indices-tensor dict format for easy assignments
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        # index `block` is useful when many blocks include the same parameter
                        self.v[(block, i, j)] = torch.clone(p)
                        self.y[(block, i, j)] = torch.clone(p)
        
        # for the following strategies we need to store
        # the state after each block-wise auxiliary problem
        if self.switching_strategy in [SwitchingStrategy.MIN_SCALAR, SwitchingStrategy.BEST_CONVEX]:
            self.blockwise = []
        
        # starts the first round
        self.swap(closure)

    def __set_y(self, coeff: float) -> None:
        """
        sets `y_k` to the convex combination of `v_k` and `x_k`
        with coefficients `coeff` and `1-coeff` correspondingly 
        """
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        self.y[(block, i, j)] = p + coeff * \
                            (self.v[(block, i, j)] - p)

    def __set_params(self) -> None:
        """
        sets `x_k` to the `y_k`
        """
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        p.set_(torch.clone(self.y[(block, i, j)]))

    def __squared_norm(self) -> float:
        """
        calculates squared l2 norm of the gradient in current point
        """
        result = 0.0
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        if p.grad is not None:
                            result += torch.norm(p.grad, p=2).item() ** 2
        return result

    def update(self, closure: Callable[[], float] = None) -> None:
        """
        performs the update of `v_k` sequence
        """
        # evaluates loss in latest obtained point `x_{k+1}`
        fx = closure(backward=False)

        # temporary point to return to after all
        tmp = dict()
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        tmp[(block, i, j)] = torch.clone(p)
        
        # sets params to `y_k` still stored in `self.y`
        self.__set_params()
        # frees all the gradients and backpropagate them
        # to calculate loss and the full gradient in `y_k`
        self.takeoff()
        fy = closure()

        # solving quadratic equation on `a_{k+1}`
        # `fx - fy` must be negative
        # `self.a` must be positive
        if len(numpy.roots([self.__squared_norm() / 2, fx - fy, self.A*(fx - fy)])) == 0:
            self.a = 0
        else:
            self.a = numpy.max(numpy.roots([self.__squared_norm() / 2, fx - fy, self.A*(fx - fy)]))
        print(self.a)
        self.A += self.a

        # preforms the gradient step for `v_k`
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        if p.grad is not None:
                            self.v[(block, i, j)].sub_(p.grad, alpha=self.a)
        
        # returns to the temporary point
        with torch.no_grad():
                for block, block_params in enumerate(self.param_blocks):
                    for i, group in enumerate(block_params):
                        for j, p in enumerate(group['params']):
                            p.set_(torch.clone(tmp[(block, i, j)]))

    def swap(self, closure: Callable[[], float]) -> None:
        """
        performs the update of `y_k` sequence and swaps block to optimize
        according to the chosen policy `self.switching_strategy`
        """
        # temporary point to return to after the every `self.__set_param`
        # call for function value calculation
        tmp = dict()
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        tmp[(block, i, j)] = torch.clone(p)

        # auxiliary function to optimize w.r.t. coefficient `beta`,
        # that evaluates loss in `y_k` set to the corresponding convex combination
        def segment(beta: float) -> float:
            self.__set_y(beta)
            # sets params to `y_k` to evaluate the model in it
            self.__set_params()
            value = closure(backward=False)
            # returns to the temporary point
            with torch.no_grad():
                for block, block_params in enumerate(self.param_blocks):
                    for i, group in enumerate(block_params):
                        for j, p in enumerate(group['params']):
                            p.set_(torch.clone(tmp[(block, i, j)]))
            return value

        # finds optimal [convex] coefficient
        beta = scipy.optimize.minimize_scalar(segment, bounds=(0.0, 1.0), method='bounded').x
        self.__set_y(beta)
        # sets `x_k` to the `y_k` as the starting point, just in case
        self.__set_params()

        if self.switching_strategy == SwitchingStrategy.SUBSEQUENT:
            # chooses the next block circularly
            self.current_block = (self.current_block +
                                  1) % len(self.param_blocks)
            if self.verbose:
                print("swap to", self.current_block)

        elif self.switching_strategy == SwitchingStrategy.RANDOM:
            self.current_block = random.randint(0, len(self.param_blocks) - 1)
            if self.verbose:
                print("swap to", self.current_block)

        elif self.switching_strategy == SwitchingStrategy.MAX_GRAD:
            # frees all the gradients and backpropagate them
            # to calculate the full gradient
            self.takeoff()
            closure()

            # calculates squared l2 norm of the gradient in `y_k`
            df_block_squared_norm = []
            with torch.no_grad():
                for block, block_params in enumerate(self.param_blocks):
                    result = 0.0
                    for i, group in enumerate(block_params):
                        for j, p in enumerate(group['params']):
                            if p.grad is not None:
                                result += torch.norm(p.grad, p=2).item() ** 2
                    df_block_squared_norm.append(result)

            # and greedy chooses the block with maximal value
            self.current_block = numpy.argmax(df_block_squared_norm)
            if self.verbose:
                print("swap to", self.current_block,
                      "| grads:", df_block_squared_norm)

        self._set_mask()

    def __next_block(self, closure: Callable[[], float]) -> bool:
        """
        prepares auxiliary problem w.r.t. specific block 
        and stores solutions for every of them
        """
        point = dict()
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        point[(block, i, j)] = torch.clone(p)

        self.blockwise.append({
            # evaluates loss in last obtained point
            "f": closure(backward=False),
            "point": point
        })

        # returns to the starting point `y_k`
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        p.set_(torch.clone(self.y[(block, i, j)]))
        
        self.current_block += 1
        if self.current_block >= len(self.param_blocks):
            # process all the obtained solutions
            return False
        else:
            # swith to the next auxiliary problem
            self._set_mask()
            return True

    def __set_convex_combination(self, weights: numpy.ndarray, points: List[dict]) -> None:
        """
        sets params to the convex combination of `points` with coefficients from `weights`
        """
        with torch.no_grad():
            for block, block_params in enumerate(self.param_blocks):
                for i, group in enumerate(block_params):
                    for j, p in enumerate(group['params']):
                        p.set_(sum(
                            weights[index] * points[index][(block, i, j)] 
                            for index in range(len(points))
                        ))

    def __aggregate(self, closure: Callable[[], float]) -> None:
        """
        aggregates the solutions of auxiliary problems
        according to the chosen switching policy
        """
        if self.switching_strategy == SwitchingStrategy.MIN_SCALAR:
            # chooses solution with the best function value
            optimal_index = min(enumerate(self.blockwise), key=lambda block_state: block_state[1]["f"])[0]
            
            # sets params to the corresponding point
            with torch.no_grad():
                for block, block_params in enumerate(self.param_blocks):
                    for i, group in enumerate(block_params):
                        for j, p in enumerate(group['params']):
                            p.set_(torch.clone(self.blockwise[optimal_index]["point"][(block, i, j)]))

            if self.verbose:
                print("chosen", optimal_index,
                      "| fvals:", [state["f"] for state in self.blockwise])
        elif self.switching_strategy == SwitchingStrategy.BEST_CONVEX:
            points = [state["point"] for state in self.blockwise]

            # temporary point to return to after auxiliary optimization
            tmp = dict()
            with torch.no_grad():
                for block, block_params in enumerate(self.param_blocks):
                    for i, group in enumerate(block_params):
                        for j, p in enumerate(group['params']):
                            tmp[(block, i, j)] = torch.clone(p)

            # auxiliary function to optimize w.r.t. the convex coefficients `weights`
            # combining the obtained block-wise solutions and evaluating loss
            def convex_combination(weights: numpy.ndarray) -> float:
                weights = norm(weights)
                self.__set_convex_combination(weights, points)
                value = closure(backward=False)
                with torch.no_grad():
                    for block, block_params in enumerate(self.param_blocks):
                        for i, group in enumerate(block_params):
                            for j, p in enumerate(group['params']):
                                p.set_(torch.clone(tmp[(block, i, j)]))
                return value
            
            # finds optimal weights, projects them onto the unit simplex,
            # and sets params with them
            auxiliary = scipy.optimize.minimize(
                convex_combination, random_from_simplex(len(points)), 
                method='Nelder-Mead', tol=1e-3)
            alphas = norm(auxiliary.x)
            self.__set_convex_combination(alphas, points)

            if self.verbose:
                print("combination weights:", alphas, 
                      "| optimized", "succesfully" if auxiliary.success else "with error")
        
        # clears list of solutions
        self.blockwise = []
        # chooses the first block since cycle is already passed
        self.current_block = 0

    def step(self, closure: Callable[[], float] = None, blank: bool = False) -> None:
        """
        procedure to call at every iteration of the training cycle
        to perform alternating: combines all the neccesary updates
        and calculations and maintains event loop of auxiliary problems

        :param closure: function without arguments returning the loss,
        evaluated on all the data points; having boolean
        keyword-argument-flag `backward` to turn on/off backpropagation;
        necessarly zeroing gradients before all

        :param blank: flag to turn of the updates but continue counting iterations
        """
        super().step(closure=closure)

        if self.iteration % self.period == 0 and not blank:
            if self.switching_strategy in [
                SwitchingStrategy.SUBSEQUENT,
                SwitchingStrategy.RANDOM,
                SwitchingStrategy.MAX_GRAD
            ]:
                # update `v_k` after passed round,
                self.update(closure=closure)
                # and choose `y_k` and block to optimize on for the next one
                self.swap(closure=closure)

            elif self.switching_strategy in [
                SwitchingStrategy.MIN_SCALAR,
                SwitchingStrategy.BEST_CONVEX
            ]:
                # or iterate block after block until the end,
                if not self.__next_block(closure):
                    # then aggregate the solutions,
                    self.__aggregate(closure)
                    # update `v_k` after passed round,
                    self.update(closure=closure)
                    # and choose `y_k` for the next one
                    self.swap(closure=closure)
