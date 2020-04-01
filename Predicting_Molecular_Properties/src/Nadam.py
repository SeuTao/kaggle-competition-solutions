import torch
from torch.optim import Optimizer
import math


class Nadam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 schedule_decay=0.004, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        amsgrad=amsgrad, schedule_decay=schedule_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Nadam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['m_schedule'] = 1
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                momentum_cache_t = beta1 * (
                        1. - 0.5 * math.pow(0.96, state['step'] * group['schedule_decay']))
                momentum_cache_t_1 = beta1 * (
                        1. - 0.5 * math.pow(0.96, (state['step'] + 1) * group['schedule_decay']))
                state['m_schedule'] = state['m_schedule'] * momentum_cache_t

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                m_t_prime = exp_avg / (1 - state['m_schedule'] * momentum_cache_t_1)

                g_prime = grad.div(1 - state['m_schedule'])
                m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    v_t_prime = max_exp_avg_sq / (1 - beta2 ** state['step'])
                else:
                    v_t_prime = exp_avg_sq / (1 - beta2 ** state['step'])

                denom = v_t_prime.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], m_t_bar, denom)

        return loss
