# sorted in order of increasing complexity
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None:
        x.requires_grad = True

    self.params = [x for x in params if x.requires_grad]

  # TODO: this probably shouldn't change the gradients, just the ones used by the optimizer
  def clipnorm(self, amount=1):
    for param in self.params:
      # clipnorm is the L2 norm, not value: is this right?
      param.grad.assign(param.grad.clip(-(amount**2), (amount**2)))

  def zero_grad(self):
    for param in self.params:
      param.grad = None

  def realize(self, extra=None):
    # TODO: corealize
    for p in self.params + extra if extra is not None else self.params:
      p.realize()

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr

  def step(self):
    for t in self.params:
      t.assign(t.detach() - t.grad * self.lr)
    self.realize()

class ASGD(SGD):
  def __init__(self, params, lr=0.01, lambd=1e-4, alpha=0.75, t0=1e6):
    super().__init__(params, lr)
    self.lr = lr
    self.lambd = lambd
    self.alpha = alpha
    self.t0 = t0
    self.stepcount = 0

  def step(self):
    self.stepcount += 1
    for t in self.params:
      eta = self.lr / (1 + self.lambd * self.lr * self.stepcount)**self.alpha
      mu = 1 / max(1, self.stepcount - self.t0)
      grad = t.detach() - t.grad * self.lr
      grad *= (1 - self.lambd * eta)
      t.assign(grad - eta)
      # if mu != 1:
      #   ax = (param - ax) * mu
      # else:
      #   ax = param
    self.realize()

class RMSprop(Optimizer):
  def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
    super().__init__(params)
    self.lr, self.decay, self.eps = lr, decay, eps

    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1.0 - self.decay) * (t.grad * t.grad)
      t.assign(t.detach() - (t.grad * self.lr).div(self.v[i].sqrt() + self.eps))
    self.realize(self.v)

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    self.m = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]
    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
      self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (t.grad * t.grad)
      t.assign(t.detach() - a * self.m[i].div(self.v[i].sqrt() + self.eps))
    self.realize(self.m + self.v)

class Adamax(Adam):
  def __init__(self, params, lr=0.002, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params, lr, b1, b2, eps)
    self.infs = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
      self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
      self.infs[i] = (self.b2 * self.infs[i]) + (t.grad.abs() + self.eps)
      t.assign(t.detach() - a * self.m[i].div(self.infs[i]))
      #self.v[i] = amax(norm_buf)
    self.realize(self.m + self.infs)

class Adan(Adam):
  def __init__(self, params, lr=0.002, b1=0.98, b2=0.92, b3=0.99, eps=1e-8):
    super().__init__(params, lr, b1, b2, eps)
    self.b3 = b3
    self.exp_avg_diff = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]
    self.pre_grads = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr / (1.0 - self.b1**self.t)
    bias_correction1 = 1.0 - self.b1 ** self.t
    bias_correction2 = 1.0 - self.b2 ** self.t
    bias_correction3 = 1.0 - self.b3 ** self.t

    for i, t in enumerate(self.params):
      #t.grad = t.grad * clip_global_grad_norm
      diff = t.grad - self.pre_grads[i]
      update = t.grad + self.b2 * diff

        # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
      self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
      self.exp_avg_diff[i] = self.b2 * self.exp_avg_diff[i] + (1.0 - self.b2) * (diff * diff)
      self.v[i] = self.b3 * self.v[i] + (1.0 - self.b3) * (update * update)

      denom = self.v[i].sqrt() / bias_correction3**0.5 + self.eps
      update = ((self.m[i] / bias_correction1 + self.b2 * self.exp_avg_diff[i] / bias_correction2)) / denom
      self.pre_grads[i] = t.grad
      # t.assign(t.detach() * (1 - a) - a * update)
      t.assign((t.detach() - a * update) / (1 + a))
    self.realize(self.m + self.v)

def get_parameters(obj):
  parameters = []
  if isinstance(obj, Tensor):
    parameters.append(obj)
  elif isinstance(obj, list) or isinstance(obj, tuple):
    for x in obj:
      parameters.extend(get_parameters(x))
  elif hasattr(obj, '__dict__'):
    for v in obj.__dict__.values():
      parameters.extend(get_parameters(v))
  return parameters
