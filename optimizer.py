from jax.example_libraries.optimizers import Optimizer, optimizer
import jax.numpy as np
from copy import deepcopy
from logger import get_logger
logger = get_logger()

@optimizer
def sgd(step_size):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """

  def init(x0):
    return x0
  def update(i, g, x):
    logger.debug(g)
    return x - step_size * g
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)

@optimizer
def lbfgs(lr=1, 
          max_iter=20,
          max_eval=None,
          tolerance_grad=1e-7,
          tolerance_change=1e-9, 
          history_size=100):
    """Construct optimizer triple for LBFGS.
       This method is heavily inspired by the LBFGS algorithm in Pytorch.
    """
    state = dict(
            d = None,
            t = lr,
            ro = [],
            n_iter = 0,
            # layer_n_iter = 0,
            prev_flat_grad =  None,
            old_dirs = [],
            old_stps = [],
            H_diag = 1,
            history_size = history_size,
            max_eval = max_iter * 5 // 4 if max_eval is None else max_eval,
            success = False,
        )
    
    def init(x0):
        return x0
    
    def update(i, g, x):
        # flat grad: (1, h) -> (h,)
        flat_grad = g.flatten()
        # logger.debug(f"{g.shape} -> {flat_grad.shape}, {flat_grad.dtype}")

        opt_cond = np.max(np.abs(flat_grad)) <= tolerance_grad
        if opt_cond:
            # logger.debug("Optimization condition met")
            state["success"] = True
            return x
        
        d = state['d']
        t = state['t']
        old_dirs = state['old_dirs']
        old_stps = state['old_stps']
        ro = state['ro']
        H_diag = state['H_diag']
        prev_flat_grad = state['prev_flat_grad']
        
        n_iter = 0
        
        # if prev_flat_grad is not None and flat_grad.shape != prev_flat_grad.shape:
        #     prev_flat_grad = None
        #     state['layer_n_iter'] = 0
        
        while n_iter < max_iter:
            ############################################################
            # compute gradient descent direction
            ############################################################
            # keep track of num of iterations
            n_iter += 1
            state['n_iter'] += 1
            # state['layer_n_iter'] += 1
            
            if state['n_iter'] == 1:
                d = -flat_grad # (h,)
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad - prev_flat_grad # (h,)
                s = d * t # (h,)
                ys = y @ s # (h,)
                if ys > 1e-10:
                    # update memory
                    if len(old_dirs) == history_size:
                        del old_dirs[0]
                        del old_stps[0]
                        del ro[0]
                        
                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)
                    
                    # update scale of initial Hessian approximation
                    H_diag = ys / (y @ y)  # (y*y)
                    
                # compute the approximate inverse Hessian
                num_old = len(old_dirs)
                # logger.debug(f"num_old: {num_old}")
                
                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']
                
                # BUG
                q = -flat_grad
                for j in range(num_old - 1, -1, -1):
                    al[j] = old_stps[j] @ q * ro[j]
                    # logger.debug(f"q: {q}, al[i]: {al[j]}, old_dirs[i]: {old_dirs[j]}")
                    q += -al[j] * old_dirs[j]
                    # logger.debug(f"i: {j}; q: {q} {q.dtype}")
                    
                # multiply by initial Hessian
                # r/d is the final direction
                d = r = q * H_diag
                # logger.debug(f"d=r: {r}, q: {q}, {q.dtype}, H_diag: {H_diag}")
                for j in range(num_old):
                    be_i = (old_dirs[j] @ r) * ro[j]
                    r += (al[j] - be_i) * old_stps[j]
                    d = r
                    
            prev_flat_grad = deepcopy(flat_grad)
                    
            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / np.sum(np.abs(flat_grad))) * lr
            else:
                t = lr
                
            # directional derivative
            # logger.debug(f"d: {d}, flat_grad: {flat_grad}")
            gtd = flat_grad @ d  # g * d
            # logger.debug(f"gtd: {gtd}")
            
            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                # logger.debug("Directional derivative below tolerance")
                break
            
            # simply move with fixed-step
            # logger.debug(f"Step size: {t}; Direction: {d}")
            x += t * d.reshape(g.shape)
                
            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                # logger.debug("Max iterations reached")
                break

            # optimal condition
            if opt_cond:
                # logger.debug("Optimal condition reached")
                state["success"] = True
                break

            # lack of progress
            if np.max(np.abs(d * t)) <= tolerance_change:
                # logger.debug("Lack of progress")
                break
            
        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
                
        return x
    
    def get_params(x):
        return x
    
    return Optimizer(init, update, get_params)