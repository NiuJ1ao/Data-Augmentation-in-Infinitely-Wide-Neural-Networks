from neural_tangents import stax
from util import jit_fns

class FCN(): 
    def __init__(self, 
                 num_layers=2,
                 hid_dim=512, 
                 hid_w_std=1.5, 
                 hid_b_std=0.05, 
                 out_dim=1, 
                 out_w_std=1.5, 
                 out_b_std=0.05,
                 nonlinearity=stax.Erf):
        
        super().__init__()
        hid_layers = [stax.Dense(hid_dim, W_std=hid_w_std, b_std=hid_b_std), nonlinearity()] * num_layers
        self.init_fn, apply_fn, kernel_fn = stax.serial(
            *hid_layers,
            stax.Dense(out_dim, W_std=out_w_std, b_std=out_b_std)
        )
        
        self.apply_fn, self.kernel_fn = jit_fns(apply_fn, kernel_fn)
        
        