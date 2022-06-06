from neural_tangents import stax
from util import jit_fns
from modules import ResNetGroup

class BaseModel():
    def __init__(self):
        self.params = None

class FCN(BaseModel):
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

class ResFCN(BaseModel):
    def __init__(self):
        super().__init__()
        ResBlock = stax.serial(
            stax.FanOut(2),
            stax.parallel(
                stax.serial(
                    stax.Erf(),
                    stax.Dense(512, W_std=1.1, b_std=0),
                ),
                stax.Identity()
            ),
            stax.FanInSum()
        )
        
        self.init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(512, W_std=1, b_std=0),
            ResBlock, ResBlock, stax.Erf(),
            stax.Dense(1, W_std=1.5, b_std=0)
        )

        self.apply_fn, self.kernel_fn = jit_fns(apply_fn, kernel_fn)

class ResNet(BaseModel):
    def __init__(self, block_size, k, num_classes):
        super().__init__()
        self.init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Conv(16, (3, 3), padding='SAME'),
            ResNetGroup(block_size, int(16 * k)),
            ResNetGroup(block_size, int(32 * k), (2, 2)),
            ResNetGroup(block_size, int(64 * k), (2, 2)),
            stax.AvgPool((8, 8)),
            stax.Flatten(),
            stax.Dense(num_classes, W_std=1., b_std=0.),
        )
        self.apply_fn, self.kernel_fn = jit_fns(apply_fn, kernel_fn)
        
