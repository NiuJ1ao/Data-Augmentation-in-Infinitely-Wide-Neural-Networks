from neural_tangents import stax
from util import jit_and_batch
from modules import ResNetGroup

class BaseModel():
    def __init__(self, batch_size=0, device_count=-1, store_on_device=True):
        self.params = None
        self.apply_fn, self.kernel_fn = jit_and_batch(self.apply_fn, self.kernel_fn, 
                                                      batch_size=batch_size,
                                                      device_count=device_count,
                                                      store_on_device=store_on_device)
        
    def init_params(self, key, shape):
        _, self.params = self.init_fn(key, shape)
        
    def update_params(self, params):
        self.params = params
        
    def predict(self, x):
        if self.params is None:
            raise Exception('Model parameters not initialized.')
        return self.apply_fn(self.params, x)

class FCN(BaseModel):
    def __init__(self, 
                 batch_size=0, 
                 device_count=-1, 
                 store_on_device=True, 
                 num_layers=2,
                 hid_dim=512, 
                 hid_w_std=1.5, 
                 hid_b_std=0.05, 
                 out_dim=1, 
                 out_w_std=1.5, 
                 out_b_std=0.05,
                 nonlinearity=stax.Relu):
        hid_layers = [stax.Dense(hid_dim, W_std=hid_w_std, b_std=hid_b_std), nonlinearity()] * num_layers
        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            *hid_layers,
            stax.Dense(out_dim, W_std=out_w_std, b_std=out_b_std)
        )
        super(FCN, self).__init__(batch_size, device_count, store_on_device)

class ResFCN(BaseModel):
    def __init__(self, batch_size, device_count, store_on_device):
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
        
        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            stax.Dense(512, W_std=1, b_std=0),
            ResBlock, ResBlock, stax.Erf(),
            stax.Dense(1, W_std=1.5, b_std=0)
        )
        super(ResFCN, self).__init__(batch_size, device_count, store_on_device)

class ResNet(BaseModel):
    def __init__(self, 
                 batch_size=0, 
                 device_count=-1, 
                 store_on_device=True, 
                 block_size=4, 
                 k=1, 
                 num_classes=10):
        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            stax.Conv(16, (3, 3), padding='SAME'),
            ResNetGroup(block_size, int(16 * k)),
            ResNetGroup(block_size, int(32 * k), (2, 2)),
            ResNetGroup(block_size, int(64 * k), (2, 2)),
            stax.AvgPool((7, 7)),
            stax.Flatten(),
            stax.Dense(num_classes, W_std=1., b_std=0.),
        )
        super(ResNet, self).__init__(batch_size, device_count, store_on_device)
        
