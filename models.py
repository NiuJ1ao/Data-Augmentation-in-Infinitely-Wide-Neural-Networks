from neural_tangents import stax
from util import jit_and_batch, split_key
from modules import ResNetGroup, ConvBlock
from logger import get_logger
logger = get_logger()

class BaseModel():
    def __init__(self, batch_size=0, device_count=-1, store_on_device=True):
        self.params = None
        self.apply_fn, self.kernel_fn = jit_and_batch(self.apply_fn, self.kernel_fn, 
                                                      batch_size=batch_size,
                                                      device_count=device_count,
                                                      store_on_device=store_on_device)
        
    def init_params(self, shape, force_init=False):
        # initialise model if not yet initialised
        if self.params == None or force_init:
            _, net_key = split_key()
            _, self.params = self.init_fn(net_key, shape)
        
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
                 out_dim=1, 
                 W_std=1.5, 
                 b_std=0.05,
                 nonlinearity=stax.Relu):
        hid_layers = [stax.Dense(hid_dim, W_std=W_std, b_std=b_std), nonlinearity()] * num_layers
        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            *hid_layers,
            stax.Dense(out_dim, W_std=W_std, b_std=b_std)
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
                 W_std=1.5,
                 b_std=0.05,
                 block_size=2, 
                 num_classes=10):
        
        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            stax.Conv(out_chan=64, filter_shape=(7, 7), strides=(2, 2), padding='SAME', W_std=W_std, b_std=b_std),
            ResNetGroup(block_size, channels=64, W_std=W_std, b_std=b_std),
            ResNetGroup(block_size, channels=128, W_std=W_std, b_std=b_std, strides=(2, 2)),
            ResNetGroup(block_size, channels=256, W_std=W_std, b_std=b_std, strides=(2, 2)),
            ResNetGroup(block_size, channels=512, W_std=W_std, b_std=b_std, strides=(2, 2)),
            stax.AvgPool((7, 7)),
            stax.Flatten(),
            stax.Dense(num_classes, W_std=W_std, b_std=b_std),
        )
        super(ResNet, self).__init__(batch_size, device_count, store_on_device)
        
