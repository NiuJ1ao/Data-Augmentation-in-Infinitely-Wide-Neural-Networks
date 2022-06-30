import itertools
from neural_tangents import stax
import jax.numpy as np
from util import jit_and_batch, split_key
from modules import ResNetGroup
from logger import get_logger
logger = get_logger()

class BaseModel():
    def __init__(self, batch_size=0, device_count=-1, store_on_device=True):
        self.params = None
        self.apply_fn, self.kernel_fn = jit_and_batch(self.apply_fn, self.kernel_fn, 
                                                      batch_size=batch_size,
                                                      device_count=device_count,
                                                      store_on_device=store_on_device)
        
    def init_params(self, shape):
        # initialise model if not yet initialised
        if self.params == None:
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
        
    def pack_params(self):
        param_list = []
        self.index_list = []
        i = 0
        for param in self.params:
            if len(param) > 0: # parametric layers
                W, b = param # (in, out), (1, out)
                layer = np.concatenate((W, b), axis=0).flatten()
                num_params = layer.shape[0]
                param_list.append(layer)
                self.index_list.append((i, i + num_params))
                i += num_params

        packed_params = np.concatenate(param_list, axis=0)
        logger.debug(f"packed_params: {packed_params.shape}")
        logger.debug(self.index_list)
        return packed_params

    def unpack_params(self, params):
        param_list = []
        counter = itertools.count()
        for param in self.params:
            if len(param) == 0:
                param_list.append(()) # non-parametric layers have no parameters
            else:
                i = next(counter)
                W, b = param # (in, out), (1, out)
                input_dim, output_dim = W.shape
                layer_shape = (input_dim + 1, output_dim)
                start, end = self.index_list[i]
                layer_param = params[start:end].reshape(layer_shape)
                recon_W, recon_b = layer_param[:-output_dim], layer_param[output_dim:]
                recon_W, recon_b = recon_W.reshape(W.shape), recon_b.reshape(b.shape)
                
                assert W == recon_W and b == recon_b
                
        assert False 
        return param_list
    
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
        
