from neural_tangents import stax
# from jax.example_libraries import stax

def ResNetBlock(out_channels, W_std, b_std, strides=(1,1), channel_mismatch=False):
    conv = stax.serial(
        stax.Relu(), stax.Conv(out_channels, (3,3), strides, padding='SAME', W_std=W_std, b_std=b_std),
        stax.Relu(), stax.Conv(out_channels, (3,3), strides=(1, 1), padding='SAME', W_std=W_std, b_std=b_std),
    )
    shortcut = stax.Identity() if not channel_mismatch else stax.Conv(out_channels, (3,3), strides, padding='SAME', W_std=W_std, b_std=b_std)
    
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(conv, shortcut),
        stax.FanInSum()
        )
    
def ResNetGroup(n, out_channels, W_std, b_std, strides=(1,1)):
    blocks = []
    blocks += [ResNetBlock(out_channels, W_std=W_std, b_std=b_std, strides=strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [ResNetBlock(out_channels, W_std=W_std, b_std=b_std, strides=(1,1))]
        
    return stax.serial(*blocks)
