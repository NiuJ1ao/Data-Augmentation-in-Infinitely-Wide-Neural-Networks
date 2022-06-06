from neural_tangents import stax

def ResNetBlock(channels, strides=(1,1), channel_mismatch=False):
    conv = stax.serial(
        stax.Relu(), stax.Conv(channels, (3,3), strides=strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3,3), padding='SAME'),
    )
    shortcut = stax.Identity() if not channel_mismatch else stax.Conv(channels, (3,3), strides, padding='SAME')
    
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(conv, shortcut),
        stax.FanInSum()
        )
    
def ResNetGroup(n, channels, strides=(1,1)):
    blocks = []
    blocks += [ResNetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [ResNetBlock(channels, (1,1))]
        
    return stax.serial(*blocks)