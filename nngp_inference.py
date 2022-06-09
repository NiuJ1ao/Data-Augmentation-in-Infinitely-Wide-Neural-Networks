from tabnanny import check
from data_loader import load_mnist
from nngp import NNGP
from models import FCN, ResNet
from neural_tangents import stax
from metrics import accuracy
from util import args_parser
import logger
logger = logger.init_logger(log_level=logger.INFO)

def check_divisibility(train, test, batch_size, device_count):
    '''check if the number of data is divisible by batch size and device count'''
    train_num = train.shape[0]
    test_num = test.shape[0]
    if batch_size > 0:
        assert train_num % (batch_size * device_count) == 0, "training data size is not divisible by (batch size x device count)"
        assert test_num % batch_size == 0, "test data size is not divisible by batch size"

def run():
    args = args_parser()
    
    batch_size = args.batch_size
    device_count = args.device_count
    
    if args.model == 'resnet':
        model = ResNet(batch_size=batch_size, device_count=device_count, 
                       block_size=1, k=1, num_classes=10)
        flatten = False
    elif args.model == 'fcn':
        model = FCN(batch_size=batch_size, device_count=device_count, 
                    num_layers=2, hid_dim=1024, out_dim=10, nonlinearity=stax.Relu)
        flatten = True
    
    train, _, test = load_mnist(flatten=flatten)
    check_divisibility(train[0], test[0], batch_size, device_count)
    
    nngp = NNGP(model)
    nngp.fit(*train)

    test_x, test_y = test
    
    nngp_mean, _ = nngp.inference(test_x)
    acc = accuracy(nngp_mean, test_y)
    logger.info(f"accuracy of NNGP: {acc}")
    
    ntk_mean, _ = nngp.inference(test_x, ntk=True)
    acc = accuracy(ntk_mean, test_y)
    logger.info(f"accuracy of NTK: {acc}")
    
if __name__ == "__main__":
    run()
