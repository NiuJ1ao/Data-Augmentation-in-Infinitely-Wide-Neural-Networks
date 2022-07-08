from tabnanny import check
from data_loader import load_mnist
from nngp import NNGP
from models import FCN, CNN
from neural_tangents import stax
from metrics import accuracy
from util import args_parser, check_divisibility
import logger
logger = logger.init_logger(log_level=logger.INFO)

def run():
    args = args_parser()
    
    batch_size = args.batch_size
    device_count = args.device_count
    
    if args.model == 'cnn':
        model = CNN(kernel_batch_size=batch_size, device_count=device_count, num_classes=10)
        flatten = False
    elif args.model == 'fcn':
        model = FCN(kernel_batch_size=batch_size, device_count=device_count, 
                    num_layers=2, hid_dim=1024, out_dim=10, nonlinearity=stax.Relu)
        flatten = True
    
    train, _, test = load_mnist(flatten=flatten, one_hot=True)
    check_divisibility(train[0], test[0], batch_size, device_count)
    
    nngp = NNGP(model)
    nngp.fit(*train)

    test_x, test_y = test
    
    nngp_mean, _ = nngp.inference(test_x)
    acc = accuracy(nngp_mean, test_y)
    logger.info(f"accuracy of NNGP: {acc}")
    
if __name__ == "__main__":
    run()
