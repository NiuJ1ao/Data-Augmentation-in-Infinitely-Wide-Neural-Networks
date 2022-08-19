from tabnanny import check
from data_loader import load_mnist
from nngp import NNGP
from models import FCN, CNNShallow
from neural_tangents import stax
from metrics import accuracy, RMSE
from util import args_parser
import logger
logger = logger.init_logger(log_level=logger.DEBUG)

def run():
    args = args_parser()
    
    batch_size = args.batch_size
    device_count = args.device_count
    
    if args.model == 'cnn':
        model = CNNShallow(kernel_batch_size=0, device_count=device_count, num_classes=10)
        flatten = False
    elif args.model == 'fcn':
        model = FCN(kernel_batch_size=0, device_count=device_count, 
                    num_layers=2, out_dim=10, nonlinearity=stax.Relu)
        flatten = True
    
    train, _, test = load_mnist(shuffle=False, flatten=flatten, one_hot=True, val_size=0)
    # check_divisibility(train[0], test[0], batch_size, device_count)
    
    nngp = NNGP(model, train=train)
    lml = nngp.log_marginal_likelihood()
    logger.info(f"Log marginal likelihood: {lml}")

    test_x, test_y = test
    
    nngp_mean, _ = nngp.predict_fn_(test_x)
    acc = accuracy(nngp_mean, test_y)
    logger.info(f"accuracy of NNGP: {acc}")
    rmse = RMSE(nngp_mean, test_y)
    logger.info(f"RMSE of NNGP: {rmse}")
    
if __name__ == "__main__":
    run()
