from data_loader import load_mnist
from snngp import SNNGP
from models import FCN, ResNet
from neural_tangents import stax
from metrics import accuracy
from inducing_points import random_select
from util import args_parser, check_divisibility
import logger
logger = logger.init_logger(log_level=logger.DEBUG)

def run():
    args = args_parser()
    
    batch_size = args.batch_size
    device_count = args.device_count
    
    flatten = True if args.model == "fcn" else False
    train, _, test = load_mnist(flatten=flatten, one_hot=False)
    train_x, _ = train
    test_x, _ = test
    input_shape = train_x.shape
    
    inducing_points = random_select(train_x, args.num_inducing_points)
    logger.debug(f"inducing_points: {inducing_points.shape}")
    
    check_divisibility(train_x, test_x, batch_size, device_count)
    if args.model == 'resnet':
        model = ResNet(batch_size=batch_size, device_count=device_count, 
                       block_size=1, k=1, num_classes=10)
    elif args.model == 'fcn':
        model = FCN(batch_size=batch_size, device_count=device_count, 
                    num_layers=2, hid_dim=1024, out_dim=10, nonlinearity=stax.Relu)
    
    model.init_params(input_shape)
    snngp = SNNGP(model, train, inducing_points, 10)
    snngp.log_marginal_likelihood_bound(model.params)
    snngp.optimize()
    
if __name__ == "__main__":
    run()
