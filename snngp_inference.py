from data_loader import load_mnist, synthetic_dataset
from snngp import SNNGP
from models import FCN, ResNet
from neural_tangents import stax
from metrics import accuracy
from inducing_points import random_select
from util import args_parser, check_divisibility
import logger as logging
logger = logging.init_logger(log_level=logging.DEBUG)

def run():
    args = args_parser()
    
    batch_size = args.batch_size
    device_count = args.device_count
    
    # flatten = True if args.model == "fcn" else False
    # train, _, test = load_mnist(flatten=flatten, one_hot=False)
    # train_x, _ = train
    # test_x, _ = test
    # input_shape = train_x.shape
    
    train, test = synthetic_dataset()
    train_x, train_y = train
    train = (train_x, train_y.flatten())
    test_x, test_y = test
    input_shape = train_x.shape
    
    inducing_points = random_select(train_x, args.num_inducing_points)
    logger.debug(f"inducing_points: {inducing_points.shape}")
    
    check_divisibility(train_x, test_x, batch_size, device_count)
    if args.model == 'resnet':
        model = ResNet(batch_size=batch_size, device_count=device_count, 
                       block_size=1, k=1, num_classes=10)
    elif args.model == 'fcn':
        # model = FCN(batch_size=batch_size, device_count=device_count, 
        #             num_layers=2, hid_dim=1024, out_dim=10, nonlinearity=stax.Relu)
        model = FCN
        model_params = dict(
            nonlinearity=stax.Erf
        )
    
    snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=1)
    # snngp.log_marginal_likelihood_bound([1.5, 0.05], **model_params)
    snngp.optimize()
    
if __name__ == "__main__":
    run()
