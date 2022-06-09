from data_loader import load_mnist
from neural_tangents import stax
from nn import Trainer
from metrics import accuracy, mse_loss
from jax.example_libraries import optimizers
from util import args_parser
import logger as logging
from logger import get_logger
logger = get_logger()

from models import FCN, ResFCN, ResNet, TestDense

def main():
    logging.init_logger(log_level=logging.INFO)
    args = args_parser()
    
    # initialise model
    if args.model == 'resnet':
        model = ResNet(block_size=1, k=1, num_classes=10)
        flatten = False
    elif args.model == 'fcn':
        model = FCN(num_layers=2, hid_dim=1024, out_dim=10, nonlinearity=stax.Relu)
        flatten = True
    
    # load dataset
    train, val, test = load_mnist(flatten=flatten)
    
    # optimizer = optimizers.sgd(args.lr)
    optimizer = optimizers.momentum(args.lr, mass=args.momentum)
    loss = mse_loss(model)
    acc = accuracy(model)
    trainer = Trainer(model, args.epochs, args.batch_size, optimizer, loss)

    opt_params = trainer.fit(train, test, metric=acc)
    # logger.info('Train accuracy: {}'.format(train_accs))
    # logger.info('Val accuracy: {}'.format(val_accs))
    
    # opt_params, train_losses, val_losses = trainer.fit(train, val)
    # logger.info('Train accuracy: {}'.format(train_losses))
    # logger.info('Val accuracy: {}'.format(val_losses))
    
if __name__ == "__main__":
    main()