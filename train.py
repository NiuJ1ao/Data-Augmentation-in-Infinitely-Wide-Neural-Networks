from data_loader import load_mnist
from nn import Trainer
from metrics import accuray, mse_loss, nll
from jax.example_libraries import optimizers
from util import args_parser
import logger as logging
from logger import logger
from models import FCN, ResFCN, ResNet

def main():
    logging.init_logger()
    args = args_parser()
    
    # initialise model
    if args.model == 'resnet':
        model = ResNet(block_size=4, k=1, num_classes=10)
    elif args.model == 'resfcn':
        model = ResFCN()
    
    # load dataset
    train, val, test = load_mnist()
    
    optimizer = optimizers.sgd(args.lr)
    loss = nll(model)
    acc = accuray(model)
    trainer = Trainer(model, args.epochs, args.batch_size, optimizer, loss)

    # opt_params, train_accs, val_accs = trainer.fit(train, val, metric=acc)
    # logger.info('Train accuracy: {}'.format(train_accs))
    # logger.info('Val accuracy: {}'.format(val_accs))
    
    opt_params, train_losses, val_losses = trainer.fit(train, val)
    logger.info('Train accuracy: {}'.format(train_losses))
    logger.info('Val accuracy: {}'.format(val_losses))
    
if __name__ == "__main__":
    main()