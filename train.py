from data_loader import load_mnist
from nn import Trainer
from losses import mse_loss, nll
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
    
    training_steps = args.training_steps
    optimizer = optimizers.sgd(args.lr)
    loss = nll(model)
    trainer = Trainer(model, training_steps, 0, optimizer, loss)

    opt_params, train_losses, val_losses = trainer.fit(train, val)
    
if __name__ == "__main__":
    main()