from data_loader import load_mnist
from nn import Trainer
from metrics import accuracy, mse_loss
from jax.example_libraries import optimizers
from util import args_parser
import logger
logger = logger.init_logger(log_level=logger.INFO)

from models import FCN, ResNet

def main():
    args = args_parser()
    
    logger.warn("Due to different parameterization strategy in neural tangents, you may need to try different learning rate to get the same results. (https://github.com/google/neural-tangents/issues/155)")
    
    # initialise model
    if args.model == 'resnet':
        model = ResNet(block_size=4, k=1, num_classes=10)
        flatten = False
    elif args.model == 'fcn':
        model = FCN(num_layers=2, hid_dim=1024, out_dim=10)
        flatten = True
    
    # load dataset
    train, val, test = load_mnist(flatten=flatten)
    
    # optimizer = optimizers.sgd(args.lr)
    optimizer = optimizers.momentum(args.lr, mass=args.momentum)
    loss = mse_loss(model)
    trainer = Trainer(model, args.epochs, args.batch_size, optimizer, loss)

    trainer.fit(train, val, metric=accuracy)
    
    predicts = model.predict(test[0])
    acc = accuracy(predicts, test[1])
    logger.info(f"Test accuracy: {acc}")
    
if __name__ == "__main__":
    main()