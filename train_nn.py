from data_loader import load_mnist
from nn import Trainer
from metrics import accuracy, mse_loss
from jax.example_libraries import optimizers
from util import args_parser
import matplotlib.pyplot as plt
import logger
logger = logger.init_logger(log_level=logger.DEBUG)

from models import FCN, ResNet

def main():
    args = args_parser()
    
    logger.warning(f"Due to different parameterization strategy in neural tangents, you may need to try different learning rate to get the same results. (https://github.com/google/neural-tangents/issues/155)")
    
    # initialise model
    if args.model == 'resnet':
        model = ResNet(num_classes=10)
        flatten = False
    elif args.model == 'fcn':
        model = FCN(num_layers=2, hid_dim=1024, out_dim=10)
        flatten = True
    
    # load dataset
    train, val, test = load_mnist(flatten=flatten, one_hot=True)
    
    # optimizer = optimizers.sgd(args.lr)
    optimizer = optimizers.momentum(args.lr, mass=args.momentum)
    loss = mse_loss(model)
    trainer = Trainer(model, args.epochs, args.batch_size, optimizer, loss)

    opt_params, train_losses, val_losses, train_accs, val_accs = trainer.fit(train, val, metric=accuracy)
    
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.savefig(f'figures/{args.model}_loss.png')
    plt.close()
    
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='val')
    plt.legend()
    plt.savefig(f'figures/{args.model}_accuracy.png')
    plt.close()
    
    predicts = model.predict(test[0])
    acc = accuracy(predicts, test[1])
    logger.info(f"Test accuracy: {acc:.4%}")
    
if __name__ == "__main__":
    main()