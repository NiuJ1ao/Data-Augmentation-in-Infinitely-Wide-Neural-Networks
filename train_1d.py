from data_loader import synthetic_dataset2
from nn import Trainer
from jax.example_libraries import optimizers
from metrics import RMSE, mse_loss
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from models import FCN
from neural_tangents import stax
import logger as logging
logger = logging.init_logger(log_level=logging.INFO)

train, test = synthetic_dataset2()

fcn = FCN(
    kernel_batch_size=0, 
    device_count=-1, 
    num_layers=20,
    hid_dim=512, 
    nonlinearity=stax.Relu
)

optimizer = optimizers.momentum(0.1, mass=0.9)
loss = mse_loss(fcn)
trainer = Trainer(model=fcn, epochs=100, batch_size=100, optimizer=optimizer, loss=loss)

_, train_losses, test_losses = trainer.fit(train, test)

plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.savefig("figures/1d_losses")
plt.close()

f_test = fcn.predict(test[0])
rmse = RMSE(f_test, test[1])
logger.info(f"Test loss: {rmse:.4f}")

plt.plot(test[0], test[1], label='Latent function', lw=0.5)
plt.plot(test[0], f_test, label='Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-3.0, 3.0)
plt.legend()
plt.savefig("figures/1d_NN.pdf", bbox_inches='tight')
plt.close()