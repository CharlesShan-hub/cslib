from clib.model.gan import gan
import config

opts = gan.TrainOptions().parse(config.opts['GAN'])
gan.train(opts)