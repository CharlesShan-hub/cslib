from clib.model.gan import gan
import config

opts = gan.TestOptions().parse(config.opts['GAN'])
gan.generate(opts)