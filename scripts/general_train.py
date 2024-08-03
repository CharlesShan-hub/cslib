import click
import clib
import config

@click.command()
@click.option('--name','-n',default='LeNet', help='Name of algorithm.')
@click.option('--field','-f',default='classical', help='Field of algorithm.')
@click.option('--param', '-p', multiple=True, type=(str, str, str), help='Any parameter in the format --param key value')
def main(name,field,param):
    assert hasattr(clib.model, field)
    all_algorithms = getattr(clib.model, field)
    
    assert hasattr(all_algorithms, name)
    assert name in config.opts
    algorithm = getattr(all_algorithms, name)

    opts = config.opts[name]
    opts.update({k:eval(t)(v) for k,v,t in param})

    algorithm.train(opts)

if __name__ == '__main__':
    main()