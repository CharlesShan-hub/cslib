import click

@click.command()
@click.option('--pre_trained', type=str, required=True)
@click.option('--batch_size', type=int, default=8, show_default=True, required=False)
@click.option('--use_relu', type=bool, default=False, show_default=True)
@click.option('--use_max_pool', type=bool, default=False, show_default=True)
def test(pre_trained,batch_size,use_relu,use_max_pool):
    print(pre_trained)
    print(batch_size)
    print(use_relu)
    print(use_max_pool)

if __name__ == '__main__':
    test()