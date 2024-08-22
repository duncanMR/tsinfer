import click
import datetime
import os
import pandas as pd
import tsinfer
import tszip
import stdpopsim
import msprime
from tqdm import tqdm


def simulate_stdpopsim(n, seed, seq_length):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_3G09")
    contig = species.get_contig(length=seq_length)
    samples = {"CEU": n}
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples, seed=seed)
    return ts

def simulate_msprime(n, seed, seq_length, recomb_rate, mut_rate):
    ts_raw = msprime.sim_ancestry(
        n,
        sequence_length=seq_length,
        random_seed=seed,
        recombination_rate=recomb_rate,
    )
    ts = msprime.sim_mutations(
        ts_raw, rate=mut_rate, random_seed=seed, model="binary"
    )
    return ts

def subset_1kgp(n, start=5e6, seq_length=1e6):
    assert n in [100, 300, 1500]
    ts = tszip.decompress(f'../gel-dating-paper/data/1kgp_bal{n}-chr20p-filterNton23-truncate-0-0-0-mm0-post-processed-simplified-SDN-dated-1.29e-08-100.trees.tsz')
    ts = ts.keep_intervals([[start, start+seq_length]])
    return ts

def compare_methods(
    pop_sizes,
    method,
    method_kwargs,
    output_dir="temp/logs",
    output_prefix=None,
    num_skipped=10,
    engines=["C", "NUMBA"],
    num_iterations=1,
    overwrite=False,
):
    if output_prefix is None:
        output_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    log_path = f"{output_dir}/{output_prefix}_log.tsv"
    
    if os.path.exists(log_path):
        if overwrite:
            print('Removing existing log file')
            os.remove(log_path)
        else:
            raise ValueError(f"Log file {log_path} already exists")
    
    assert engines[0] == "C"
    
    for pop_size in tqdm(pop_sizes):
        for iteration in range(num_iterations):
            ts = method(n=pop_size, **method_kwargs)
            print(f'Data with {pop_size} individuals has {ts.num_sites} sites and {ts.num_trees} trees (iteration {iteration})')
            sample_data = tsinfer.SampleData.from_tree_sequence(ts)

            for i, name in enumerate(engines):
                engine = getattr(tsinfer, name + "_ENGINE")
                ancestors = tsinfer.generate_ancestors(
                    sample_data,
                    engine=engine,
                    log_path=log_path,
                    num_skipped=num_skipped,
                    iteration=iteration,
                )
                if i == 0:
                    c_ancestors = ancestors
                if not c_ancestors.data_equal(ancestors):
                    raise ValueError(f"Ancestor arrays of {name} and C engine differ")                

@click.command()
@click.option('--pop-sizes', type=str, required=True, help='Comma-separated list of population sizes')
@click.option('--method', type=click.Choice(['simulate_stdpopsim', 'simulate_msprime', 'subset_1kgp']), required=True, help='Tree sequence generation method')
@click.option('--method-kwargs', type=str, help='Comma-separated key=value pairs for method kwargs')
@click.option('--output-dir', type=str, default='temp/logs', help='Output directory')
@click.option('--output-prefix', type=str, default=None, help='Filename prefix')
@click.option('--num-skipped', type=int, default=10, help='Number of skipped generations')
@click.option('--engines', type=str, default="C,NUMBA", help='Comma-separated list of engines')
@click.option('--num-iterations', type=int, default=1, help='Number of repetitions for each population size')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing log files')

def run_compare_methods(pop_sizes, method, method_kwargs, output_dir, output_prefix, num_skipped, engines, num_iterations, overwrite):
    """
    Run the compare_methods function from the command line with various tree sequence methods.
    """
    pop_sizes = [int(x) for x in pop_sizes.split(',')]
    engines = engines.split(',')
    kwargs_dict = {}
    if method_kwargs:
        kwargs_list = method_kwargs.split(',')
        for kwarg in kwargs_list:
            key, value = kwarg.split('=')
            try:
                value = eval(value)
            except:
                pass
            kwargs_dict[key] = value

    method_mapping = {
        'simulate_stdpopsim': simulate_stdpopsim,
        'simulate_msprime': simulate_msprime,
        'subset_1kgp': subset_1kgp
    }
    
    if method in method_mapping:
        method_func = method_mapping[method]
    else:
        raise ValueError(f"Method {method} not recognized. Available methods: {list(method_mapping.keys())}")

    compare_methods(
        pop_sizes=pop_sizes,
        method=method_func,
        method_kwargs=kwargs_dict,
        output_dir=output_dir,
        output_prefix=output_prefix,
        num_skipped=num_skipped,
        engines=engines,
        num_iterations=num_iterations,
        overwrite=overwrite,
    )
    
    click.echo(f"Ancestor data saved to {output_dir}/{output_prefix}_log.tsv")

if __name__ == '__main__':
    run_compare_methods()