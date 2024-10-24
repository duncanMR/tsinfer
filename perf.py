import click
import datetime
import os
import pandas as pd
import tsinfer
import tszip
import stdpopsim
import msprime
import time as time_
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class TestMatchingPerformance:
    def __init__(self, sample_data, freq_list=[0.8, 1], num_threads=24):
        self.sample_data = sample_data
        self.freq_list = freq_list
        self.data_list = [] 
        self.dataframe = pd.DataFrame()
        self.inferred_ts = {}
        self.num_threads = num_threads

    def run_matching(self):
        inferred_anc = tsinfer.generate_ancestors(self.sample_data, progress_monitor=True)
        
        for i, freq in enumerate(self.freq_list):
            print(f'Inferring ARG with frequency cutoff {freq}')
            filtered_anc = inferred_anc.filter_old_ancestors(max_frequency=freq)

            before_wall = time_.perf_counter()
            before_cpu = time_.process_time()
            ancestor_ts = tsinfer.match_ancestors(
                self.sample_data, filtered_anc, progress_monitor=True, num_threads=self.num_threads)
            ancestor_wall_time = time_.perf_counter() - before_wall
            ancestor_cpu_time = time_.process_time() - before_cpu

            before_wall = time_.perf_counter()
            before_cpu = time_.process_time()
            inferred_ts = tsinfer.match_samples(
                self.sample_data, ancestor_ts, post_process=False, progress_monitor=True, num_threads=self.num_threads)
            sample_wall_time = time_.perf_counter() - before_wall
            sample_cpu_time = time_.process_time() - before_cpu
            inferred_ts = tsinfer.post_process(inferred_ts)
            self.inferred_ts[freq] = inferred_ts

            ancestor_grouping = tsinfer.match_ancestors(
                self.sample_data, filtered_anc, return_grouping=True)
            ancestors_per_epoch = np.zeros(len(ancestor_grouping) + 1)
            for index, ancestors in ancestor_grouping.items():
                ancestors_per_epoch[index] = len(ancestors)
            num_epochs = len(ancestors_per_epoch)

            data_row = {
                'frequency': freq,
                'ancestor_match_walltime': ancestor_wall_time,
                'sample_match_walltime': sample_wall_time,
                'ancestor_match_cputime': ancestor_cpu_time,
                'sample_match_cputime': sample_cpu_time,
                'ancestors_per_epoch': str(ancestors_per_epoch.tolist()),
                'num_epochs': num_epochs
            }
            self.data_list.append(data_row)

        self.dataframe = pd.DataFrame(self.data_list)

    def dump(self, output_folder, prefix):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        csv_path = os.path.join(output_folder, f"{prefix}_wall_times.csv")
        self.dataframe.to_csv(csv_path, index=False)

        for freq, ts in self.inferred_ts.items():
            formatted_freq = f"{freq:.1f}"
            ts_path = os.path.join(output_folder, f"{prefix}_{formatted_freq}.tsz")
            tszip.compress(ts, ts_path)
        print(f"Data saved to {output_folder} with prefix '{prefix}'")

    def load(self, output_folder, prefix):
        csv_path = os.path.join(output_folder, f"{prefix}_wall_times.csv")
        self.dataframe = pd.read_csv(csv_path)
        self.freq_list = self.dataframe['frequency'].tolist()

        self.inferred_ts = {}
        for freq in self.freq_list:
            formatted_freq = f"{freq:.1f}"
            print(f"Loading tree sequence for frequency {formatted_freq}")
            ts_path = os.path.join(output_folder, f"{prefix}_{formatted_freq}.tsz")
            self.inferred_ts[freq] = tszip.decompress(ts_path)

    def visualise(self):
        freqs = self.dataframe['frequency'].tolist()
        colors = plt.colormaps.get_cmap('tab10').colors[:len(freqs)]

        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 2])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(len(freqs)), self.dataframe['ancestor_match_walltime'],
                color=colors, width=0.35)
        ax1.set_xlabel('Frequency cutoff')
        ax1.set_ylabel('Wall time (seconds)')
        ax1.set_title('Ancestor matching wall time')
        ax1.set_xticks(range(len(freqs)))
        ax1.set_xticklabels(freqs)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(range(len(freqs)), self.dataframe['sample_match_walltime'],
                color=colors, width=0.35)
        ax2.set_xlabel('Frequency cutoff')
        ax2.set_ylabel('Wall time (seconds)')
        ax2.set_title('Sample matching wall time')
        ax2.set_xticks(range(len(freqs)))
        ax2.set_xticklabels(freqs)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(range(len(freqs)), self.dataframe['ancestor_match_cputime'],
                color=colors, width=0.35)
        ax3.set_xlabel('Frequency cutoff')
        ax3.set_ylabel('CPU time (seconds)')
        ax3.set_title('Ancestor matching CPU time')
        ax3.set_xticks(range(len(freqs)))
        ax3.set_xticklabels(freqs)

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.bar(range(len(freqs)), self.dataframe['sample_match_cputime'],
                color=colors, width=0.35)
        ax4.set_xlabel('Frequency cutoff')
        ax4.set_ylabel('CPU time (seconds)')
        ax4.set_title('Sample matching CPU time')
        ax4.set_xticks(range(len(freqs)))
        ax4.set_xticklabels(freqs)

        ax5 = fig.add_subplot(gs[1, :])
        for i, freq in enumerate(freqs):
            ancestors_per_epoch_str = self.dataframe.loc[
                self.dataframe['frequency'] == freq, 'ancestors_per_epoch'].iloc[0]
            ancestors_per_epoch = np.array(eval(ancestors_per_epoch_str))
            ax5.scatter(range(len(ancestors_per_epoch)), ancestors_per_epoch,
                        color=colors[i], label=f'{freq}', s=50)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Number of Ancestors')
        ax5.set_title('Ancestor Group Counts per Epoch')
        ax5.legend(title='Frequency cutoff', loc='upper left')

        plt.tight_layout()
        plt.show()

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