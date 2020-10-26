from causaldag.structure_learning import gsp
from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_test, gauss_ci_suffstat
from causaldag import UndirectedGraph
import numpy as np
import itertools
from joblib import Parallel, delayed
from sklearn.utils import safe_mask
from sklearn.utils.random import sample_without_replacement
from typing import Dict, Optional, Any, List, Set, Union


def run_gsp(
    X, 
    alpha,
    nodes: set,
    depth: Optional[int] = 4,
    nruns: int = 5,
    verbose: bool = False,
    initial_undirected: Optional[Union[str, UndirectedGraph]] = 'threshold',
    initial_permutations: Optional[List] = None,
    fixed_orders=set(),
    fixed_adjacencies=set(),
    fixed_gaps=set(),
    use_lowest=True,
    max_iters=float('inf'),
    factor=2,
    progress_bar=False,
    summarize=False):
    # obtain sufficient statistics (causaldag.utils.ci_tests)
    obs_suffstat = gauss_ci_suffstat(X, invert=False)

    # define CI tester
    ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha)

    # run GSP
    est_dag = gsp(nodes=nodes, ci_tester=ci_tester, depth=depth, nruns=nruns, verbose=verbose, initial_undirected=initial_undirected, initial_permutations=initial_permutations,
        fixed_orders=fixed_orders, fixed_adjacencies=fixed_adjacencies, fixed_gaps=fixed_gaps, use_lowest=use_lowest, max_iters=max_iters, factor=factor, progress_bar=progress_bar,summarize=summarize)

    # convert dag to adjacency matrix, here specifying that the columns are "source" axis, so edge from j->i
    est_cpdag, _ = est_dag.cpdag().to_amat(source_axis=1)

    return est_cpdag


def gsp_stability_selection(
    X, 
    alpha_grid, 
    nodes: set,
    depth: Optional[int] = 4,
    nruns: int = 5,
    verbose: bool = False,
    initial_undirected: Optional[Union[str, UndirectedGraph]] = 'threshold',
    initial_permutations: Optional[List] = None,
    fixed_orders=set(),
    fixed_adjacencies=set(),
    fixed_gaps=set(),
    use_lowest=True,
    max_iters=float('inf'),
    factor=2,
    progress_bar=False,
    summarize=False,
    sample_fraction: float = 0.7,
    n_bootstrap_iterations: int = 100,
    bootstrap_threshold: float = 0.5,
    n_jobs: int = 1,
    random_state: int = None):
    """
    X: array, shape = [n_samples, n_features]
        Dataset. 
    alpha_grid: array-like
        Parameters to iterate over.
    nodes:
        Labels of nodes in the graph.
    depth:
        Maximum depth in depth-first search. Use None for infinite search depth.
    nruns:
        Number of runs of the algorithm. Each run starts at a random permutation and the sparsest DAG from all
        runs is returned.
    verbose:
        Verbosity of logging messages.
    initial_undirected:
        Option to find the starting permutation by using the minimum degree algorithm on an undirected graph that is
        Markov to the data. You can provide the undirected graph yourself, use the default 'threshold' to do simple
        thresholding on the partial correlation matrix, or select 'None' to start at a random permutation.
    initial_permutations:
        A list of initial permutations with which to start the algorithm. This option is helpful when there is
        background knowledge on orders. This option is mutually exclusive with initial_undirected.
    fixed_orders:
        Tuples (i, j) where i is known to come before j.
    fixed_adjacencies:
        Tuples (i, j) where i and j are known to be adjacent.
    fixed_gaps:
        Tuples (i, j) where i and j are known to be non-adjacent.
    sample_fraction: float, default = 0.7
        The fraction of samples to be used in each bootstrap sample.
        Should be between 0 and 1. If 1, all samples are used.
    n_bootstrap_iterations: int, default = 100
        Number of bootstrap samples to create.
    bootstrap_threshold: float, default = 0.5
        Threshold defining the minimum cutoff value for the stability scores. Edges with stability scores above
        the bootstrap_threshold are kept as part of the difference-DAG.
    n_jobs: int, default = 1
        Number of jobs to run in parallel.
    random_state: int, default = None
        Seed used by the random number generator.
    Returns
    -------
    adjacency_matrix: array, shape  = [n_features, n_features]
        Estimated CPDAG. Undirected edges are represented by assigning 1 in both directions, i.e. adjacency_matrix[i,j] = 1
        and adjacency_matrix[j,i] = 1. Otherwise for oriented edges, only adjacency_matrix[i,j] = 1 is assigned. 
        Assignment of 0 in the adjacency matrix represents no edge.
    stability_scores: array, shape = [n_params, n_features, n_features]
        Stability score of each edge for each parameter value in alpha_grid.
    References
    ----------
        [1] Meinshausen, N. and Buhlmann, P. (2010). Stability selection.
           Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), pp.417-473.
    """
    n_variables = len(nodes)
    n_params = len(alpha_grid)

    hyperparams = alpha_grid
    stability_scores = np.zeros((n_params, n_variables, n_variables))

    for idx, param in enumerate(hyperparams):
        print(param)
        if verbose > 0:
            print(
                "Fitting estimator for alpha = %.5f with %d bootstrap iterations" %
                (param, n_bootstrap_iterations))

        bootstrap_samples = bootstrap_generator(n_bootstrap_iterations, sample_fraction,
                                                 X)
        bootstrap_results = Parallel(n_jobs, verbose=verbose
                                     )(delayed(run_gsp)(X[safe_mask(X, subsample), :],
                                                    alpha=param,
                                                    nodes=nodes,
                                                    depth=depth,
                                                    nruns=nruns,
                                                    verbose=verbose,
                                                    initial_undirected=initial_undirected,
                                                    initial_permutations=initial_permutations,
                                                    fixed_orders=fixed_orders,
                                                    fixed_adjacencies=fixed_adjacencies,
                                                    fixed_gaps=fixed_gaps,
                                                    use_lowest=use_lowest,
                                                    max_iters=max_iters,
                                                    factor=factor,
                                                    progress_bar=progress_bar,
                                                    summarize=summarize)
                                       for subsample in bootstrap_samples)

        # calculate stability scores
        stability_scores[idx] = np.array(bootstrap_results).mean(axis=0)

    return stability_scores


def bootstrap_generator(n_bootstrap_iterations, sample_fraction, X, random_state=None):
    """Generates bootstrap samples from dataset."""
    n_samples = len(X)
    n_subsamples = np.floor(sample_fraction * n_samples).astype(int)
    subsamples = []
    for _ in range(n_bootstrap_iterations):
        subsample = sample_without_replacement(n_samples, n_subsamples, random_state=None)
        subsamples.append(subsample)
    return subsamples
