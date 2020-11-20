import networkx as nx
import copy
import random
import numpy as np

__all__ = ["XMark_benchmark"]

# Accommodates for both SciPy and non-SciPy implementations..
try:
    from scipy.special import zeta as _zeta

    def zeta(x, q, tolerance):
        return _zeta(x, q)
except ImportError:
    def zeta(x, q, tolerance):
        """The Hurwitz zeta function, or the Riemann zeta function of two
        arguments.
        ``x`` must be greater than one and ``q`` must be positive.
        This function repeatedly computes subsequent partial sums until
        convergence, as decided by ``tolerance``.
        """
        z = 0
        z_prev = -float('inf')
        k = 0
        while abs(z - z_prev) > tolerance:
            z_prev = z
            z += 1 / ((k + q) ** x)
            k += 1
        return z

def _zipf_rv_below(gamma, xmin, threshold, seed):
    """Returns a random value chosen from the bounded Zipf distribution.
    Repeatedly draws values from the Zipf distribution until the
    threshold is met, then returns that value.
    """
    result = nx.utils.zipf_rv(gamma, xmin, seed)
    while result > threshold:
        result = nx.utils.zipf_rv(gamma, xmin, seed)
    return result

def _powerlaw_sequence(gamma, low, high, condition, length, max_iters, seed):
    """Returns a list of numbers obeying a constrained power law distribution.
    ``gamma`` and ``low`` are the parameters for the Zipf distribution.
    ``high`` is the maximum allowed value for values draw from the Zipf
    distribution. For more information, see :func:`_zipf_rv_below`.
    ``condition`` and ``length`` are Boolean-valued functions on
    lists. While generating the list, random values are drawn and
    appended to the list until ``length`` is satisfied by the created
    list. Once ``condition`` is satisfied, the sequence generated in
    this way is returned.
    ``max_iters`` indicates the number of times to generate a list
    satisfying ``length``. If the number of iterations exceeds this
    value, :exc:`~networkx.exception.ExceededMaxIterations` is raised.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    for i in range(max_iters):
        seq = []
        while not length(seq):
            seq.append(_zipf_rv_below(gamma, low, high, seed))
        if condition(seq):
            return seq
    raise nx.ExceededMaxIterations("Could not create power law sequence")

# TODO Needs documentation.
def _generate_min_degree(gamma, average_degree, max_degree, tolerance,
                         max_iters):
    """Returns a minimum degree from the given average degree."""
    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations("Could not match average_degree")
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += (x ** (-gamma + 1)) / zeta(gamma, min_deg_mid,
                                                      tolerance)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    # return int(min_deg_mid + 0.5)
    return round(min_deg_mid)

def _assign_random_labels(seq, labels, lab_imb):
    """For Categorical attributes: assign the purest label of each community.
    Return a list of the purest labels."""
    tot_lab_seq = []
    for lb in labels:
        if lb == "auto":
            card_lab = [i for i in range(1, len(seq)+1)]
        else:
            card_lab = [i for i in range(1, lb+1)]
        lab_seq = []
        for i in range(len(seq)):
            if lb == "auto":
                lab_seq.append(i+1)
            else:
                lab_seq.append(random.randrange(1, lb+1))
            #maj = random.choice(card_lab)
            #for j in range(len(lab_seq)):
            #    if random.uniform(0, 1) < lab_imb:
            #        lab_seq[j] = maj # will be the purest label within the community
        tot_lab_seq.append(lab_seq)
    return tot_lab_seq

    """
    matrix_lab = np.zeros((len(seq), len(labels)))
    tmp_profile = {k: [] for k in range(len(labels))}

    for i, _ in enumerate(matrix_lab):
        for j, __ in enumerate(_):
            matrix_lab[i][j] = random.randrange(1, labels[j]+1)

    #TODO: user-defined
    todo = 4
    chosen_profiles = [random.choice(matrix_lab) for i in range(todo)]
    profile_seq = []

    for i in range(len(seq)):
        profile_com = random.choice(chosen_profiles)
        profile_com = [int(l) for l in profile_com]
        profile_seq.append(list(profile_com))

    for i, _ in enumerate(profile_seq):
        for j, l in enumerate(_):
            if j == list(tmp_profile.keys())[j]:
                tmp_profile[j].append(l)

    tot_profile_seq = list(tmp_profile.values())
    return tot_profile_seq
    """

def _assign_random_means(seq, labels, lab_imb, mu):
    """For Categorical attributes: assign the desired mean of each community.
    Return a list of the means."""
    tot_lab_seq = []
    for k, lb in enumerate(labels):
        if lb == "auto":
            dist = [i*10 for i in range(len(seq))]
            multimodal = [random.uniform(dist[i]-2, dist[i]+2) for i in range(len(seq))]
        else:
            dist = [i*10 for i in range(lb)]
            multimodal = [random.uniform(dist[i]-2, dist[i]+2) for i in range(lb)]
        lab_seq = []
        for i in range(len(seq)):
            if lb == "auto":
                lab_seq.append(multimodal[i])
            else:
                lab_seq.append(random.choice(multimodal)) # will be the mean of community
            # maj = random.choice(lab_seq)
            # for k in range(len(lab_seq)):
            #    if random.uniform(0, 1) < lab_imb:
            #        lab_seq[k] = maj
        tot_lab_seq.append(lab_seq)
    return tot_lab_seq

    """
    merge = list(zip(*tot_lab_seq))
    to_assign = [(el, seq[i]) for i, el in enumerate(merge)]
    to_assign.sort(key=itemgetter(0))
    to_assign_0 = [el[0] for el in to_assign]
    tmp = {k: [] for k in list(to_assign_0)}
    for el in to_assign:
        tmp[el[0]].append(el[1])

    prob_merge = []
    not_merge = []
    for m_c in list(tmp.values()):
        if len(m_c) > 1:
            l_c = []
            for c in m_c:
                if random.uniform(0, 1) < mu:
                    l_c.append(c)
                else:
                    not_merge.append(c)

            prob_merge.append(l_c)

    new_seq = [sum(c) for c in list(tmp.values())]
    #new_seq = [sum(c) for c in prob_merge] + not_merge
    return new_seq
    """

# TODO: Overlapping communities (not defined yet)
def _assign_node_memberships(degree_seq):
    overlap_seq = [2 for el in degree_seq]
    return overlap_seq

# TODO: Overlapping communities (not defined yet)
def _generate_overlapping_communities(degree_seq, community_sizes, overlap_seq, mu, max_iters):
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        for _ in range(overlap_seq[v]):
            c = random.choice(range(len(community_sizes)))
            # s = int(degree_seq[v] * (1 - mu) + 0.5)
            s = round(degree_seq[v] * (1 - mu))
            # If the community is large enough, add the node to the chosen
            # community. Otherwise, return it to the list of unaffiliated
            # nodes
            if s < community_sizes[c]:
                result[c].add(v)
            else:
                free.append(v)
            # If the community is too big, remove a node from it.
            if len(result[c]) > community_sizes[c]:
                free.append(result[c].pop())
            if not free:
                print(result)
            #return result

def _generate_communities(degree_seq, community_sizes, lab_coms, mu, labels, noise, std, max_iters, seed, type_attr):
    """Returns a list of sets, each of which represents a community.
    ``degree_seq`` is the degree sequence that must be met by the
    graph.
    ``community_sizes`` is the community size distribution that must be
    met by the generated list of sets.
    ``mu`` is a float in the interval [0, 1] indicating the fraction of
    intra-community edges incident to each node.
    ``max_iters`` is the number of times to try to add a node to a
    community. This must be greater than the length of
    ``degree_seq``, otherwise this function will always fail. If
    the number of iterations exceeds this value,
    :exc:`~networkx.exception.ExceededMaxIterations` is raised.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    The communities returned by this are sets of integers in the set {0,
    ..., *n* - 1}, where *n* is the length of ``degree_seq``.
    """
    card_lab = []
    for lb in labels:
        if lb == "auto": # number of labels equal to the number of communities
            card_lab.append([i for i in range(1, len(community_sizes)+1)])
        else:
            card_lab.append([i for i in range(1, lb+1)])

    # This assumes the nodes in the graph will be natural numbers.
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    lab_nodes = [list(range(n)) for i in range(len(card_lab))]
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        c = random.choice(range(len(community_sizes)))
        # s = int(degree_seq[v] * (1 - mu) + 0.5)
        s = round(degree_seq[v] * (1 - mu))
        # If the community is large enough, add the node to the chosen
        # community. Otherwise, return it to the list of unaffiliated nodes.
        if s < community_sizes[c]:
            result[c].add(v)

            if type_attr == "continuous":
                for j, attr in enumerate(lab_coms):
                    lab_nodes[j][v] = float(np.random.normal(attr[c], std, 1))
                    #lab_nodes[j][v] = round(float(np.random.normal(attr[c], std, 1)))
            #if categorical
            else:
                for j, attr in enumerate(lab_coms):
                    if random.uniform(0, 1) < 1 - noise:
                        lab_nodes[j][v] = attr[c]
                    else:
                        l = random.choice(card_lab[j])
                        lab_nodes[j][v] = l
                        # copy_card_lab = copy.copy(card_lab)
                        # copy_card_lab.remove(lab_coms[c])
                        # l = random.choice(copy_card_lab)
        else:
            free.append(v)
        # If the community is too big, remove a node from it.
        if len(result[c]) > community_sizes[c]:
            free.append(result[c].pop())
        if not free:
            return lab_nodes, result
    msg = 'Could not assign communities'
    raise nx.ExceededMaxIterations(msg)

#TODO
"""
def _heterogeneous_mixing (communities, deg_nodes, mu, delta_mu):
    seq_mu_c = []
    mu_min = 0.025

    for c in communities:
        k_max = [max(deg_nodes[u]) for u in c]
        k_avg = np.mean([deg_nodes[u] for u in c])
        mu_max = (k_avg - k_max) / k_avg
        mu_c = np.random.uniform(max(mu_min, mu - delta_mu), min(mu_max, mu + delta_mu), 1)
        for el in mu_c:
            seq_mu_c.append(el)

    return seq_mu_c
"""

def XMark_benchmark(n, tau1, tau2, mu, delta_mu=0.1, labels=2, std=0.1, noise=0, lab_imb=0, average_degree=None,
                        min_degree=None, max_degree=None, min_community=None,
                        max_community=None, tol=1.0e-7, max_iters=500,
                        seed=None, type_attr = "categorical", overlap="no"):
    r"""Returns the LFR benchmark graph.
    This algorithm proceeds as follows:
    1) Find a degree sequence with a power law distribution, and minimum
       value ``min_degree``, which has approximate average degree
       ``average_degree``. This is accomplished by either
       a) specifying ``min_degree`` and not ``average_degree``,
       b) specifying ``average_degree`` and not ``min_degree``, in which
          case a suitable minimum degree will be found.
       ``max_degree`` can also be specified, otherwise it will be set to
       ``n``. Each node *u* will have `\mu \mathrm{deg}(u)` edges
       joining it to nodes in communities other than its own and `(1 -
       \mu) \mathrm{deg}(u)` edges joining it to nodes in its own
       community.
    2) Generate community sizes according to a power law distribution
       with exponent ``tau2``. If ``min_community`` and
       ``max_community`` are not specified they will be selected to be
       ``min_degree`` and ``max_degree``, respectively.  Community sizes
       are generated until the sum of their sizes equals ``n``.
    3) Each node will be randomly assigned a community with the
       condition that the community is large enough for the node's
       intra-community degree, `(1 - \mu) \mathrm{deg}(u)` as
       described in step 2. If a community grows too large, a random node
       will be selected for reassignment to a new community, until all
       nodes have been assigned a community.
    4) Each node *u* then adds `(1 - \mu) \mathrm{deg}(u)`
       intra-community edges and `\mu \mathrm{deg}(u)` inter-community
       edges.
    Parameters
    ----------
    n : int
        Number of nodes in the created graph.
    tau1 : float
        Power law exponent for the degree distribution of the created
        graph. This value must be strictly greater than one.
    tau2 : float
        Power law exponent for the community size distribution in the
        created graph. This value must be strictly greater than one.
    mu : float
        Fraction of intra-community edges incident to each node. This
        value must be in the interval [0, 1].
    average_degree : float
        Desired average degree of nodes in the created graph. This value
        must be in the interval [0, *n*]. Exactly one of this and
        ``min_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.
    min_degree : int
        Minimum degree of nodes in the created graph. This value must be
        in the interval [0, *n*]. Exactly one of this and
        ``average_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.
    max_degree : int
        Maximum degree of nodes in the created graph. If not specified,
        this is set to ``n``, the total number of nodes in the graph.
    min_community : int
        Minimum size of communities in the graph. If not specified, this
        is set to ``min_degree``.
    max_community : int
        Maximum size of communities in the graph. If not specified, this
        is set to ``n``, the total number of nodes in the graph.
    tol : float
        Tolerance when comparing floats, specifically when comparing
        average degree values.
    max_iters : int
        Maximum number of iterations to try to create the community sizes,
        degree distribution, and community affiliations.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    Returns
    -------
    G : NetworkX graph
        The LFR benchmark graph generated according to the specified
        parameters.
        Each node in the graph has a node attribute ``'community'`` that
        stores the community (that is, the set of nodes) that includes
        it.
    Raises
    ------
    NetworkXError
        If any of the parameters do not meet their upper and lower bounds:
        - ``tau1`` and ``tau2`` must be strictly greater than 1.
        - ``mu`` must be in [0, 1].
        - ``max_degree`` must be in {1, ..., *n*}.
        - ``min_community`` and ``max_community`` must be in {0, ...,
          *n*}.
        If not exactly one of ``average_degree`` and ``min_degree`` is
        specified.
        If ``min_degree`` is not specified and a suitable ``min_degree``
        cannot be found.
    ExceededMaxIterations
        If a valid degree sequence cannot be created within
        ``max_iters`` number of iterations.
        If a valid set of community sizes cannot be created within
        ``max_iters`` number of iterations.
        If a valid community assignment cannot be created within ``10 *
        n * max_iters`` number of iterations.
    Examples
    --------
    Basic usage::
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> n = 250
        >>> tau1 = 3
        >>> tau2 = 1.5
        >>> mu = 0.1
        >>> G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5,
        ...                         min_community=20, seed=10)
    Continuing the example above, you can get the communities from the
    node attributes of the graph::
        >>> communities = {frozenset(G.nodes[v]['community']) for v in G}
    Notes
    -----
    This algorithm differs slightly from the original way it was
    presented in [1].
    1) Rather than connecting the graph via a configuration model then
       rewiring to match the intra-community and inter-community
       degrees, we do this wiring explicitly at the end, which should be
       equivalent.
    2) The code posted on the author's website [2] calculates the random
       power law distributed variables and their average using
       continuous approximations, whereas we use the discrete
       distributions here as both degree and community size are
       discrete.
    Though the authors describe the algorithm as quite robust, testing
    during development indicates that a somewhat narrower parameter set
    is likely to successfully produce a graph. Some suggestions have
    been provided in the event of exceptions.
    References
    ----------
    .. [1] "Benchmark graphs for testing community detection algorithms",
           Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi,
           Phys. Rev. E 78, 046110 2008
    .. [2] http://santo.fortunato.googlepages.com/inthepress2
    """
    # Perform some basic parameter validation.
    if not tau1 > 1:
        raise nx.NetworkXError("tau1 must be greater than one")
    if not tau2 > 1:
        raise nx.NetworkXError("tau2 must be greater than one")
    if not 0 <= mu <= 1:
        raise nx.NetworkXError("mu must be in the interval [0, 1]")

    # Validate parameters for generating the degree sequence.
    if max_degree is None:
        max_degree = n
    elif not 0 < max_degree <= n:
        raise nx.NetworkXError("max_degree must be in the interval (0, n]")
    if not ((min_degree is None) ^ (average_degree is None)):
        raise nx.NetworkXError("Must assign exactly one of min_degree and"
                               " average_degree")
    if min_degree is None:
        min_degree = _generate_min_degree(tau1, average_degree, max_degree,
                                          tol, max_iters)

    # Generate a degree sequence with a power law distribution.
    low, high = min_degree, max_degree

    def condition(seq): return sum(seq) % 2 == 0

    def length(seq): return len(seq) >= n
    deg_seq = _powerlaw_sequence(tau1, low, high, condition,
                                 length, max_iters, seed)

    # Validate parameters for generating the community size sequence.
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)

    # Generate a community size sequence with a power law distribution.
    #
    # TODO The original code incremented the number of iterations each
    # time a new Zipf random value was drawn from the distribution. This
    # differed from the way the number of iterations was incremented in
    # `_powerlaw_degree_sequence`, so this code was changed to match
    # that one. As a result, this code is allowed many more chances to
    # generate a valid community size sequence.
    low, high = min_community, max_community

    def condition(seq): return sum(seq) == n

    def length(seq): return sum(seq) >= n
    comms = _powerlaw_sequence(tau2, low, high, condition,
                               length, max_iters, seed)

    #TODO: Overlapping communities
    overlap_seq = _assign_node_memberships(deg_seq)

    # TODO TODO TODO
    overlap_comms = copy.copy(comms)
    overlap_deg_seq = copy.copy(deg_seq)
    cycle = [overlap_comms, overlap_deg_seq]
    if sum(overlap_comms) < sum(overlap_seq):
        add_size =  abs(sum(overlap_comms) - sum(overlap_seq))
        for new in range(add_size):
            for cy in cycle:
                pick = random.choice(list(range(len(cy))))
            if cy[pick] < np.median(cy):
                cy[pick] += 1
            else:
                cy[random.choice(list(range(len(cy))))] += 1

    if type_attr == "continuous":
        #lab imb and mu not used
        lab_coms = _assign_random_means(comms, labels, lab_imb, mu)
        # old idea: merge communities when clear attributes and ambiguous structure (theoretically wrong?)
        #lab_coms, merge_comms = _assign_random_means(comms, labels, lab_imb, mu)
        #if mu > 0.6:
        #    comms = merge_comms
    else:
        #lab imb not used
        lab_coms = _assign_random_labels(comms, labels, lab_imb)

    # Generate the communities based on the given degree sequence and
    # community sizes.
    max_iters *= 10 * n

    lab_nodes, communities = _generate_communities(deg_seq, comms, lab_coms, mu, labels, noise, std, max_iters, seed, type_attr)

    # TODO
    #het_mu = _heterogeneous_mixing(communities, deg_seq, mu, delta_mu)

    # Finally, generate the benchmark graph based on the given
    # communities, joining nodes according to the intra- and
    # inter-community degrees.
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, c in enumerate(communities):

        for u in c:
            while G.degree(u) < round(deg_seq[u] * (1 - mu)):
            #while G.degree(u) < round(deg_seq[u] * (1 - het_mu[i])):
                v = random.choice(list(c))
                G.add_edge(u, v)

            while G.degree(u) < deg_seq[u]:
                v = random.choice(range(n))
                if v not in c:
                    G.add_edge(u, v)

            for j, lab in enumerate(lab_nodes):
                G.nodes[u]['label_' + str(j)] = lab[u]
            G.nodes[u]['community'] = c

    # TODO
    #overlap_communities = _generate_overlapping_communities(overlap_deg_seq, overlap_comms, overlap_seq, mu, max_iters)

    return G
