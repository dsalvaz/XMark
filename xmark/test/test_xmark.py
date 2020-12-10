import unittest
from xmark import XMark_benchmark
import networkx as nx

class XMarkTestCase(unittest.TestCase):

    def test_xmark_auto(self):

        N = 2000
        gamma = 3
        beta = 2
        m_cat = ["auto", "auto"]
        theta = 0.3
        mu = 0.5
        avg_k = 10
        min_com = 20

        g = XMark_benchmark(N, gamma, beta, mu,
                            labels=m_cat,
                            noise=theta,
                            average_degree=avg_k, min_community=min_com,
                            type_attr="categorical")

        set1 = nx.get_node_attributes(g, 'label_0')
        set2 = nx.get_node_attributes(g, 'label_1')

        coms = {frozenset(g.nodes[v]['community']) for v in g}
        coms = [list(c) for c in coms]

        self.assertEquals(len(set(set1.values())), len(coms))
        self.assertEquals(len(set(set2.values())), len(coms))

