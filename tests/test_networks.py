import networkx as nx
from app_main import build_graph

# minimum_driver_nodes is defined locally in the Network Control section
# We reimport it here to test it:
from app_main import minimum_driver_nodes

def test_build_graph_simple():
    G = build_graph(["A","B"], [("A","B")])
    assert isinstance(G, nx.DiGraph)
    assert ("A","B") in G.edges()

def test_minimum_driver_nodes_runs():
    G = build_graph(["A","B"], [("A","B")])
    drivers = minimum_driver_nodes(G)

    assert isinstance(drivers, list)
    assert all(isinstance(x, str) for x in drivers)