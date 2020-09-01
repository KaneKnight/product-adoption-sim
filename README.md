# product-adoption-sim

Simulation of the mean-field game proposed in chapter 3 of "Pricing and Referrals in Diffusion on Networks" seen here: https://arxiv.org/pdf/1509.06544.pdf

The file simulation.py contains the simulation which is construted as a python object.

There are 4 options for deciding the input graph for the simulation:
- Degree Distribution - A probabilty distribution deciding the likelihood that particular degree nodes will show up in the graph.  
- Degree Sequence - The list of degrees that will form the graph.
- Initial Graph - Networkx graph already constructed.
- Saved Json Graph - Previous Networkx graph that has been dumped to a json file.

The file utils.py contains mathematical functions for determining expected payoff of nodes within the network. Tested with doctests.

