import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Agent:
    def __init__(self, id):
        self.id = id
        self.adopted = False

class Simulation:
    def __init__(self, initGraph, vH0, vH1, vL0, vL1, p1, p0, reward, alpha):
        self.G = initGraph
        self.tick = 0
        self.vH0 = vH0
        self.vH1 = vH1
        self.vL0 = vL0
        self.vL1 = vL1
        self.p1 = p1
        self.p0 = p0
        self.reward = reward
        self.alpha = alpha

    def simulate(self):
        pass

    def drawGraphState(self):
        # Get pos of nodes
        pos=nx.spectral_layout(self.G)

        # Find nodes which have adopted product
        adopted = [node for node in self.G.nodes() if node.adopted]
        notAdopted = [node for node in self.G.nodes() if node not in adopted]
        
        # Draw Adopted nodes
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=adopted,
                        node_color='g',
                        node_size=500,
                        alpha=0.8)
        # Draw non adotped nodes                   
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=notAdopted,
                        node_color='r',
                        node_size=500,
                        alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        # Labels for nodes
        labels = {node:node.id for node in self.G.nodes()}               

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, labels, font_size=16)

        plt.savefig("./fig_tick{}.png".format(self.tick))



if __name__ == "__main__":

    #Set up graph
    nodes = []
    for i in range(10):
        nodes.append(Agent(i))
    nodes[0].adopted = True    
    G = nx.Graph()
    G.add_edges_from([
        (nodes[0], nodes[1]),
        (nodes[0], nodes[2]),
        (nodes[0], nodes[3]),
        (nodes[1], nodes[4]),
    ])

    productValues = {"vH1": 1,
                     "vH0": 2,
                     "vL1": 0.2,
                     "vL0": 0.2,}
                     
    pricingPolicy = {"p0": 0.5,
                     "p1": 1,
                     "reward": 0.2,}

    # Agent's belief on the probabilty that their neigbours adopt in round 0
    agentAssumptions = {"alpha": 0.1,}

    sim = Simulation(G, **productValues, **pricingPolicy, **agentAssumptions)
    print(sim.__dict__)
    sim.drawGraphState()  