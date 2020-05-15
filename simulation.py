import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import utils
import math

class Agent:
    def __init__(self, id):
        self.id = id
        self.adopted = False
        self.payoff = 0

class Simulation:
    def __init__(self, initGraph, vH0, vH1, vL0, vL1, quality, p1, p0, reward, alpha, pHigh):
        # Graph topology info, intial adopters in round 0 assigned later. 
        self.G = initGraph

        # Counter to show different stages of information diffusion.
        self.tick = 0

        # Value of product to agents in round 0 when quality high.
        self.vH0 = vH0
        # Value of product to agents in round 1 when quality high.
        self.vH1 = vH1
        # Value of product to agents in round 0 when quality low.
        self.vL0 = vL0
        # Value of product to agents in round 1 when quality low.
        self.vL1 = vL1
        # Quality of product
        self.quality = quality

        # Price of product in round 1.
        self.p1 = p1
        # Price of product in round 0.
        self.p0 = p0
        # Reward agents who adopt in round 0 receive if their neigbour adopts in round 1.
        self.reward = reward

        # Common belief of probabilty that neigbours adopt in round 0.
        self.alpha = alpha
        # Common belief of probabilty that product quality is high.
        self.pHigh = pHigh

        # Function that specifies, for every possible degree d, the probability that an agent of degree d adopts in round 0.
        self.meanFieldStrategy = {}

        # Initial adopters for starting the simulation.
        self.initialAdopters = []

    def simulate(self):
        self.drawGraphState()
        self.calculateMeanFieldStrategyForAgents()
        self.assignInitialAdopters()
        self.simulateDiffusionOfProduct()

    def calculateMeanFieldStrategyForAgents(self):
        meanFieldStrategy = {}
        for degree in range(10):
            payoffDelta = utils.payoffDeltaEarlyLate(self.alpha, self.vH0, self.vH1, self.vL0, degree, self.pHigh, self.p0, self.p1, self.reward)
            if math.isclose(payoffDelta, 0):
                # Node indifferent
                meanFieldStrategy[degree] = 0.5
            elif payoffDelta < 0:
                # Node defers
                meanFieldStrategy[degree] = 0
            elif payoffDelta > 0:
                # Node adopts in round 0     
                meanFieldStrategy[degree] = 1
        self.meanFieldStrategy = meanFieldStrategy
        print(meanFieldStrategy)        
    
    def assignInitialAdopters(self):
        for node in self.G.nodes():
            degree = len(self.G[node])
            adopted = random.random() < self.meanFieldStrategy[degree]
            node.adopted = adopted
            if adopted:
                value = self.vH0 if self.quality == 1 else self.vL0
                node.payoff += value - self.p0
                self.initialAdopters.append(node)
        self.drawGraphState()    

    def simulateDiffusionOfProduct(self):
        unvisited = []
        for node in self.initialAdopters:
            # If quality is high and payoff for neighbours is positive 
            if self.quality == 1 and self.vH1 - self.p1 > 0:
                neighbours = self.G[node]
                node.payoff += len(neighbours) * self.reward
                sublist = []
                for n in neighbours:
                    if n not in self.initialAdopters:
                        sublist.append(n)
                unvisited.append(sublist)
            else:
                node.adopted = False   
                self.drawGraphState()   
                  
        for stage in unvisited:
            for node in stage:
                node.adopted = True
                node.payoff += self.vH1 - self.p1
            self.drawGraphState()     
            
    def drawGraphState(self):
        plt.figure()
        # Get pos of nodes
        pos=nx.spectral_layout(self.G)

        # Find nodes which have adopted product
        adopted = [node for node in self.G.nodes() if node.adopted]
        notAdopted = [node for node in self.G.nodes() if node not in adopted]
        
        # Draw Adopted nodes
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=adopted,
                        node_color='g',
                        node_size=1200,
                        alpha=0.8)
        # Draw non adotped nodes                   
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=notAdopted,
                        node_color='r',
                        node_size=1200,
                        alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        # Labels for nodes
        labels = {node:(node.id, node.payoff) for node in self.G.nodes()}             

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10)

        plt.savefig("./fig_tick{}.png".format(self.tick))
        self.tick += 1



if __name__ == "__main__":

    #Set up graph
    nodes = []
    for i in range(10):
        nodes.append(Agent(i))
    G = nx.Graph()
    G.add_edges_from([
        (nodes[0], nodes[1]),
        (nodes[0], nodes[2]),
        (nodes[0], nodes[3]),
        (nodes[1], nodes[4]),
    ])

    simulationParameters = {
                   # Product info
                   "vH1": 1.25,
                   "vH0": 4,
                   "vL1": 0.2,
                   "vL0": 0.2,
                   "quality": 1,

                   # Pricing policy
                   "p0": 2,
                   "p1": 1,
                   "reward": 0.2,
                    
                   # Agent's belief on the probabilty that their neigbours adopt in round 0
                   "alpha": 0.1,
                   # Agent's belief on the probabilty that the product qualtiy is high
                   "pHigh" : 0.336,
                   }        

    sim = Simulation(G, **simulationParameters)
    print(sim.__dict__)
    sim.simulate()  