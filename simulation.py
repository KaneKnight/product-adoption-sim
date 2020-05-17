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
    def __init__(self, initGraph, vH0, vH1, vL0, vL1, quality, p1, p0, reward, alpha, pHigh, degreeDistribution, n):
        # Graph topology info, intial adopters in round 0 assigned later. 
        self.G = initGraph
        self.F = initGraph
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

        # Distribution of the degrees of the graph.
        self.degreeDistribution = degreeDistribution
        # Number of nodes to make up the graph.
        self.n = n

    def simulate(self):
        self.buildGraphFromDegreeDistribution()
        self.drawGraphState()
        self.calculateMeanFieldStrategyForAgents()
        self.assignInitialAdopters()
        self.simulateDiffusionOfProduct()

    def buildGraphFromDegreeDistribution(self):
        degreeSequence = []
        temp = 0            
        for i in range(self.n - 1):
           d = utils.sampleDegreeFromDistribution(self.degreeDistribution)
           degreeSequence.append(d)
           temp += d

        d = utils.sampleDegreeFromDistribution(self.degreeDistribution)   
        while (temp + d) % 2 != 0:
            d = utils.sampleDegreeFromDistribution(self.degreeDistribution)
        
        degreeSequence.append(d)
        temp += d

        print("Degree Sequence:", degreeSequence)   
        self.G = nx.configuration_model(degreeSequence)
        degreeSequence = [d for n, d in self.G.degree()]
        print("Real Degree Sequence:", degreeSequence) 
        for i in range(self.n):
            self.G.nodes[i]["id"] = i
            self.G.nodes[i]["adopted"] = False
            self.G.nodes[i]["payoff"] = 0    
        

    def calculateMeanFieldStrategyForAgents(self):
        meanFieldStrategy = {}
        largest = max([d for n, d in self.G.degree()])
        for degree in range(1, largest + 1):
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
        for i in range(self.n):
            degree = len(self.G[i])
            adopted = random.random() < self.meanFieldStrategy[degree]
            self.G.nodes[i]["adopted"] = adopted
            if adopted:
                value = self.vH0 if self.quality == 1 else self.vL0
                self.G.nodes[i]["payoff"] += value - self.p0
                self.initialAdopters.append(i)
        self.drawGraphState()    

    def simulateDiffusionOfProduct(self):
        unvisited = []
        for i in self.initialAdopters:
            # If quality is high and payoff for neighbours is positive 
            if self.quality == 1 and self.vH1 - self.p1 > 0:
                neighbours = self.G[i]
                self.G.nodes[i]["payoff"] += len(neighbours) * self.reward
                sublist = []
                for n in neighbours:
                    if n not in self.initialAdopters:
                        sublist.append(n)
                unvisited.append(sublist)
            else:
                self.G.nodes[i]["adopted"] = False   
                self.drawGraphState()   
                  
        for stage in unvisited:
            for i in stage:
                self.G.nodes[i]["adopted"] = True
                self.G.nodes[i]["payoff"] += self.vH1 - self.p1
            if len(stage) != 0:    
                self.drawGraphState()     
            
    def drawGraphState(self):
        plt.figure(figsize=(20,20))
        # Get pos of nodes
        pos=nx.shell_layout(self.G)

        # Find nodes which have adopted product
        adopted = [i for i in range(self.n) if self.G.nodes[i]["adopted"]]
        notAdopted = [node for node in self.G.nodes if node not in adopted]
        
        # Draw Adopted nodes
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=adopted,
                        node_color='g',
                        node_size=4000,
                        alpha=0.8)
        # Draw non adotped nodes                   
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=notAdopted,
                        node_color='r',
                        node_size=4000,
                        alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        # Labels for nodes
        labels = {i:(i, self.G.nodes[i]["payoff"]) for i in range(self.n)}             

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, labels, font_size=30)

        plt.savefig("./fig_tick{}.png".format(self.tick))
        self.tick += 1



if __name__ == "__main__":

    #Set up graph
    nodes = []
    for i in range(10):
        nodes.append(Agent(i))
    G = nx.Graph()
    G.add_nodes_from([
        (0, {"id": 0, "adopted": False, "payoff": 0}),
        (1, {"id": 1, "adopted": False, "payoff": 0}),
        (2, {"id": 2, "adopted": False, "payoff": 0}),
        (3, {"id": 3, "adopted": False, "payoff": 0}),
        (4, {"id": 4, "adopted": False, "payoff": 0}),
    ])
    G.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
    ])

    degreeDistribution = {
        1 : 0.3,
        2 : 0.3,
        3 : 0.2,
        4 : 0.1,
        5 : 0.1,
    }

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
                   "pHigh" : 0.32,

                   # Distribution of the degrees of the graph.
                   "degreeDistribution" : degreeDistribution,
                   # Number of nodes to make up the graph.
                   "n": 15,
                   }        

    sim = Simulation(G, **simulationParameters)
    print(sim.__dict__)
    sim.simulate()  