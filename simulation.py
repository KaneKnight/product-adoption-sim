import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import utils
import math
import scipy
import scipy.stats as stats
import collections
import itertools

class Agent:
    def __init__(self, id):
        self.id = id
        self.adopted = False
        self.payoff = 0

class Simulation:
    def __init__(self, vH0, vH1, vL0, vL1, quality, p1, p0, reward, alpha, pHigh, n, degreeDistribution=None, degreeSequence=None, initGraph=None):
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

        # Distribution of the degrees of the graph.
        self.degreeDistribution = degreeDistribution
        # Actual degrees of nodes as and alternative to distribution
        self.degreeSequence = degreeSequence
        # Number of nodes to make up the graph.
        self.n = n

    def simulate(self):
        self.buildGraphFromDegreeInfo()
        self.graphPayoffs()
        self.drawGraphState()
        self.calculateMeanFieldStrategies()
        self.calculateEdgePerspectiveDegreeDistribution()
        self.calculateInformationalAccessOfStrategy()
        self.assignInitialAdopters()
        self.simulateDiffusionOfProduct()

    def buildGraphFromDegreeInfo(self):
        if self.G != None:
            degreeSequence = sorted([d for n, d in self.G.degree()], reverse=True)
            degreeCount = dict(collections.Counter(degreeSequence))
            for d in range(1, max(degreeSequence) + 1):
                degreeCount[d] = degreeCount[d] / self.n if d in degreeCount else 0
            self.degreeDistribution = degreeCount
            return
        
        if self.degreeDistribution != None:
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
            self.degreeSequence = degreeSequence 
        if self.degreeSequence != None:
            degreeCount = dict(collections.Counter(self.degreeSequence))
            for d in range(1, max(self.degreeSequence) + 1):
                degreeCount[d] = degreeCount[d] / self.n if d in degreeCount else 0
            self.degreeDistribution = degreeCount  
            self.G = nx.configuration_model(self.degreeSequence)
            for i in range(self.n):
                self.G.nodes[i]["id"] = i
                self.G.nodes[i]["adopted"] = False
                self.G.nodes[i]["payoff"] = 0

    def calculateEdgePerspectiveDegreeDistribution(self):
        denom = 0
        for d in self.degreeDistribution:
            denom += self.degreeDistribution[d] * d

        self.edgePerspective = {}
        for d in self.degreeDistribution:
            self.edgePerspective[d] = (self.degreeDistribution[d] * d) / denom    
        

    def calculateMeanFieldStrategies(self):
        bestResponses = []
        largest = max([d for n, d in self.G.degree()])
        for degree in range(1, largest + 1):
            payoffDelta = utils.payoffDeltaEarlyLate(self.alpha, self.vH0, self.vH1, self.vL0, degree, self.pHigh, self.p0, self.p1, self.reward)
            if math.isclose(payoffDelta, 0):
                # Node indifferent, ie choose either
                bestResponses.append([0, 1])
            elif payoffDelta < 0:
                # Node defers
                bestResponses.append([0])
            elif payoffDelta > 0:
                # Node adopts in round 0     
                bestResponses.append([1])
        cartesianProd = prod = list(itertools.product(*bestResponses))
        meanFieldBestResponses = []
        for i in range(len(cartesianProd)):
            meanFieldBestResponse = {}
            for degree in range(1, largest + 1):
                meanFieldBestResponse[degree] = cartesianProd[i][degree - 1]
            meanFieldBestResponses.append(meanFieldBestResponse)    
        self.meanFieldBestResponses = meanFieldBestResponses
        self.meanFieldStrategy = self.meanFieldBestResponses[0]       

    def calculateInformationalAccessOfStrategy(self):
        self.infoAccess = 0
        for d in self.meanFieldStrategy:
            self.infoAccess += self.meanFieldStrategy[d] * self.edgePerspective[d]  
        print((self.infoAccess, self.alpha))    
        if math.isclose(self.infoAccess, self.alpha):
            print("*****************************************************alpha", self.alpha)          
    
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
                self.G.nodes[i]["payoff"] += self.vH1
                sublist = []
                for n in neighbours:
                    if n not in self.initialAdopters:
                        sublist.append(n)
                unvisited.append(sublist)
            else:
                self.G.nodes[i]["adopted"] = False   
                self.drawGraphState()   
                  
        for stage in unvisited:
            actVisit = False
            for i in stage:
                # If not already visted in another stage
                if not self.G.nodes[i]["adopted"]:
                    self.G.nodes[i]["adopted"] = True
                    self.G.nodes[i]["payoff"] += self.vH1 - self.p1
                    actVisit = True
            if actVisit:    
                self.drawGraphState()
                pass     
            
    def drawGraphState(self):
        plt.figure(figsize=(20,20))
        # Get pos of nodes
        pos=nx.kamada_kawai_layout(self.G)

        # Find nodes which have adopted product
        adopted = [i for i in range(self.n) if self.G.nodes[i]["adopted"]]
        notAdopted = [node for node in self.G.nodes if node not in adopted]
        
        # Draw Adopted nodes
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=adopted,
                        node_color='g',
                        node_size=2000,
                        alpha=0.8)
        # Draw non adotped nodes                   
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=notAdopted,
                        node_color='r',
                        node_size=2000,
                        alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        # Labels for nodes
        labels = {i:(i, round(self.G.nodes[i]["payoff"], 2)) for i in range(self.n)}             

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, labels, font_size=30)

        plt.savefig("./fig_tick{}.png".format(self.tick))
        self.tick += 1

    def graphPayoffs(self):
        largest = max([d for n, d in self.G.degree()])
        early = [utils.payoffEarly(self.alpha, self.vH0, self.vH1, self.vL0, d, self.pHigh, self.p0, self.reward) for d in range(1, largest + 1)]
        defer = [utils.payoffLate(self.alpha, self.vH1, d, self.pHigh, self.p1) for d in range(1, largest + 1)]
        degree = [i for i in range(1, largest + 1)]
        plt.figure(figsize=(20,20))
        plt.plot(degree, early, label="Adopt Early")
        plt.plot(degree, defer, label="Defer Adoption")
        plt.legend(fontsize=30)
        plt.xticks(range(1, largest + 1), fontsize=30)
        plt.xlabel("Degree", fontsize=40)
        plt.ylabel("Payoff", fontsize=40)
        plt.savefig("./payoffs_{}.png".format(round(self.alpha, 3)))
            



if __name__ == "__main__":

    random.seed(305324)

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
        (5, {"id": 5, "adopted": False, "payoff": 0}),
        (6, {"id": 6, "adopted": False, "payoff": 0}),
        (7, {"id": 7, "adopted": False, "payoff": 0}),
        (8, {"id": 8, "adopted": False, "payoff": 0}),
    ])
    G.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (2, 6),
        (1, 5),
        (0, 7),
        (0, 8),
    ])

    F = nx.Graph()
    F.add_nodes_from([
        (0, {"id": 0, "adopted": False, "payoff": 0}),
        (1, {"id": 1, "adopted": False, "payoff": 0}),
        (2, {"id": 2, "adopted": False, "payoff": 0}),
        (3, {"id": 3, "adopted": False, "payoff": 0}),
    ])
    F.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
    ])

    d = [40]
    a = [1] * 48
    b = [3] * 6
    d = d + a + b

    simulationParameters = {
                   # Product info
                   "vH1": 0.5,
                   "vH0": 1,
                   "vL1": 0,
                   "vL0": 0,
                   "quality": 1,

                   # Pricing policy
                   "p0": 1,
                   "p1": 0.2,
                   "reward": 1.5,
                    
                   # Agent's belief on the probabilty that their neigbours adopt in round 0
                   "alpha": 0.5,
                   # Agent's belief on the probabilty that the product qualtiy is high
                   "pHigh" : 0.3,

                   # Number of nodes that make up the graph.
                   "n": 4,
                   # Distribution of the degrees of the graph.
                   #"degreeDistribution" : degreeDistribution,
                   # Alternative to distribution
                   #"degreeSequence" : d,
                   "initGraph" : F,
                   }

    
    
    #for i in range(10):
        #simulationParameters["alpha"] += 0.1
    sim = Simulation(**simulationParameters)
    sim.simulate()  
    print(sim.__dict__)