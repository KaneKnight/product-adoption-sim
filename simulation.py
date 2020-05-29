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
        self.calculateBestResponse()
        self.calculateEdgePerspectiveDegreeDistribution()
        self.calculateInformationalAccessOfStrategy()
        self.assignInitialAdopters()
        self.simulateDiffusionOfProduct()

    def buildGraphFromDegreeInfo(self):
        if self.G != None:
            degreeSequence = sorted([d for n, d in self.G.degree()], reverse=True)
            degreeCount = dict(collections.Counter(degreeSequence))
            for d in degreeCount:
                degreeCount[d] = degreeCount[d] / self.n
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
        print(cartesianProd)
        meanFieldBestResponses = []
        for i in range(len(cartesianProd)):
            meanFieldBestResponse = {}
            for degree in range(1, largest + 1):
                meanFieldBestResponse[degree] = cartesianProd[i][degree - 1]
            meanFieldBestResponses.append(meanFieldBestResponse)    
        print(meanFieldBestResponses)        

    def calculateBestResponse(self):
        res = []
        for d in self.meanFieldStrategy:
            res.append(self.meanFieldStrategy[d])
        prod = list(itertools.product(res))
        print(prod)    

    def calculateInformationalAccessOfStrategy(self):
        self.infoAccess = 0
        for d in self.meanFieldStrategy:
            self.infoAccess += self.meanFieldStrategy[d] * self.edgePerspective[d]  
        print(self.infoAccess)         
    
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
                # If not already visted in another stage
                if not self.G.nodes[i]["adopted"]:
                    self.G.nodes[i]["adopted"] = True
                    self.G.nodes[i]["payoff"] += self.vH1 - self.p1
            if len(stage) != 0:    
                self.drawGraphState()     
            
    def drawGraphState(self):
        plt.figure(figsize=(40,40))
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
        early = [utils.payoffEarly(self.alpha, self.vH0, self.vH1, self.vL0, d, self.pHigh, self.p0, self.reward) for d in range(largest)]
        defer = [utils.payoffLate(self.alpha, self.vH1, d, self.pHigh, self.p1) for d in range(largest)]
        degree = [i for i in range(1, largest + 1)]
        plt.figure(figsize=(20,20))
        plt.plot(degree, early)
        plt.plot(degree, defer)
        plt.savefig("./payoffs.png")
            



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
        6 : 0.1,
    }

    lower, upper = 1, 20
    mu, sigma = 8, 5
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    
    N = 100
    rounded = [int(round(x)) for x in X.rvs(N)]
    largestDeg = max(rounded) 
    degreeDistribution = {i : 0 for i in range(1, largestDeg + 1)}
    for i in rounded:
        degreeDistribution[i] += 1
    degreeDistribution[30] = 8
    for k in degreeDistribution:
        degreeDistribution[k] = degreeDistribution[k] / (N + 8)    

    simulationParameters = {
                   # Product info
                   "vH1": 1.2,
                   "vH0": 0.3,
                   "vL1": 0,
                   "vL0": 0,
                   "quality": 1,

                   # Pricing policy
                   "p0": 1,
                   "p1": 0.2,
                   "reward": 0.05,
                    
                   # Agent's belief on the probabilty that their neigbours adopt in round 0
                   "alpha": 0.1,
                   # Agent's belief on the probabilty that the product qualtiy is high
                   "pHigh" : 0.72,

                   # Number of nodes to make up the graph.
                   "n": 5,
                   # Distribution of the degrees of the graph.
                   #"degreeDistribution" : degreeDistribution,
                   # Alternative to distribution
                   #"degreeSequence" : [10, 7, 3, 3, 2, 2, 1, 1, 1]
                   "initGraph" : G,
                   }        

    sim = Simulation(**simulationParameters)
    print(sim.__dict__)
    sim.simulate()  