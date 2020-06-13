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
import json
from networkx.readwrite import json_graph

class Agent:
    def __init__(self, id):
        self.id = id
        self.products = set([1, 2])
        self.payoff = 0

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
       if isinstance(obj, set):
          return list(obj)
       return json.JSONEncoder.default(self, obj)        

class Simulation:
    def __init__(self, vH0, vH1, vL0, vL1, quality, p1, p0, reward, alpha, pHigh, n, degreeDistribution=None, degreeSequence=None, initGraph=None, readGraph=False):
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

        # Initial adopters for starting the simulation.
        self.initialAdopters = []

        # Distribution of the degrees of the graph.
        self.degreeDistribution = degreeDistribution
        # Actual degrees of nodes as and alternative to distribution
        self.degreeSequence = degreeSequence
        # Number of nodes to make up the graph.
        self.n = n

        # Define pair of strategies.
        self.meanFieldBestResponses = [None, None]
        self.meanFieldStrategies = [None, None]

        # Define pair of info access
        self.infoAccess = [0, 0]
       
        # Whether to read the graph from json.
        self.read = readGraph

    def simulate(self):
        self.buildGraphFromDegreeInfo()
        self.graphPayoffs()
        self.drawGraphState()
        self.calculateMeanFieldStrategiesForProduct(0)
        self.calculateMeanFieldStrategiesForProduct(1)
        self.calculateEdgePerspectiveDegreeDistribution()
        self.calculateInformationalAccessOfStrategyForProduct(0)
        self.calculateInformationalAccessOfStrategyForProduct(1)
        self.assignInitialAdopters()
        self.simulateDiffusionOfProduct()
        self.pdfOfGraph()
        self.saveGraph()

    def buildGraphFromDegreeInfo(self):

        if self.read:
            self.readGraph()

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
            self.G = nx.configuration_model(self.degreeSequence)
            self.degreeSequence = []
            for i in range(self.n):
                self.degreeSequence.append(len(self.G[i]))
                self.G.nodes[i]["id"] = i
                self.G.nodes[i]["products"] = set([0, 1])
                self.G.nodes[i]["payoff"] = 0
            degreeCount = dict(collections.Counter(self.degreeSequence))
            for d in range(1, self.n):
                degreeCount[d] = degreeCount[d] / self.n if d in degreeCount else 0
            self.degreeDistribution = degreeCount
            print(self.degreeSequence)
            print(len(self.degreeSequence))        

    def calculateEdgePerspectiveDegreeDistribution(self):
        denom = 0
        for d in self.degreeDistribution:
            denom += self.degreeDistribution[d] * d

        self.edgePerspective = {}
        for d in self.degreeDistribution:
            self.edgePerspective[d] = (self.degreeDistribution[d] * d) / denom    
        

    def calculateMeanFieldStrategiesForProduct(self, product):
        bestResponses = []
        largest = max([d for n, d in self.G.degree()])
        for degree in range(1, largest + 1):
            payoffDelta = utils.payoffDeltaEarlyLate(self.alpha[product], self.vH0[product], self.vH1[product], self.vL0[product], degree, self.pHigh[product], self.p0[product], self.p1[product], self.reward[product])
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
        self.meanFieldBestResponses[product] = meanFieldBestResponses
        self.meanFieldStrategies[product] = meanFieldBestResponses[0]   

    def calculateInformationalAccessOfStrategyForProduct(self, product):
        for d in self.meanFieldStrategies[product]:
            self.infoAccess[product] += self.meanFieldStrategies[product][d] * self.edgePerspective[d]    
        if math.isclose(self.infoAccess[product], self.alpha[product]):
            print("*****************************************************alpha", self.alpha)          
    
    def assignInitialAdopters(self):
        for i in range(self.n):
            degree = len(self.G[i])
            adoptFirstProduct = random.random() < self.meanFieldStrategies[0][degree]
            adoptSecondProduct = random.random() < self.meanFieldStrategies[1][degree]
            if adoptFirstProduct and adoptSecondProduct:
                # Figure out which has best expected payoff
                payoffFirst = utils.payoffDeltaEarlyLate(self.alpha[0], self.vH0[0], self.vH1[0], self.vL0[0], degree, self.pHigh[0], self.p0[0], self.p1[0], self.reward[0])
                payoffSecond = utils.payoffDeltaEarlyLate(self.alpha[1], self.vH0[1], self.vH1[1], self.vL0[1], degree, self.pHigh[1], self.p0[1], self.p1[1], self.reward[1])
                product = 0
                if payoffFirst > payoffSecond:
                    product = 0
                else:
                    product = 1
                self.G.nodes[i]["products"].remove(product ^ 1)
                value = self.vH0[product] if self.quality[product] == 1 else self.vL0[product]
                self.G.nodes[i]["payoff"] += value - self.p0[product]
                self.initialAdopters.append((i, product))
            elif adoptFirstProduct:
                self.G.nodes[i]["products"].remove(1)
                value = self.vH0[0] if self.quality[0] == 1 else self.vL0[0]
                self.G.nodes[i]["payoff"] += value - self.p0[0]
                self.initialAdopters.append((i, 0))
            elif adoptSecondProduct:
                self.G.nodes[i]["products"].remove(0)
                value = self.vH0[1] if self.quality[1] == 1 else self.vL0[1]
                self.G.nodes[i]["payoff"] += value - self.p0[1]
                self.initialAdopters.append((i, 1))
            
        self.drawGraphState()    

    def simulateDiffusionOfProduct(self):
        unvisited = []
        for i, product in self.initialAdopters:
            # If quality is high and payoff for neighbours is positive 
            if self.quality[product] == 1 and self.vH1[product] - self.p1[product] > 0:
                neighbours = self.G[i]
                self.G.nodes[i]["payoff"] += len(neighbours) * self.reward[product]
                self.G.nodes[i]["payoff"] += self.vH1[product]
                sublist = []
                for n in neighbours:
                    if (n, product) not in self.initialAdopters:
                        sublist.append((n, product))
                unvisited.append(sublist)
            else:
                self.G.nodes[i]["products"].add(product ^ 1)   
                self.drawGraphState()   
                  
        for stage in unvisited:
            actVisit = False
            for i, product in stage:
                # If not already visted in another stage
                if len(self.G.nodes[i]["products"]) == 2:
                    self.G.nodes[i]["products"].remove(product ^ 1)
                    self.G.nodes[i]["payoff"] += self.vH1[product] - self.p1[product]
                else:
                    isMore = (self.vH1[product] - self.p1[product]) > (self.vH1[product ^ 1] - self.p1[product ^ 1])
                    if isMore:
                        self.G.nodes[i]["products"].add(product)
                        self.G.nodes[i]["products"].discard(product ^ 1)
                        self.G.nodes[i]["payoff"] -= self.vH1[product ^ 1] - self.p1[product ^ 1]
                        self.G.nodes[i]["payoff"] += self.vH1[product] - self.p1[product]
                actVisit = True        

            if actVisit:    
                self.drawGraphState()
                actVisit = False
                pass     
            
    def drawGraphState(self):
        plt.figure(figsize=(20,20))
        # Get pos of nodes
        pos=nx.kamada_kawai_layout(self.G)

        # Find nodes which have adopted product
        adoptedFirst = [i for i in range(self.n) if (0 in self.G.nodes[i]["products"] and len(self.G.nodes[i]["products"]) == 1)]
        
        adoptedSecond = [i for i in range(self.n) if (1 in self.G.nodes[i]["products"] and len(self.G.nodes[i]["products"]) == 1)]
        
        notAdopted = [node for node in self.G.nodes if node not in adoptedFirst and node not in adoptedSecond]
        
        
        # Draw Adopted first nodes
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=adoptedFirst,
                        node_color='b',
                        node_size=1000,
                        alpha=0.8)
        # Draw Adopted second nodes
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=adoptedSecond,
                        node_color='g',
                        node_size=1000,
                        alpha=0.8)                
        # Draw non adotped nodes                   
        nx.draw_networkx_nodes(self.G, pos,
                        nodelist=notAdopted,
                        node_color='r',
                        node_size=1000,
                        alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        # Labels for nodes
        labels = {i:i for i in range(self.n)}             

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, labels, font_size=30)

        plt.savefig("./fig_tick{}.png".format(self.tick))
        self.tick += 1

    def graphPayoffs(self):
        largest = max([d for n, d in self.G.degree()])
        earlyFirst = [utils.payoffEarly(self.alpha[0], self.vH0[0], self.vH1[0], self.vL0[0], d, self.pHigh[0], self.p0[0], self.reward[0]) for d in range(1, largest + 1)]
        deferFirst = [utils.payoffLate(self.alpha[0], self.vH1[0], d, self.pHigh[0], self.p1[0]) for d in range(1, largest + 1)]
        earlySecond = [utils.payoffEarly(self.alpha[1], self.vH0[1], self.vH1[1], self.vL0[1], d, self.pHigh[1], self.p0[1], self.reward[1]) for d in range(1, largest + 1)]
        deferSecond = [utils.payoffLate(self.alpha[1], self.vH1[1], d, self.pHigh[1], self.p1[1]) for d in range(1, largest + 1)]
        degree = [i for i in range(1, largest + 1)]
        plt.figure(figsize=(20,20))
        plt.plot(degree, earlyFirst, label="Adopt Early First")
        plt.plot(degree, earlySecond, label="Adopt Early Second")
        plt.plot(degree, deferFirst, label="Defer Adoption First")
        plt.plot(degree, deferSecond, label="Defer Adoption Second")
        plt.legend(fontsize=30)
        plt.xticks(range(1, largest + 1, 5), fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("Degree", fontsize=40)
        plt.ylabel("Payoff", fontsize=40)
        plt.savefig("./payoffs_{}.png".format(round(self.alpha[0], 3)))

    def pdfOfGraph(self):
        plt.figure(figsize=(20,20))
        plt.bar(list(self.degreeDistribution.keys()), self.degreeDistribution.values())
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("Degree", fontsize=40)
        plt.ylabel("Probability", fontsize=40)
        plt.savefig("./strat.png")

    def saveGraph(self):
        data = json_graph.adjacency_data(self.G)
        with open('graph.json', 'w') as outfile:
            json.dump(data, outfile, cls=SetEncoder)

    def readGraph(self):
        with open('graph.json', 'r') as f:
            data = json.load(f)
            G = json_graph.adjacency_graph(data)
            for i in range(self.n):
                 G.nodes[i]["payoff"] = 0
                 G.nodes[i]["products"] = set([0, 1])
            self.G = G           




if __name__ == "__main__":

    random.seed(305324)

    #Set up graph
    nodes = []
    for i in range(10):
        nodes.append(Agent(i))
    G = nx.Graph()
    G.add_nodes_from([
        (0, {"id": 0, "products": set([0,1]), "payoff": 0}),
        (1, {"id": 1, "products": set([0,1]), "payoff": 0}),
        (2, {"id": 2, "products": set([0,1]), "payoff": 0}),
        (3, {"id": 3, "products": set([0,1]), "payoff": 0}),
        (4, {"id": 4, "products": set([0,1]), "payoff": 0}),
        (5, {"id": 5, "products": set([0,1]), "payoff": 0}),
        (6, {"id": 6, "products": set([0,1]), "payoff": 0}),
        (7, {"id": 7, "products": set([0,1]), "payoff": 0}),
        (8, {"id": 8, "products": set([0,1]), "payoff": 0}),
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
        (0, {"id": 0, "products": set([0,1]), "payoff": 0}),
        (1, {"id": 1, "products": set([0,1]), "payoff": 0}),
        (2, {"id": 2, "products": set([0,1]), "payoff": 0}),
        (3, {"id": 3, "products": set([0,1]), "payoff": 0}),
    ])
    F.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
    ])

    fig, ax = plt.subplots(1, 1)
    lambda_, N = 0.05, 60
    bolt = np.arange(stats.boltzmann.ppf(0, lambda_, N, 1), stats.boltzmann.ppf(1, lambda_, N, 1))
    probs = stats.boltzmann.pmf(bolt, lambda_, N, 1)
    ax.plot(bolt, probs, 'bo', ms=8, label='boltzmann pmf')
    degreeDist = {}
    temp = 0
    for i in range(len(bolt)):
        degreeDist[int(bolt[i])] = probs[i]
        temp += probs[i]

    simulationParameters = {
                   # Product info
                   "vH1": (2, 1.5),
                   "vH0": (1, 1.5),
                   "vL1": (0, 0),
                   "vL0": (0, 0),
                   "quality": (1, 1),

                   # Pricing policy
                   "p0": (1, 1),
                   "p1": (0.2, 0.2),
                   "reward": (0.2, 0),
                    
                   # Agent's belief on the probabilty that their neigbours adopt in round 0
                   "alpha": (0.8, 0.05),
                   # Agent's belief on the probabilty that the product qualtiy is high
                   "pHigh" : (0.4, 0.4),

                   # Number of nodes that make up the graph.
                   "n": 70,
                   # Distribution of the degrees of the graph.
                   "degreeDistribution" : degreeDist,
                   # Alternative to distribution
                   #"degreeSequence" : de,
                   #"initGraph" : G,
                   "readGraph" : False,
                   }

    
    
    #for i in range(10):
     #   simulationParameters["alpha"] += 0.1
    sim = Simulation(**simulationParameters)
    sim.simulate()  
    print(sim.meanFieldStrategies[0])
    print(sim.meanFieldStrategies[1])