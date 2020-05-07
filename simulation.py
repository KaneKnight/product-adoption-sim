import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Agent:
    def __init__(self, id):
        self.id = id
        self.adopted = False

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
    ])

    
    # Get pos of nodes
    pos=nx.spring_layout(G)

    # Find nodes which have adopted product
    adopted = [node for node in G.nodes() if node.adopted]
    notAdopted = [node for node in G.nodes() if node not in adopted]
    
    # Draw Adopted nodes
    nx.draw_networkx_nodes(G,pos,
                       nodelist=adopted,
                       node_color='g',
                       node_size=500,
                       alpha=0.8)
    # Draw non adotped nodes                   
    nx.draw_networkx_nodes(G,pos,
                       nodelist=notAdopted,
                       node_color='r',
                       node_size=500,
                       alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)

    # Labels for nodes
    labels = {node:node.id for node in G.nodes()}               

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=16)

    plt.savefig("./fig.png")  