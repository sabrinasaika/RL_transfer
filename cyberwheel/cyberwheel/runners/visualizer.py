import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from cyberwheel.network.network_base import Network, Host, Subnet
from cyberwheel.red_agents.red_agent_base import AgentHistory, KnownHostInfo
from typing import Any
from importlib.resources import files
from cyberwheel.observation import RedObservation

def color_map(host_view) -> str:
        """
        Maps the state of the Host with a corresponding color.

        sweeped or scanned          -->     Green
        discovered                  -->     Yellow
        escalated                   -->     Orange
        impacted                    -->     Red
        """
        if host_view["impacted"]:
            return "red"
        elif host_view["escalated"]:
            return "orange"
        elif host_view["discovered"]:
            return "yellow"
        elif host_view["scanned"]:
            return "green"
        elif host_view["sweeped"]:
            return "green"
        else:
            return "gray"

class Visualizer:

    def __init__(self, network: Network, experiment_name: str,):
        self.experiment_dir = files("cyberwheel.data.graphs").joinpath(experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        self.host_mapping = {}
        for h in list(network.graph.nodes):
            self.host_mapping[h] = {"color": "gray"}
        self.draw_graph(network.graph)


        #node_color = []
        #node_edgecolor = []
        #node_labels = []
        #for h in list(self.G.nodes):
        #    if isinstance(G.nodes[h]["data"], Subnet):

        ##self.node_collection = nx.draw_networkx_nodes(self.G, pos=self.pos, node_color=["gray" for _ in network.hosts], ax=self.ax)
        #self.edge_collection = nx.draw_networkx_edges(self.G, pos=self.pos, ax=self.ax)
        #self.label_artists = nx.draw_networkx_labels(self.G, pos=self.pos, ax=self.ax)

    def draw_graph(self, G):
        self.G = G
        self.pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        new_host_mapping = {}
        for h in list(G.nodes):
            if h not in self.host_mapping:
                new_host_mapping[h] = {"color": "blue"}
            else:
                new_host_mapping[h] = {"color": self.host_mapping[h]["color"]}
        self.host_mapping = new_host_mapping

    def visualize(
        self,
        episode: int,
        step: int,
        info: dict[str, Any],
    ):
        """
        A function to visualize the state of the network at a given episode/step.
        Given the state of the environment at this episode and step, generates a
        visualization as a graph object and saves it in `graphs/{experiment_name}`

        * `network`: Network object representing the network at this step of the evaluation.
        * `episode`: integer representing the episode of the evaluation.
        * `step`: integer representing the step of the evaluation.
        * `experiment_name`: string representing the experiment name to save graphs under.
        * `history`: AgentHistory object representing the red agent history at this step of the evaluation.
        * `killchain`: List of KillChain Phases representing the killchain of the red agent.
        """

        #print(len(self.G.nodes))
        #print(len(self.pos.keys()))

        # Create `graphs/experiment_name` directory if it doesn't exist
        host_info = info["host_info"]
        network: Network = info["network"]

        source_host = info["source_host"]
        target_host = info["target_host"]
        step_commands = info["commands"]

        # Initialize network graph and environment state information
        G = network.graph

        # TODO: if G is different in num_hosts than self.G, reinitialize the graphviz stuff, and redraw the graph fully

        if len(G.nodes) != len(self.pos):
            new_host = (set(G.nodes.keys()) - set(self.pos.keys())).pop()
            self.draw_graph(G)
            self.host_mapping[new_host] = {"color": "blue"}

        on_host = ""
        for h, host_view in host_info.items():
            color = color_map(host_view)
            on_host = h if host_view.on_host else source_host
            if h not in self.host_mapping:
                self.host_mapping[h] = {"color": "blue"}
            elif color != self.host_mapping[h]["color"]:
                #new_host_color[h] = color
                self.host_mapping[h]["color"] = color
        
        #self.node_collection.set_color(new_host_color)

        # Set design of nodes in graph based on state
        colors = []
        for node_name in list(G.nodes):
            color = "gray"
            state = "Safe"
            commands = []
            if (
                isinstance(G.nodes[node_name]["data"], Host)
                and G.nodes[node_name]["data"].name == target_host
            ):
                commands = step_commands

            edgecolor = "black"
            linewidth = 2
            if "subnet" in node_name:
                color = "gray"
                state = "Scanned" if color == "yellow" else "Safe"
            else:
                if node_name in self.host_mapping:
                    color = self.host_mapping[node_name]["color"]
                    if color == "green":
                        state = "PingSweep/PortScan"
                    elif color == "yellow":
                        state = "Discovery"
                    elif color == "orange":
                        state = "Privilege Escalation - Process level escalated to 'root'"
                    elif color == "red":
                        state = "Impact"
                    elif color == "blue":
                        state = "Decoy"
                    else:
                        color = "gray"
                        state = "Safe"
                else:
                    color = "gray"

            if node_name == on_host:
                edgecolor = "blue"
                linewidth = 4
                state += "<br>Red Agent Position<br>"

            if "commands" not in G.nodes[node_name]:
                G.nodes[node_name]["commands"] = []

                    
            G.nodes[node_name]["color"] = color
            G.nodes[node_name]["state"] = state
            G.nodes[node_name]["commands"].extend(commands)
            G.nodes[node_name]["outline_color"] = edgecolor
            G.nodes[node_name]["outline_width"] = linewidth

            

        #fig, axe = plt.subplots(figsize=window_size)

        # Use Graphviz for neat, hierarchical layout
        #pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        #print(list(self.pos.keys()))
        for node in list(G.nodes):
            G.nodes[node]["pos"] = (self.pos[node][0], self.pos[node][1])

        # Draw graph
        #nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=colors)
        #edges = nx.draw_networkx_edges(G, pos=pos)
        #labels = nx.draw_networkx_labels(G, pos=pos)

        # Save graph to experiment directory
        outpath = self.experiment_dir.joinpath(f"{episode}_{step}.pickle")
        with open(outpath, "wb") as f:
            pickle.dump(G, f)
        plt.close()