import click
import networkx as nx
from littleballoffur.exploration_sampling import ForestFireSampler, RandomWalkSampler


class GraphLoader:

    def __init__(self, path):
        self.graph = nx.MultiGraph()
        self.path_to_dataset = path
        self.node2idx = {}
        self.idx2node = {}
        self.node_count = 0
        self.relation_count = 0
        self.attribute_count = 0
        self.timestamp_count = 0        

    def load_graph(self):
        data_file = open(self.path_to_dataset, "r")
        visited_attr_nodes = []
        visited_tuples = []
        count_nodes = 0
        for fact in data_file:
            f = fact.split("\t")
            s = f[0]
            r = f[1]
            o = f[2]
            t = f[3]
            if (s, r, o, t) not in visited_tuples:
                visited_tuples.append((s, r, o, t))    
                if o.startswith("Q"):
                    if s in self.node2idx.keys():
                        s_idx = self.node2idx[s]
                    else:
                        s_idx = count_nodes
                        count_nodes += 1
                        self.node2idx.update({s: s_idx})
                        self.idx2node.update({s_idx: s})
                    if o in self.node2idx.keys():
                        o_idx = self.node2idx[o]
                    else:
                        o_idx = count_nodes
                        count_nodes += 1
                        self.node2idx.update({o: o_idx})
                        self.idx2node.update({o_idx: o})
                    self.graph.add_edge(s_idx, o_idx, label=r, time=t)
                else:
                    if s in self.node2idx.keys():
                        s_idx = self.node2idx[s]
                    else:
                        s_idx = count_nodes
                        count_nodes += 1
                        self.node2idx.update({s: s_idx})
                        self.idx2node.update({s_idx: s})
                    if s in visited_attr_nodes:
                        attributes = self.graph.nodes[s_idx]["attr"]
                        attributes.append({r: o, "time": t})
                    else:
                        attributes = [{r: o, "time": t}]
                        visited_attr_nodes.append(s)
                    self.graph.add_node(node_for_adding=s_idx, attr=attributes)
        data_file.close()
        self.node_count = count_nodes-1


def forest_fire_sampling(train_path, test_path, valid_path, new_train_path, new_test_path, new_valid_path):
    train_graph = GraphLoader(train_path)
    test_graph = GraphLoader(test_path)
    valid_graph = GraphLoader(valid_path)

    train_graph.load_graph()
    test_graph.load_graph()
    valid_graph.load_graph()

    train_sampler = ForestFireSampler(number_of_nodes=train_graph.node_count*2/5, p=0.8)
    test_sampler = ForestFireSampler(number_of_nodes=test_graph.node_count*2/5, p=0.8)
    valid_sampler = ForestFireSampler(number_of_nodes=valid_graph.node_count*2/5, p=0.8)

    new_train_graph = train_sampler.sample(train_graph.graph)
    new_test_graph = test_sampler.sample(test_graph.graph)
    new_valid_graph = valid_sampler.sample(valid_graph.graph)

    write_sampled_data(new_train_path, new_train_graph, train_graph, new_test_path, new_test_graph, test_graph, new_valid_path, new_valid_graph, valid_graph)


def random_walk_sampling(train_path, test_path, valid_path, new_train_path, new_test_path, new_valid_path):
    train_graph = GraphLoader(train_path)
    test_graph = GraphLoader(test_path)
    valid_graph = GraphLoader(valid_path)

    train_graph.load_graph()
    test_graph.load_graph()
    valid_graph.load_graph()

    train_sampler = RandomWalkSampler(number_of_nodes=train_graph.node_count/2)
    test_sampler = RandomWalkSampler(number_of_nodes=test_graph.node_count/2)
    valid_sampler = RandomWalkSampler(number_of_nodes=valid_graph.node_count/2)

    new_train_graph = train_sampler.sample(train_graph.graph)
    new_test_graph = test_sampler.sample(test_graph.graph)
    new_valid_graph = valid_sampler.sample(valid_graph.graph)

    write_sampled_data(new_train_path, new_train_graph, train_graph, new_test_path, new_test_graph, test_graph, new_valid_path, new_valid_graph, valid_graph)


def write_sampled_data(new_train_path, new_train_graph, train_graph, new_test_path, new_test_graph, test_graph, new_valid_path, new_valid_graph, valid_graph):
    with open(new_train_path, "a") as new_train_file:

        for edge in new_train_graph.edges.data():
            s = train_graph.idx2node[edge[0]]
            o = train_graph.idx2node[edge[1]]
            r = edge[2]["label"]
            t = edge[2]["time"]

            new_train_file.write(f"""{s}\t{r}\t{o}\t{t}""")
        
        for node, data in new_train_graph.nodes.data():
            s = train_graph.idx2node[node]
            if len(data) > 0:
                attributes = data["attr"]
                for attribute in attributes:
                    for key, value in attribute.items():
                        if key == "time":
                            t = value
                        elif key.startswith("P"):
                            a = key
                            x = value
                        else:
                            print(f"Unerwarteter Key: {key}")
                        new_train_file.write(f"""{s}\t{a}\t{x}\t{t}""")

    with open(new_test_path, "a") as new_test_file:

        for edge in new_test_graph.edges.data():
            s = test_graph.idx2node[edge[0]]
            o = test_graph.idx2node[edge[1]]
            r = edge[2]["label"]
            t = edge[2]["time"]

            new_test_file.write(f"""{s}\t{r}\t{o}\t{t}""")
        
        for node, data in new_test_graph.nodes.data():
            s = test_graph.idx2node[node]
            if len(data) > 0:
                attributes = data["attr"]
                for attribute in attributes:
                    for key, value in attribute.items():
                        if key == "time":
                            t = value
                        elif key.startswith("P"):
                            a = key
                            x = value
                        else:
                            print(f"Unerwarteter Key: {key}")
                        new_test_file.write(f"""{s}\t{a}\t{x}\t{t}""")

    with open(new_valid_path, "a") as new_valid_file:

        for edge in new_valid_graph.edges.data():
            s = valid_graph.idx2node[edge[0]]
            o = valid_graph.idx2node[edge[1]]
            r = edge[2]["label"]
            t = edge[2]["time"]

            new_valid_file.write(f"""{s}\t{r}\t{o}\t{t}""")
        
        for node, data in new_valid_graph.nodes.data():
            s = valid_graph.idx2node[node]
            if len(data) > 0:
                attributes = data["attr"]
                for attribute in attributes:
                    for key, value in attribute.items():
                        if key == "time":
                            t = value
                        elif key.startswith("P"):
                            a = key
                            x = value
                        else:
                            print(f"Unerwarteter Key: {key}")
                        new_valid_file.write(f"""{s}\t{a}\t{x}\t{t}""")


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--method", type=str, default="forest_fire", help="Choose the sampling algorithm: forest_fire or random_walk")
def main(data_home, method):
    train_path = data_home + "\\train"
    test_path = data_home + "\\test"
    valid_path = data_home + "\\valid"
    train_sample_path = data_home + "\\train_sample"
    test_sample_path = data_home + "\\test_sample"
    valid_sample_path = data_home + "\\valid_sample"
    if method == "forest_fire":
        forest_fire_sampling(train_path=train_path, test_path=test_path, valid_path=valid_path, new_train_path=train_sample_path, new_test_path=test_sample_path, new_valid_path=valid_sample_path)
    elif method == "random_walk":
        random_walk_sampling(train_path=train_path, test_path=test_path, valid_path=valid_path, new_train_path=train_sample_path, new_test_path=test_sample_path, new_valid_path=valid_sample_path)
    elif method == "no_sampling":
        train_graph = GraphLoader(train_path)
        test_graph = GraphLoader(test_path)
        valid_graph = GraphLoader(valid_path)

        train_graph.load_graph()
        test_graph.load_graph()
        valid_graph.load_graph()

        write_sampled_data(train_sample_path, train_graph.graph, train_graph, test_sample_path, test_graph.graph, test_graph, valid_sample_path, valid_graph.graph, valid_graph)
    else:
        print("Method not known: Choose between forest_fire and random_walk")
        

if __name__ == '__main__':
    main()