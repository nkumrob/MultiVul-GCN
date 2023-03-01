import matplotlib
from parser_1 import _parser
matplotlib.use('agg')
import numpy as np
import os
from os.path import join as pjoin
from load_data import split_data

args = _parser()
n_folds = args.folds

class DataReader():
        """
        This class reads the text files containing data from training data folder
        """

        def __init__(self, data_dir, rnd_state=None, use_cont_node_attr=False, folds=n_folds):
            self.data_dir = data_dir
            self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
            self.use_cont_node_attr = use_cont_node_attr
            files = os.listdir(self.data_dir)
            data = {}
            nodes, graphs, unique_id = self.read_graph_nodes_relations(
                list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
            data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                    nodes, graphs, fn=lambda s: int(s.strip()))
            data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
            data['targets'] = np.array(
                self.parse_txtfile(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                    line_parse_fn=lambda s: int(float(s.strip()))))
            data['ids'] = unique_id
            if self.use_cont_node_attr:
                data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                    nodes, graphs,
                                                    fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
            features, n_edges, degrees = [], [], []
            for sample_id, adj in enumerate(data['adj_list']):
                N = len(adj)  # number of nodes
                if data['features'] is not None:
                    assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
                n = np.sum(adj)  # total sum of edges
                # assert n % 2 == 0, n
                n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
                if not np.allclose(adj, adj.T):
                    print(sample_id, 'not symmetric')
                degrees.extend(list(np.sum(adj, 1)))
                features.append(np.array(data['features'][sample_id]))

            # Create features over graphs as one-hot vectors for each node
            features_all = np.concatenate(features)
            features_min = features_all.min()
            num_features = int(features_all.max() - features_min + 1)  # number of possible values

            features_onehot = []
            for i, x in enumerate(features):
                feature_onehot = np.zeros((len(x), num_features))
                for node, value in enumerate(x):
                    feature_onehot[node, value - features_min] = 1
                if self.use_cont_node_attr:
                    feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
                features_onehot.append(feature_onehot)

            if self.use_cont_node_attr:
                num_features = features_onehot[0].shape[1]

            shapes = [len(adj) for adj in data['adj_list']]
            labels = data['targets']  # graph class labels
            labels -= np.min(labels)  # to start from 0

            classes = np.unique(labels)
            num_classes = len(classes)

            if not np.all(np.diff(classes) == 1):
                print('making labels sequential, otherwise pytorch might crash')
                labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
                for lbl in range(num_classes):
                    labels_new[labels == classes[lbl]] = lbl
                labels = labels_new
                classes = np.unique(labels)
                assert len(np.unique(labels)) == num_classes, np.unique(labels)

            for lbl in classes:
                print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

            for u in np.unique(features_all):
                print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

            N_graphs = len(labels)  # number of samples (graphs) in training_data
            assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid training_data'

            # Create test sets first
            train_ids, test_ids = split_data(rnd_state.permutation(N_graphs), folds=folds)

            # Create train sets
            splits = []
            for fold in range(folds):
                splits.append({'train': train_ids[fold], 'test': test_ids[fold]})

            data['features_onehot'] = features_onehot
            data['targets'] = labels
            data['splits'] = splits
            data['N_nodes_max'] = np.max(shapes)  # max number of nodes
            data['num_features'] = num_features
            data['num_classes'] = num_classes
            self.data = data

        def parse_txtfile(self, fpath, line_parse_fn=None):
            with open(pjoin(self.data_dir, fpath), 'r') as f:
                lines = f.readlines()
            data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
            return data

        def read_graph_adj(self, fpath, nodes, graphs):
            edges = self.parse_txtfile(fpath, line_parse_fn=lambda s: s.split(','))
            adj_dict = {}
            for edge in edges:
                node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
                node2 = int(edge[1].strip()) - 1
                graph_id = nodes[node1]
                assert graph_id == nodes[node2], ('invalid training_data', graph_id, nodes[node2])
                if graph_id not in adj_dict:
                    n = len(graphs[graph_id])
                    adj_dict[graph_id] = np.zeros((n, n))
                ind1 = np.where(graphs[graph_id] == node1)[0]
                ind2 = np.where(graphs[graph_id] == node2)[0]
                assert len(ind1) == len(ind2) == 1, (ind1, ind2)
                adj_dict[graph_id][ind1, ind2] = 1
            adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
            return adj_list

        def read_graph_nodes_relations(self, fpath):
            graph_ids = self.parse_txtfile(fpath, line_parse_fn=lambda s: int(s.rstrip()))
            nodes, graphs = {}, {}
            for node_id, graph_id in enumerate(graph_ids):
                if graph_id not in graphs:
                    graphs[graph_id] = []
                graphs[graph_id].append(node_id)
                nodes[node_id] = graph_id
            graph_ids = np.unique(list(graphs.keys()))
            unique_id = graph_ids
            for graph_id in graph_ids:
                graphs[graph_id] = np.array(graphs[graph_id])
            return nodes, graphs, unique_id

        def read_node_features(self, fpath, nodes, graphs, fn):
            node_features_all = self.parse_txtfile(fpath, line_parse_fn=fn)
            node_features = {}
            for node_id, x in enumerate(node_features_all):
                graph_id = nodes[node_id]
                if graph_id not in node_features:
                    node_features[graph_id] = [None] * len(graphs[graph_id])
                ind = np.where(graphs[graph_id] == node_id)[0]
                assert len(ind) == 1, ind
                assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
                node_features[graph_id][ind[0]] = x
            node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
            return node_features_lst