from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
import torch
from sklearn import preprocessing
from torch_geometric.data import HeteroData
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
import tqdm
import yaml

def fetch_graph(random_list, nodes_uri, edges_uri):
    print("--- LOADING DATASET ---")
    index_list = []
    nodes_list = []
    depgraph_list = []
    head_list = []
    prev_list = []
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    add = 0
    for idx in random_list:
        graph = ntx.fetch_graph([idx])
        # number_of_words = 0
        for s, p, o in graph.triples((None, RDF.type, URIRef("http://ieeta-bit.pt/wikiner#Word"))):
            # number_of_words += 1
            entity = ''
            for s2, p2, o2 in graph.triples((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), None)):
                entity = o2
            if entity:
                entity_no_break = entity.split("\n")
                entity_correct = entity_no_break[0].split("-")
                graph.remove((s2, p2, o2))
                graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal(entity_correct[1])))
            else:
                add += 1
                if add == 20:
                    graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal("No")))
                    add = 0

        depgraph_list, head_list, prev_list, nodes_list, index_list = fetch_graph_info(graph, nodes_uri, edges_uri,
                                                                                       depgraph_list, head_list,
                                                                                       prev_list, nodes_list,
                                                                                       index_list)
    #map word URI to consecutive values
    unique_ids_df = pd.DataFrame(data={
        'originalId': index_list,
        'mappedId': pd.RangeIndex(len(index_list))
    })

    #Build the previousWord relationship: word - previousWord - word
    previous_graph = build_node_relationships(unique_ids_df, prev_list)
    head_graph = build_node_relationships(unique_ids_df, head_list)
    depgraph_graph = build_node_relationships(unique_ids_df, depgraph_list)

    #Build dataframe with wordId -> features of each word
    nodes_df = pd.DataFrame.from_records(nodes_list, index=unique_ids_df['mappedId'])
    nodes_df.index.name = "wordId"
    feats = nodes_df['feats'].str.get_dummies('|')
    nodes_df = nodes_df.join(feats)
    nodes_df = nodes_df.drop(['feats', 'type', 'id'], axis=1)
    nodes_df.edge = nodes_df.edge.astype('category').cat.codes
    nodes_df.lemma = nodes_df.lemma.astype('category').cat.codes
    nodes_df.pos = nodes_df.pos.astype('category').cat.codes
    nodes_df.poscoarse = nodes_df.poscoarse.astype('category').cat.codes
    nodes_df.word = nodes_df.word.astype('category').cat.codes

    #handle the Y (targets)
    #targets = nodes_df.pop('wikinerEntity')
    unique_targets_id = nodes_df['wikinerEntity'].unique()
    unique_targets_df = pd.DataFrame(data={
        'originalId': unique_targets_id,
        'mappedId': pd.RangeIndex(len(unique_targets_id))
    })
    targets_df = pd.merge(nodes_df['wikinerEntity'], unique_targets_df,
                                  left_on='wikinerEntity', right_on='originalId', how='left')
    targets = torch.from_numpy(targets_df['mappedId'].values)
    nodes_df = nodes_df.drop(['wikinerEntity'], axis=1)
    nodes_tensor = torch.from_numpy(nodes_df.values).to(torch.float)
    return nodes_tensor, depgraph_graph, head_graph, previous_graph, targets, unique_ids_df, unique_targets_df
    #return graph


def build_node_relationships(uniqueIds, nodeList):
    original_df = pd.DataFrame(data=nodeList, columns=["source", "target"])
    source_df = pd.merge(original_df['source'], uniqueIds,
                         left_on='source', right_on='originalId', how='left')
    source = torch.from_numpy(source_df['mappedId'].values)
    target_df = pd.merge(original_df['target'], uniqueIds,
                         left_on='target', right_on='originalId', how='left')
    target = torch.from_numpy((target_df['mappedId'].values))
    node_tensor = torch.stack([source, target], dim=0)
    return node_tensor


def fetch_graph_info(graph, nodes_uri, edges_uri, depgraph_list, head_list, prev_list, nodes_list, index_list):
    # index_list = []
    # nodes_list = []
    # depgraph_list = []
    # head_list = []
    # prev_list = []
    for s, p, o in graph.triples((None, RDF.type, URIRef(nodes_uri))):
        node_features = {}
        # Index to build the node/edges dataframe.
        # In the edges this index will be the "source" node
        index_list.append(s.__str__())
        for s2, p2, o2 in graph.triples((URIRef(s), None, None)):
            # Add the edges information
            if p2.__str__() in edges_uri:
                if p2.__str__() == depgraph_uri:
                    depgraph_list.append([s2.__str__(), o2.__str__()])
                elif p2.__str__() == head_uri:
                    head_list.append([s2.__str__(), o2.__str__()])
                elif p2.__str__() == previous_uri:
                    prev_list.append([s2.__str__(), o2.__str__()])

            # Add the node features
            else:
                node_feat_split = p2.__str__().split("#")
                feat_name = node_feat_split[-1]
                node_features[feat_name] = o2.__str__()
        nodes_list.append(node_features)
    return depgraph_list, head_list, prev_list, nodes_list, index_list
    # map word URI to consecutive values
    # unique_ids_df = pd.DataFrame(data={
    #     'originalId': index_list,
    #     'mappedId': pd.RangeIndex(len(index_list))
    # })
    #
    # #Build the previousWord relationship: word - previousWord - word
    # previous_graph = build_node_relationships(unique_ids_df, prev_list)
    # head_graph = build_node_relationships(unique_ids_df, head_list)
    # depgraph_graph = build_node_relationships(unique_ids_df, depgraph_list)
    #
    # #Build dataframe with wordId -> features of each word
    # nodes_df = pd.DataFrame.from_records(nodes_list, index=unique_ids_df['mappedId'])
    # nodes_df.index.name = "wordId"
    # feats = nodes_df['feats'].str.get_dummies('|')
    # nodes_df = nodes_df.join(feats)
    # nodes_df = nodes_df.drop(['feats', 'type', 'id'], axis=1)
    # nodes_df.edge = nodes_df.edge.astype('category').cat.codes
    # nodes_df.lemma = nodes_df.lemma.astype('category').cat.codes
    # nodes_df.pos = nodes_df.pos.astype('category').cat.codes
    # nodes_df.poscoarse = nodes_df.poscoarse.astype('category').cat.codes
    # nodes_df.word = nodes_df.word.astype('category').cat.codes
    #
    # #handle the Y (targets)
    # #targets = nodes_df.pop('wikinerEntity')
    # unique_targets_id = nodes_df['wikinerEntity'].unique()
    # unique_targets_df = pd.DataFrame(data={
    #     'originalId': unique_targets_id,
    #     'mappedId': pd.RangeIndex(len(unique_targets_id))
    # })
    # targets_df = pd.merge(nodes_df['wikinerEntity'], unique_targets_df,
    #                               left_on='wikinerEntity', right_on='originalId', how='left')
    # targets = torch.from_numpy(targets_df['mappedId'].values)
    # nodes_df = nodes_df.drop(['wikinerEntity'], axis=1)
    # nodes_tensor = torch.from_numpy(nodes_df.values).to(torch.float)
    # return nodes_tensor, depgraph_graph, head_graph, previous_graph, targets, unique_ids_df, unique_targets_df


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data_split.x_dict, data_split.edge_index_dict)
    mask = data_split['word'].train_mask
    loss = F.cross_entropy(out['word'][mask], data_split['word'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


words_uri = "http://ieeta-bit.pt/wikiner#Word"
sentence_uri = "http://ieeta-bit.pt/wikiner#Sentence"

depgraph_uri = "http://ieeta-bit.pt/wikiner#depGraph"
head_uri = "http://ieeta-bit.pt/wikiner#head"
previous_uri = "http://ieeta-bit.pt/wikiner#previousWord"
fromsentence_uri = "http://ieeta-bit.pt/wikiner#fromSentence"
fromtext_uri = "http://ieeta-bit.pt/wikiner#fromText"

# graph = fetch_graph([*range(1, 300, 1)])
nodes, edged_graphs, head_graphs, prev_graphs, targets, mapped_ids, target_ids = fetch_graph([*range(1, 2000, 1)],
                                                                                             nodes_uri=words_uri,
                                                                                             edges_uri=[depgraph_uri, head_uri, previous_uri, fromsentence_uri])

# nodes, edged_graphs, head_graphs, prev_graphs, targets, mapped_ids, target_ids = fetch_graph_info(graph, words_uri, [depgraph_uri, head_uri, previous_uri, fromsentence_uri])


data = HeteroData()
# --- NODES ---
data['word'].x = nodes
data['word'].y = targets
data.num_classes = len(target_ids)
# data['sentence']

# --- EDGES ---
# --- word edges ---
data['word', 'depgraph', 'word'].edge_index = edged_graphs
data['word', 'head', 'word'].edge_index = head_graphs
data['word', 'previousWord', 'word'].edge_index = prev_graphs
# data['word', 'fromSentence', 'sentence']

# --- sentence edges ---
# data['sentence', '']


# CREATING THE TRAIN, TEST AND VAL MASKS
print(data)
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
data_split = split(data)
print(data_split)

model = GAT(hidden_channels=16, out_channels=data.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Pass data onto GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
data_split = data_split.to(device)

for epoch in range(1, 50):
    print("---- Epoch ", epoch, "----")
    loss_final = train()
    print(loss_final)
    print("Loss: ", loss_final)

# out = model(data_split.x_dict, data_split.edge_index_dict)
# results = out["word"][data_split["word"].test_mask]
# truth = data_split['word'].y[data_split['word'].test_mask]
# print(results)
# print(truth)
print(data_split["word"].train_mask)
print(data_split["word"].val_mask)
preds = []
out = model(data_split.x_dict, data_split.edge_index_dict)
mask = data_split['word'].test_mask
#probabilities = torch.nn.functional.softmax(out['word'][mask], dim=0)
probabilities = out['word']['mask'].argmax(dim=1)
truth = data_split['word'].y[mask]
print(probabilities)
print(truth)
