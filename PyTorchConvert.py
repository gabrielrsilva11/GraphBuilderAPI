from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
import torch
from sklearn import preprocessing
from torch_geometric.data import HeteroData
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

def fetch_graph(random_list):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    graph = ntx.fetch_graph(random_list)
    number_of_words = 0
    le = preprocessing.LabelEncoder()
    for s, p, o in graph.triples((None, RDF.type, URIRef("http://ieeta-bit.pt/wikiner#Word"))):
        number_of_words += 1
        entity = ''
        for s2, p2, o2 in graph.triples((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), None)):
            entity = o2
        if entity:
            entity_no_break = entity.split("\n")
            entity_correct = entity_no_break[0].split("-")
            graph.remove((s2, p2, o2))
            graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal(entity_correct[1])))
        else:
            graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal("No")))

    return graph

def fetch_graph_info(graph, nodes_uri, edges_uri):
    edge_df = pd.DataFrame()
    index_list = []
    nodes_list = []
    depgraph_list = []
    head_list = []
    prev_list = []
    for s, p, o in graph.triples((None, RDF.type, URIRef(nodes_uri))):
        node_features = {}
        #Index to build the node/edges dataframe.
        #In the edges this index will be the "source" node
        index_list.append(s.__str__())
        for s2, p2, o2 in graph.triples((URIRef(s), None, None)):
            #Add the edges information
            if p2.__str__() in edges_uri:
                if p2.__str__() == depgraph_uri:
                    depgraph_list.append([s2.__str__(), o2.__str__()])
                elif p2.__str__() == head_uri:
                    head_list.append([s2.__str__(), o2.__str__()])
                elif p2.__str__() == previous_uri:
                    prev_list.append([s2.__str__(), o2.__str__()])

            #Add the node features
            else:
                node_feat_split = p2.__str__().split("#")
                feat_name = node_feat_split[-1]
                node_features[feat_name] = o2.__str__()
        nodes_list.append(node_features)

    #map word URI to consecutive values
    unique_ids_df = pd.DataFrame(data={
        'originalId': index_list,
        'mappedId': pd.RangeIndex(len(index_list))
    })

    #AUTOMATIZAR ISTO E FAZER UMA FUNCAO PARA NAO ESTAR TAO GRANDE.
    #Build the previousWord relationship: word - previousWord - word
    original_depgraph_df = pd.DataFrame(data=prev_list, columns=["source", "target"])
    source_depgraph_df = pd.merge(original_depgraph_df['source'], unique_ids_df,
                                  left_on='source', right_on='originalId', how='left')
    source_depgraph = torch.from_numpy(source_depgraph_df['mappedId'].values)
    target_depgraph_df = pd.merge(original_depgraph_df['target'], unique_ids_df,
                                  left_on='target', right_on='originalId', how='left')
    target_depgraph = torch.from_numpy((target_depgraph_df['mappedId'].values))
    prev_depgraph = torch.stack([source_depgraph, target_depgraph], dim=0)


    #Build the head relationships: word - head - word
    original_depgraph_df = pd.DataFrame(data=head_list, columns=["source", "target"])
    source_depgraph_df = pd.merge(original_depgraph_df['source'], unique_ids_df,
                                  left_on='source', right_on='originalId', how='left')
    source_depgraph = torch.from_numpy(source_depgraph_df['mappedId'].values)
    target_depgraph_df = pd.merge(original_depgraph_df['target'], unique_ids_df,
                                  left_on='target', right_on='originalId', how='left')
    target_depgraph = torch.from_numpy((target_depgraph_df['mappedId'].values))
    head_depgraph = torch.stack([source_depgraph, target_depgraph], dim=0)

    #Build the depgraph relationship: word - depgraph - word
    original_depgraph_df = pd.DataFrame(data=depgraph_list, columns=["source", "target"])
    source_depgraph_df = pd.merge(original_depgraph_df['source'], unique_ids_df,
                            left_on='source', right_on='originalId', how='left')
    source_depgraph = torch.from_numpy(source_depgraph_df['mappedId'].values)
    target_depgraph_df = pd.merge(original_depgraph_df['target'], unique_ids_df,
                            left_on='target', right_on='originalId', how='left')
    target_depgraph = torch.from_numpy((target_depgraph_df['mappedId'].values))
    edge_depgraph = torch.stack([source_depgraph, target_depgraph], dim=0)

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
    return nodes_tensor, edge_depgraph, head_depgraph, prev_depgraph, targets, unique_ids_df, unique_targets_df

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


words_uri = "http://ieeta-bit.pt/wikiner#Word"
sentence_uri = "http://ieeta-bit.pt/wikiner#Sentence"

depgraph_uri = "http://ieeta-bit.pt/wikiner#depGraph"
head_uri = "http://ieeta-bit.pt/wikiner#head"
previous_uri = "http://ieeta-bit.pt/wikiner#previousWord"
fromsentence_uri = "http://ieeta-bit.pt/wikiner#fromSentence"
fromtext_uri = "http://ieeta-bit.pt/wikiner#fromText"


#graph = fetch_graph([*range(1, 300, 1)])
graph = fetch_graph([1,2,3])
nodes, edged_graphs, head_graphs, prev_graphs, targets, mapped_ids, target_ids = fetch_graph_info(graph, words_uri, [depgraph_uri, head_uri, previous_uri, fromsentence_uri])


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
print(data)
# data['word', 'fromSentence', 'sentence']

# --- sentence edges ---
# data['sentence', '']

#Dataset Split
# train_loader = NeighborLoader(
#     data,
#     # Sample 15 neighbors for each node and each edge type for 2 iterations:
#     num_neighbors=[15] * 2,
#     # Use a batch size of 128 for sampling training nodes of type "paper":
#     batch_size=128,
#     input_nodes=('word', data['word'].x),
# )

assert data["word"].num_nodes == 81
assert data["word"].num_features == 24
assert data["word", "depgraph", "word"].num_edges == 78
assert data["word", "head", "word"].num_edges == 78
assert data["word", "previousWord", "word"].num_edges == 78

#Model Creation
model = GNN(hidden_channels=64, out_channels=data.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['word'].x
    loss = F.cross_entropy(out['word'][mask], data['word'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


loss_final = train()
