from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
)
import stellargraph as sg
from stellargraph.layer import GCN, DeepGraphInfomax,  GAT
from stellargraph.utils import plot_history
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, preprocessing
import random
from GraphConverterNetworkX import BuildNetworkx
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, callbacks, losses, regularizers, metrics, models
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph


def make_gcn_model(fullbatch_generator):
    return GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=fullbatch_generator,
        dropout=0.4,
    )


def load_dataset_wikiner(start, stop, random_list):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    graph = ntx.fetch_graph(random_list)
    add = 0
    number_of_words = 0
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
            add += 1
            if add == 20:
                graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal("No")))
                add = 0

    triples = pd.DataFrame(
        ((s.n3(), str(p), o.n3()) for s, p, o in graph),
        columns=["source", "label", "target"],
    )

    all_nodes = pd.concat([triples.source, triples.target])
    nodes = pd.DataFrame(index=pd.unique(all_nodes))
    nodes_onehot_features = pd.get_dummies(nodes.index).set_index(nodes.index)

    edges = {
        edge_type: df.drop(columns="label")
        for edge_type, df in triples.groupby("label")
    }

    affiliation = edges.pop("http://ieeta-bit.pt/wikiner#wikinerEntity")
    #affiliation = affiliation.set_index("source")["target"]
    onehot_affiliation = pd.get_dummies(affiliation.set_index("source")["target"])
    graph_sg = sg.StellarDiGraph(nodes_onehot_features, edges)
    return graph_sg, onehot_affiliation


def train_gat(start, end, trainsize, testsize, save_path, random_list):
    G, node_classes = load_dataset_wikiner(start, end, random_list)
    #G, node_classes = load_dataset()
    print("--- DATASET LOADED ---")
    print(node_classes.value_counts().to_frame())
    print(G.info())

    #Start up generators
    generator = FullBatchNodeGenerator(G, method="gat")

    train_classes, test_classes = model_selection.train_test_split(
        node_classes, train_size=trainsize, stratify=node_classes, random_state=1
    )
    val_classes, test_classes = model_selection.train_test_split(
        test_classes, train_size=testsize, stratify=test_classes, random_state=1
    )
    print(len(train_classes), len(test_classes), len(val_classes))
    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_classes)
    val_targets = target_encoding.transform(val_classes)
    test_targets = target_encoding.transform(test_classes)

    train_gen = generator.flow(train_classes.index, train_targets)
    val_gen = generator.flow(val_classes.index, val_targets)

    gat = GAT(
        layer_sizes=[8, train_targets.shape[1]],
        activations=["elu", "softmax"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )

    x_inp, predictions = gat.in_out_tensors()

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.005),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    es_callback = callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=300,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )

    sg.utils.plot_history(history)
    plt.show()
    plt.savefig(save_path)
    test_gen = generator.flow(test_classes.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    print(test_metrics)
    pretrained_test_metrics = dict(
        zip(model.metrics_names, model.evaluate(test_gen))
    )
    print(pretrained_test_metrics)
    return pretrained_test_metrics


def train_gcn(start, end, trainsize, testsize, save_path, random_list):
    G, node_classes = load_dataset_wikiner(start, end, random_list)
    #G, node_classes = load_dataset()
    print("--- DATASET LOADED ---")
    print(node_classes.value_counts().to_frame())
    print(G.info())

    #Start up generators
    fullbatch_generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")

    train_classes, test_classes = model_selection.train_test_split(
        node_classes, train_size=trainsize, stratify=node_classes, random_state=1
    )
    val_classes, test_classes = model_selection.train_test_split(
        test_classes, train_size=testsize, stratify=test_classes, random_state=1
    )
    print(len(train_classes), len(test_classes), len(val_classes))
    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_classes)
    val_targets = target_encoding.transform(val_classes)
    test_targets = target_encoding.transform(test_classes)

    train_gen = fullbatch_generator.flow(train_classes.index, train_targets)
    val_gen = fullbatch_generator.flow(val_classes.index, val_targets)

    gcn = make_gcn_model(fullbatch_generator)

    x_inp, x_out = gcn.in_out_tensors()

    predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    es_callback = callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=500,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    sg.utils.plot_history(history)
    plt.savefig(save_path)
    plt.show()

    test_gen = fullbatch_generator.flow(test_classes.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    print(test_metrics)
    pretrained_test_metrics = dict(
        zip(model.metrics_names, model.evaluate(test_gen))
    )
    print(pretrained_test_metrics)
    return pretrained_test_metrics


def train_infomax_gcn(start, end, training_size, test_size, save_path, random_list):
    G, node_classes = load_dataset_wikiner(start, end, random_list)
    #G, node_classes = load_dataset()
    print("--- DATASET LOADED ---")
    print(node_classes.value_counts().to_frame())
    print(G.info())

    #Start up generators
    fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
    pretrained_gcn_model = make_gcn_model(fullbatch_generator)
    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(G.nodes())

    train_classes, test_classes = model_selection.train_test_split(
        node_classes, train_size=training_size, stratify=node_classes, random_state=1
    )
    val_classes, test_classes = model_selection.train_test_split(
        test_classes, train_size=test_size, stratify=test_classes, random_state=1
    )
    print(len(train_classes), len(test_classes), len(val_classes))
    target_encoding = preprocessing.LabelBinarizer()

    infomax = DeepGraphInfomax(pretrained_gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    dgi_model = Model(inputs=x_in, outputs=x_out)
    dgi_model.compile(
        loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=optimizers.Adam(learning_rate=1e-3)
    )

    epochs = 500
    dgi_es = callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)
    dgi_history = dgi_model.fit(gen, epochs=epochs, verbose=2, callbacks=[dgi_es])
    plot_history(dgi_history)
    plt.show()

    train_targets = target_encoding.fit_transform(train_classes)
    val_targets = target_encoding.transform(val_classes)
    test_targets = target_encoding.transform(test_classes)

    train_gen = fullbatch_generator.flow(train_classes.index, train_targets)
    val_gen = fullbatch_generator.flow(val_classes.index, val_targets)

    pretrained_x_in, pretrained_x_out = pretrained_gcn_model.in_out_tensors()

    pretrained_predictions = tf.keras.layers.Dense(
        units=train_targets.shape[1], activation="softmax"
    )(pretrained_x_out)

    pretrained_model = Model(inputs=pretrained_x_in, outputs=pretrained_predictions)
    pretrained_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["acc"],
    )

    prediction_es = callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )

    pretrained_history = pretrained_model.fit(
        train_gen,
        epochs=epochs,
        verbose=2,
        validation_data=val_gen,
        callbacks=[prediction_es],
    )

    test_gen = fullbatch_generator.flow(test_classes.index, test_targets)
    sg.utils.plot_history(pretrained_history)
    plt.savefig(save_path)
    plt.show()

    pretrained_test_metrics = dict(
        zip(pretrained_model.metrics_names, pretrained_model.evaluate(test_gen))
    )
    print(pretrained_test_metrics)

    return pretrained_test_metrics

#def train_gcn():

def main():
    results = []
    #test_metrics = train_infomax_gcn(2123, 2422, 0.7, 0.7)
    dict_sizes = {0.8: "8020", 0.7: "7030", 0.5: "5050", 0.3: "3070", 0.2: "2080"}
    random.seed(123)
    random_list = random.sample(range(1, 3000), 300)
    sizes = [0.8, 0.7, 0.5, 0.3, 0.2]
    #sizes = [0.3, 0.2]
    for size in sizes:
        print(size)
        save_path = "C:\\Users\\Gabriel\\Documents\\GitHub\\GraphBuilderAPI\\Grapfs\\InfomaxGCN\\Training_Acc\\trainingacc_"+dict_sizes[size] +"_random_infomaxgcn.png"
        test_metrics = train_infomax_gcn(0, 300, size, size, save_path, random_list)
        results.append(test_metrics)
    for i in range(0, len(sizes)):
        print("---- TEST RESULTS ----")
        print("Split: ", sizes[i], "Results: ", results[i])

if __name__ == '__main__':
    main()

