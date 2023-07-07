from InsertData import CreateGraph

#relations_uri = {"http://demo.slate/framework#": "slate"}
relations_uri = "http://demo-ieeta.pt/ontoud#"
connection = 'http://localhost:8890/sparql'

graph = CreateGraph(folder="DemoData", graph_name="Demo", relations_uri=relations_uri,
                    connection_string=connection, main_uri="http://demo-ieeta.pt/ontoud#", language='pt')

graph.create_graph(in_memory=False, save_file="Serialized")
