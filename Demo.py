from InsertData import CreateGraph

relations_uri = {"http://demo.slate/framework#" : "slate"}
connection = 'http://localhost:8890/sparql'
graph = CreateGraph(folder="DemoData", graph_name="Demo", relations_uri=relations_uri,
                  connection_string=connection, main_uri="http://demo.slate/framework#",language='pt')
graph.create_graph(in_memory=False)#, save_file = "Serialized")