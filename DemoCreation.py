from InsertData import CreateGraph

#relations_uri = {"http://demo.slate/framework#": "slate"}
relations_uri = "http://ieeta-bit.pt/wikiner#"
connection = 'http://localhost:8890/sparql'

graph = CreateGraph(folder="DemoData", graph_name="WikiNER", relations_uri=relations_uri,
                    connection_string=connection, main_uri="http://ieeta-bit.pt/wikiner#", language='pt')

graph.create_graph(in_memory=True, save_file="WikiNER")
