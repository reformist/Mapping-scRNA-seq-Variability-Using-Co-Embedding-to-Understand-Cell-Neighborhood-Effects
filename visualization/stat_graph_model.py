from graphviz import Digraph

# Initialize Digraph
dot = Digraph(format='png')

# Set graph attributes
dot.attr(rankdir='TB', size='10')

# Define node styles
node_attrs = {'style': 'filled', 'fillcolor': 'lightgrey', 'shape': 'circle'}

# Define the main cluster for Cells N that encompasses everything
with dot.subgraph(name='cluster_cells') as cells:
    cells.attr(label='Cells N', color='black')

    # Define nodes sn and zn within Cells N
    cells.node('zn', 'z_n\njoint state\nof a cell')

    # Add CN_n as a separate node
    cells.node('cnn', 'CN_n', **node_attrs)

    # Define a subgraph for Genes G within Cells N
    with cells.subgraph(name='cluster_genes') as genes:
        genes.attr(label='Genes G')
        genes.node('xng', 'x_ng', **node_attrs)

    # Define a subgraph for Proteins T within Cells N
    with cells.subgraph(name='cluster_proteins') as proteins:
        proteins.attr(label='Proteins T')
        proteins.node('ynt', 'y_nt', **node_attrs)

# Define edges
dot.edge('zn', 'xng')
dot.edge('zn', 'ynt')

# Define edges from CN_n to both Genes and Proteins
dot.edge('cnn', 'xng', label='', color='red', fontcolor='red')
dot.edge('cnn', 'ynt', label='', color='red', fontcolor='red')

# Render and save the graph
dot.render('graph', view=True)
