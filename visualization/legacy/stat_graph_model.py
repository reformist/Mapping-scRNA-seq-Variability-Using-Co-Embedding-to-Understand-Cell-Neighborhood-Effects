from graphviz import Digraph

# Initialize Digraph with increased spacing to prevent overlap
dot = Digraph(format='png')
dot.attr(rankdir='TB', size='10', nodesep='1.0', ranksep='1.0')  # Increase nodesep and ranksep

# Define node styles
# node_attrs = {'style': 'filled', 'fillcolor': 'lightgrey', 'shape': 'circle'}
node_attrs = {'style': 'filled', 'fillcolor': 'lightgrey', 'shape': 'circle', 'fontsize': '16'}

# Define the main cluster for Cells N that encompasses everything
with dot.subgraph(name='cluster_cells') as cells:
    cells.attr(label='Cells N', color='black')

    # Define nodes zn and cnn within Cells N
    cells.node('zn', 'z_n\njoint state\nof a cell',style='filled', fillcolor='white', shape='circle', fontsize='16')
    cells.node('cnn', 'CN_n', **node_attrs)

    # Define a subgraph for Genes G within Cells N
    with cells.subgraph(name='cluster_genes') as genes:
        genes.attr(label='', color='black')  # Remove label from the subgraph itself
        genes.node('xng', 'x_ng', **node_attrs)
        genes.node('genes_label', 'Genes G', shape='plaintext', width='0.01', height='0.01')  # Dummy node for label
        genes.edge('xng', 'genes_label', style='invis')  # Invisible edge to place label at bottom

    # Define a subgraph for Proteins T within Cells N
    with cells.subgraph(name='cluster_proteins') as proteins:
        proteins.attr(label='', color='black')  # Remove label from the subgraph itself
        proteins.node('ynt', 'y_nt', **node_attrs)
        proteins.node('proteins_label', 'Proteins T', shape='plaintext', width='0.01', height='0.01')  # Dummy node for label
        proteins.edge('ynt', 'proteins_label', style='invis')  # Invisible edge to place label at bottom

# Define edges with additional spacing and label distance for clarity
dot.edge('zn', 'xng', minlen='2')
dot.edge('zn', 'ynt', minlen='2')

# Define edges from CN_n to both Genes and Proteins, with red color and no labels to avoid text overlap
dot.edge('cnn', 'xng', color='red', minlen='2')
dot.edge('cnn', 'ynt', color='red', minlen='2')

# Render and save the graph
dot.render('graph', view=True)
    