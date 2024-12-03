from graphviz import Digraph

# Initialize Digraph with increased spacing
dot = Digraph(format='png')
dot.attr(rankdir='TB', size='10', nodesep='1.2', ranksep='0.1')  # Increase spacing
dot.attr(dpi='300')  # Set high DPI for better resolution

# Define node styles
data_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgrey'}
process_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
alignment_attrs = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': 'lightgreen'}
embedding_attrs = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightyellow'}

# Data Nodes
dot.node('rna_data', 'scRNA Data', **data_attrs)
dot.node('protein_data', 'scProtein Data', **data_attrs)
dot.node('major_cell_types', 'Major Cell Types Labels\nRNA+Protein', **data_attrs)

# Archetype Vector Basiss
dot.node('rna_archetypes', 'RNA Archetype Basis', **process_attrs)
dot.node('protein_archetypes', 'Protein Archetype Basis', **process_attrs)

# Edges from data to vector basis
dot.edge('rna_data', 'rna_archetypes',label='Extract Polygone Vertices (PCHA)')
dot.edge('protein_data', 'protein_archetypes',label='Extract Polygone Vertices (PCHA)')

# Alignment Node
dot.node('align_basis', 'Align Vector Basiss\nfor Comparable Representation', **alignment_attrs)

# Edges from vector basis to alignment
dot.edge('rna_archetypes', 'align_basis')
dot.edge('protein_archetypes', 'align_basis')
dot.edge('major_cell_types', 'align_basis')

# Aligned Basiss Outputs
dot.node('aligned_rna_basis', 'Aligned RNA Archetype Basis', **process_attrs)
dot.node('aligned_protein_basis', 'Aligned Protein Archetype Basis', **process_attrs)

# Edges from alignment to aligned basis
dot.edge('align_basis', 'aligned_rna_basis')
dot.edge('align_basis', 'aligned_protein_basis')

# Cell Embedding Nodes
dot.node('rna_embedding', 'RNA Cell \nArchetype Embeddings', **embedding_attrs)
dot.node('protein_embedding', 'Protein Cell \nArchetype Embeddings', **embedding_attrs)

# Edges from aligned basis to embeddings
dot.edge('aligned_rna_basis', 'rna_embedding',label='NNLS Archetype Linear Combination')
dot.edge('aligned_protein_basis', 'protein_embedding',label='NNLS Archetype Linear Combination')

# Render the graph
dot.render('archetype_matching_pipeline_with_no_detection', view=True)
