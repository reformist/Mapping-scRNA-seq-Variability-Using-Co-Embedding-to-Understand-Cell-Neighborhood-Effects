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

# Archetype Vector Bases
dot.node('rna_archetypes', 'RNA Vector Base', **process_attrs)
dot.node('protein_archetypes', 'Protein Vector Base', **process_attrs)

# Edges from data to vector bases
dot.edge('rna_data', 'rna_archetypes')
dot.edge('protein_data', 'protein_archetypes')

# Alignment Node
dot.node('align_bases', 'Align Vector Bases\nfor Comparable Representation', **alignment_attrs)

# Edges from vector bases to alignment
dot.edge('rna_archetypes', 'align_bases')
dot.edge('protein_archetypes', 'align_bases')

# Aligned Bases Outputs
dot.node('aligned_rna_base', 'Aligned RNA Vector Base', **data_attrs)
dot.node('aligned_protein_base', 'Aligned Protein Vector Base', **data_attrs)

# Edges from alignment to aligned bases
dot.edge('align_bases', 'aligned_rna_base')
dot.edge('align_bases', 'aligned_protein_base')

# Cell Embedding Nodes
dot.node('rna_embedding', 'RNA Cell \nArchetype Embeddings', **embedding_attrs)
dot.node('protein_embedding', 'Protein Cell \nArchetype Embeddings', **embedding_attrs)

# Edges from aligned bases to embeddings
dot.edge('aligned_rna_base', 'rna_embedding')
dot.edge('aligned_protein_base', 'protein_embedding')

# Render the graph
dot.render('archetype_matching_pipeline_with_no_detection', view=True)
