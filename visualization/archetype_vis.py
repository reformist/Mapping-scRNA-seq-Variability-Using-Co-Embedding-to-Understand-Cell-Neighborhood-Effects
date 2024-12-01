from graphviz import Digraph

# Initialize Digraph with increased spacing
dot1 = Digraph(format='png')
dot1.attr(rankdir='TB', size='10', nodesep='1.2', ranksep='0.1')  # Increase nodesep and ranksep
dot1.attr(dpi='300')  # Set high DPI for better resolution

# Define node styles
process_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
data_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgrey'}
embedding_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'lightyellow'}
embedding_attrs_2 = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'yellow'}

# Data Nodes
dot1.node('rna_data', 'scRNA Data', **data_attrs)
dot1.node('protein_data', 'scProtein Data', **data_attrs)

# Archetype Detection Nodes
dot1.node('archetype_detection_rna', 'Archetype Detection\n(RNA)', **process_attrs)
dot1.node('archetype_detection_protein', 'Archetype Detection\n(Protein)', **process_attrs)

# Edges from data to archetype detection
dot1.edge('rna_data', 'archetype_detection_rna')
dot1.edge('protein_data', 'archetype_detection_protein')

# Archetype Outputs
dot1.node('rna_archetypes', 'RNA Archetypes', **embedding_attrs)
dot1.node('protein_archetypes', 'Protein Archetypes', **embedding_attrs)

# Edges from archetype detection to outputs
dot1.edge('archetype_detection_rna', 'rna_archetypes')
dot1.edge('archetype_detection_protein', 'protein_archetypes')

# Archetype Weights Nodes
dot1.node('archetype_weights_rna', 'Compute Archetype Weights\nfor RNA Cells', **process_attrs)
dot1.node('archetype_weights_protein', 'Compute Archetype Weights\nfor Protein Cells', **process_attrs)

# Edges from archetypes to weights computation
dot1.edge('rna_archetypes', 'archetype_weights_rna')
dot1.edge('protein_archetypes', 'archetype_weights_protein')

# Weights Outputs
dot1.node('rna_weights', 'RNA Archetype Weights', **embedding_attrs)
dot1.node('protein_weights', 'Protein Archetype Weights', **embedding_attrs)

# Edges from weights computation to weights outputs
dot1.edge('archetype_weights_rna', 'rna_weights')
dot1.edge('archetype_weights_protein', 'protein_weights')

# Matching Archetypes Node
dot1.node('match_archetypes', 'Match Archetypes\nAcross Modalities', **process_attrs)

# Edges from archetype outputs to matching
dot1.edge('rna_weights', 'match_archetypes')
dot1.edge('protein_weights', 'match_archetypes')

# Matched Archetypes Outputs
dot1.node('matched_archetypes_rna', 'Matched RNA Archetypes', **embedding_attrs_2)
dot1.node('matched_archetypes_protein', 'Matched Protein Archetypes', **embedding_attrs_2)

# Edges from matching to matched archetypes
dot1.edge('match_archetypes', 'matched_archetypes_rna', label='RNA Side')
dot1.edge('match_archetypes', 'matched_archetypes_protein', label='Protein Side')

# Save Matched Archetypes Nodes


# Render the graph
dot1.render('archetype_creation_pipeline_simplified', view=True)
