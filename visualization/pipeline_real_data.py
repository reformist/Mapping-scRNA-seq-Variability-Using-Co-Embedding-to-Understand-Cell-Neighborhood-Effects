from graphviz import Digraph

# Initialize Digraph
dot = Digraph(format='png')

# Set graph attributes
dot.attr(rankdir='TB', size='10')

# Define node styles
process_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
data_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgrey'}
embedding_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'lightyellow'}
model_attrs = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': 'lightgreen'}

# Data Generation Step
dot.node('rna_data', 'scRNA Data', **data_attrs)
dot.node('protein_data', 'CODEX Protein Data', **data_attrs)
dot.node('cn_data', 'CN (Cell Neighborhoods)', **data_attrs)

# Arrows for Synthetic Data Generation Step

# MaxFuse Co-embedding Step
dot.node('maxfuse', 'MaxFuse Co-embedding\n(RNA and Protein)', **process_attrs)
dot.node('co_embedding', 'Co-embedding Space\n(RNA & Protein)', **embedding_attrs)

# Arrows for MaxFuse Step
dot.edge('rna_data', 'maxfuse', label='Input')
dot.edge('protein_data', 'maxfuse', label='Input')
dot.edge('protein_data', 'cn_data', label='Input')
dot.edge('maxfuse', 'co_embedding', label='Outputs')

# VAE with Integrated Contrastive Learning
dot.node('vae_contrastive', 'VAE with Integrated\nContrastive Learning', **model_attrs)

# Plate for Latent Space and Contrastive Learning
with dot.subgraph(name='cluster_plate') as plate:
    plate.attr(label='VAE with Contrastive Learning', color='black', style='dashed')
    plate.node('latent_space', 'Latent Space', **embedding_attrs)
    plate.node('contrastive', 'Contrastive Loss\non CN and Cell Type', **process_attrs)

    # Edges within the Plate
    plate.edge('co_embedding', 'vae_contrastive', label='Input Co-embedding')
    plate.edge('vae_contrastive', 'latent_space', label='Encodes to')
    plate.edge('contrastive', 'latent_space', label='Refines')

    # Invisible node for plate connection
    plate.node('plate_rep', '', shape='point')

# Set rank for final node to be at the bottom
with dot.subgraph() as bottom:
    bottom.attr(rank='sink')
    bottom.node('cn_pred', 'Final Model for CN Prediction\n(Input: RNA, Output: CN)', **model_attrs)
    dot.edge('plate_rep', 'cn_pred', label='')
    dot.edge('cn_data', 'cn_pred', label='Validate predictions\nwith ground truth CN labels')

# Additional edges for contrastive learning context
dot.edge('cn_data', 'contrastive', label='Train using CN\nas ground truth')
dot.edge('latent_space', 'contrastive', label='Applies to Latent Space')

# Render the graph
dot.render('vae_contrastive_pipeline', view=True)
