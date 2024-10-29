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

# Synthetic Data Generation Step
dot.node('starfysh', 'Starfysh\nSynthetic Data Generation', **process_attrs)
dot.node('rna_data', 'Real RNA Data', **data_attrs)
# dot.node('protein_data', 'Real Protein Data', **data_attrs)
dot.node('cn_data', 'Synthetic CN Labels\n(Cell Neighborhoods)', **data_attrs)

# Arrows for Synthetic Data Generation Step
dot.edge('rna_data','starfysh', label='')
dot.edge('starfysh', 'cn_data', label='Generates')
dot.node('gene_selection', 'Gene Selection', **model_attrs)


# VAE with Integrated Contrastive Learning
dot.node('vae_contrastive', 'VAE with Integrated\nContrastive Learning', **model_attrs)
dot.edge('rna_data', 'gene_selection', label='')
dot.edge('gene_selection', 'vae_contrastive', label='')

# Plate for Latent Space and Contrastive Learning
with dot.subgraph(name='cluster_plate') as plate:
    plate.attr(label='VAE with Contrastive Learning', color='black', style='dashed')
    plate.node('latent_space', 'Latent Space', **embedding_attrs)
    plate.node('contrastive', 'Contrastive Loss\non CN and Cell Type', **process_attrs)

    # Edges within the Plate
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
dot.edge('cn_data', 'contrastive', label='train with\nCN Labels')
dot.edge('latent_space', 'contrastive', label='Applies to Latent Space')

# Render the graph
dot.render('vae_contrastive_pipeline', view=True)
