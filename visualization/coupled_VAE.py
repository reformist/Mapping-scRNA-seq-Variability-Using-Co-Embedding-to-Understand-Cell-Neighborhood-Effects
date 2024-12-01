from graphviz import Digraph

# Initialize Digraph with increased spacing
dot2 = Digraph(format='png')
dot2.attr(rankdir='TB', size='10', nodesep='1.2', ranksep='0.1')  # Increase nodesep and ranksep
dot2.attr(dpi='300')  # Set high DPI for better resolution

# Define node styles
process_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
data_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgrey'}
embedding_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'lightyellow'}
embedding_attrs_2 = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'yellow'}
model_attrs = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': 'lightgreen'}

# Data and Matched Archetypes Nodes
dot2.node('matched_archetypes_rna', 'Matched RNA Archetypes', **embedding_attrs_2)
dot2.node('matched_archetypes_protein', 'Matched Protein Archetypes', **embedding_attrs_2)

# VAE Models Nodes
dot2.node('vae_rna', 'VAE for RNA', **model_attrs)
dot2.node('vae_protein', 'VAE for Protein', **model_attrs)

# Edges from data and matched archetypes to VAEs
dot2.edge('matched_archetypes_rna', 'vae_rna')
dot2.edge('matched_archetypes_protein', 'vae_protein')

# Dual VAE Training Node
dot2.node('dual_vae_training', 'Dual VAE Training with\nContrastive Learning', **process_attrs)

# Edges from VAEs to Dual VAE Training
dot2.edge('vae_rna', 'dual_vae_training')
dot2.edge('vae_protein', 'dual_vae_training')

# Loss Terms Node
dot2.node('loss_terms', 'Loss Terms', **process_attrs)

# Edges from Dual VAE Training to Loss Terms
dot2.edge('dual_vae_training', 'loss_terms')

# Individual Loss Nodes
dot2.node('cn_difference_loss', 'CN Difference Loss', **process_attrs)
dot2.node('matching_loss', 'Matching Loss', **process_attrs)
dot2.node('reconstruction_loss', 'Reconstruction Loss', **process_attrs)

# Edges from Loss Terms to Individual Losses
dot2.edge('loss_terms', 'cn_difference_loss')
dot2.edge('loss_terms', 'matching_loss')
dot2.edge('loss_terms', 'reconstruction_loss')

# Latent Space Nodes
dot2.node('latent_space_rna', 'RNA Latent Space', **embedding_attrs)
dot2.node('latent_space_protein', 'Protein Latent Space', **embedding_attrs)

# Edges from Dual VAE Training to Latent Spaces
dot2.edge('dual_vae_training', 'latent_space_rna', label='RNA Encoder')
dot2.edge('dual_vae_training', 'latent_space_protein', label='Protein Encoder')

# Add a two-way arrow between the latent space nodes
dot2.edge('latent_space_rna', 'latent_space_protein', dir='both')

# Render the graph
dot2.render('vae_creation_pipeline_with_two_way_arrow', view=True)
