from graphviz import Digraph

# Initialize the Digraph
dot = Digraph(format='png')
# dot.attr(rankdir='TB', size='15', nodesep='2', ranksep='1.5')  # Increased spacing for better layout
dot.attr(rankdir='TB', size='10', nodesep='0.4', ranksep='1.5')

dot.attr(dpi='300')  # High resolution

# Define styles
process_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
data_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgrey'}
embedding_attrs = data_attrs
model_attrs = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': 'lightgreen'}
plate_attrs = {'shape': 'plaintext', 'style': 'rounded', 'color': 'black'}

# Plate: The entire DualVAE Model
with dot.subgraph(name='cluster_dual_vae') as model:
    model.attr(label='DualVAE Model', **plate_attrs)

    # RNA VAE
    model.node('rna_vae', 'RNA VAE\n(Latent Encoding)', **model_attrs)
    model.node('rna_decoded', 'RNA Decoded', **process_attrs)

    # Protein VAE
    model.node('protein_vae', 'Protein VAE\n(Latent Encoding)', **model_attrs)
    model.node('protein_decoded', 'Protein Decoded', **process_attrs)

    # Connections within VAEs
    model.edge('rna_vae', 'rna_decoded')
    model.edge('protein_vae', 'protein_decoded')

# Inputs
dot.node('rna_input', 'RNA Input', **data_attrs)
dot.node('protein_input', 'Protein Input', **data_attrs)
dot.node('cn_labels', 'CN Labels', **data_attrs)

# Inputs to VAEs
dot.edge('rna_input', 'rna_vae')
dot.edge('protein_input', 'protein_vae')
dot.edge('cn_labels', 'cn_loss',color='blue')

# Archetype Embedding Nodes
dot.node('rna_archetype_embedding', 'RNA + Protein\nArchetype Embedding', **embedding_attrs)

# Loss Terms Plate
with dot.subgraph(name='cluster_loss_terms') as loss_terms:
    loss_terms.attr(label='Loss Terms', **plate_attrs)

    # Matching Loss (KL Divergence)
    loss_terms.node('kl_divergence', 'Matching Cells Loss', **process_attrs)

    # CN Loss
    loss_terms.node('cn_loss', 'CN Loss', **process_attrs)

    # Reconstruction Loss
    loss_terms.node('reconstruction_loss', 'Reconstruction Loss', **process_attrs)

    # Total Loss
    loss_terms.node('total_loss', 'Total Loss\n(Matching + CN + Reconstruction)', **process_attrs)

    # Edges within Loss Plate
    loss_terms.edge('kl_divergence', 'total_loss')
    loss_terms.edge('cn_loss', 'total_loss')
    loss_terms.edge('reconstruction_loss', 'total_loss')

# Connections to Loss Plate
dot.edge('rna_vae', 'kl_divergence')
dot.edge('protein_vae', 'kl_divergence')
dot.edge('rna_archetype_embedding', 'kl_divergence',color='red')
dot.edge('rna_vae', 'cn_loss')
dot.edge('protein_vae', 'cn_loss')
dot.edge('rna_decoded', 'reconstruction_loss')
dot.edge('protein_decoded', 'reconstruction_loss')

# Render the graph
dot.render('dual_vae_training_pipeline', view=True)
