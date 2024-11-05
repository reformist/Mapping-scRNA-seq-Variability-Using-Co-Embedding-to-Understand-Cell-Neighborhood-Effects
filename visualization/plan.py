from graphviz import Digraph

# Create a Digraph object
dot = Digraph(comment='Detailed VAE Architecture with Genes, Proteins, and CN')

# Set the graph layout direction (left to right)
dot.attr(rankdir='LR')

# Define graph attributes for better visualization
dot.attr('node', shape='rectangle', style='filled', color='lightblue')

# Input nodes
dot.node('genes_input', 'Genes')
dot.node('proteins_input', 'Proteins')
dot.node('cn_input', 'Cell Neighborhood (CN)')

# Encoder nodes (detailed layers)
dot.node('genes_encoder1', 'Genes Encoder\nLayer 1')
dot.node('genes_encoder2', 'Genes Encoder\nLayer 2')
dot.node('genes_mu', 'Genes μ')
dot.node('genes_sigma', 'Genes σ')

dot.node('proteins_encoder1', 'Proteins Encoder\nLayer 1')
dot.node('proteins_encoder2', 'Proteins Encoder\nLayer 2')
dot.node('proteins_mu', 'Proteins μ')
dot.node('proteins_sigma', 'Proteins σ')

dot.node('cn_encoder1', 'CN Encoder\nLayer 1')
dot.node('cn_encoder2', 'CN Encoder\nLayer 2')
dot.node('cn_mu', 'CN μ')
dot.node('cn_sigma', 'CN σ')

# Latent space nodes
dot.node('z_genes', 'z_genes')
dot.node('z_proteins', 'z_proteins')
dot.node('z_cn', 'z_cn')

# Concatenated latent vector
dot.node('z_concat', 'Concatenated\nLatent Vector z')

# Decoder nodes (detailed layers)
dot.node('decoder_genes1', 'Genes Decoder\nLayer 1')
dot.node('decoder_genes2', 'Genes Decoder\nLayer 2')

dot.node('decoder_proteins1', 'Proteins Decoder\nLayer 1')
dot.node('decoder_proteins2', 'Proteins Decoder\nLayer 2')

dot.node('decoder_cn1', 'CN Decoder\nLayer 1')
dot.node('decoder_cn2', 'CN Decoder\nLayer 2')

# Output nodes
dot.node('genes_output', 'Reconstructed\nGenes')
dot.node('proteins_output', 'Reconstructed\nProteins')
dot.node('cn_output', 'Reconstructed\nCN')

# Edges from inputs to encoders
dot.edge('genes_input', 'genes_encoder1')
dot.edge('genes_encoder1', 'genes_encoder2')
dot.edge('genes_encoder2', 'genes_mu')
dot.edge('genes_encoder2', 'genes_sigma')

dot.edge('proteins_input', 'proteins_encoder1')
dot.edge('proteins_encoder1', 'proteins_encoder2')
dot.edge('proteins_encoder2', 'proteins_mu')
dot.edge('proteins_encoder2', 'proteins_sigma')

dot.edge('cn_input', 'cn_encoder1')
dot.edge('cn_encoder1', 'cn_encoder2')
dot.edge('cn_encoder2', 'cn_mu')
dot.edge('cn_encoder2', 'cn_sigma')

# Sampling from latent space (reparameterization trick)
dot.edge('genes_mu', 'z_genes', label='μ')
dot.edge('genes_sigma', 'z_genes', label='σ')
dot.edge('proteins_mu', 'z_proteins', label='μ')
dot.edge('proteins_sigma', 'z_proteins', label='σ')
dot.edge('cn_mu', 'z_cn', label='μ')
dot.edge('cn_sigma', 'z_cn', label='σ')

# Edges from latent variables to concatenated latent vector
dot.edge('z_genes', 'z_concat')
dot.edge('z_proteins', 'z_concat')
dot.edge('z_cn', 'z_concat')

# Edges from concatenated latent vector to decoders
dot.edge('z_concat', 'decoder_genes1')
dot.edge('z_concat', 'decoder_proteins1')
dot.edge('z_concat', 'decoder_cn1')

# Decoder layers to outputs
dot.edge('decoder_genes1', 'decoder_genes2')
dot.edge('decoder_genes2', 'genes_output')

dot.edge('decoder_proteins1', 'decoder_proteins2')
dot.edge('decoder_proteins2', 'proteins_output')

dot.edge('decoder_cn1', 'decoder_cn2')
dot.edge('decoder_cn2', 'cn_output')

# Group encoders in a subgraph
with dot.subgraph(name='cluster_encoders') as c:
    c.attr(style='dashed', color='gray')
    c.node('genes_encoder1')
    c.node('genes_encoder2')
    c.node('genes_mu')
    c.node('genes_sigma')
    c.node('proteins_encoder1')
    c.node('proteins_encoder2')
    c.node('proteins_mu')
    c.node('proteins_sigma')
    c.node('cn_encoder1')
    c.node('cn_encoder2')
    c.node('cn_mu')
    c.node('cn_sigma')
    c.attr(label='Encoders')

# Group latent variables in a subgraph
with dot.subgraph(name='cluster_latent') as c:
    c.attr(style='dashed', color='gray')
    c.node('z_genes')
    c.node('z_proteins')
    c.node('z_cn')
    c.node('z_concat')
    c.attr(label='Latent Space')

# Group decoders in a subgraph
with dot.subgraph(name='cluster_decoders') as c:
    c.attr(style='dashed', color='gray')
    c.node('decoder_genes1')
    c.node('decoder_genes2')
    c.node('decoder_proteins1')
    c.node('decoder_proteins2')
    c.node('decoder_cn1')
    c.node('decoder_cn2')
    c.attr(label='Decoders')

# Group outputs in a subgraph
with dot.subgraph(name='cluster_outputs') as c:
    c.attr(style='dashed', color='gray')
    c.node('genes_output')
    c.node('proteins_output')
    c.node('cn_output')
    c.attr(label='Outputs')

# Render the graph to a file and view it
dot.render('vae_detailed_visualization.gv', view=True)


