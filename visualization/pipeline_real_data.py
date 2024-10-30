from graphviz import Digraph

# Initialize Digraph with increased spacing
dot = Digraph(format='png')
dot.attr(rankdir='TB', size='10', nodesep='1.2', ranksep='0.1')  # Increase nodesep and ranksep
dot.attr(dpi='300')  # Set high DPI for better resolution

# Define node styles
process_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
data_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgrey'}
embedding_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'lightyellow'}
embedding_attrs_2 = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'yellow'}
model_attrs = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': 'lightgreen'}

# Data Generation Step
dot.node('rna_data', 'scRNA Data', **data_attrs)
dot.node('protein_data', 'CODEX Protein Data', **data_attrs)
dot.node('cn_data', 'CN (Cell Neighborhoods)', **data_attrs)

# MaxFuse Co-embedding Step
dot.node('maxfuse', 'Train MaxFuse Co-embedding', **process_attrs)
dot.node('co_embedding', 'Protein Co-embedding Space', **embedding_attrs)
dot.node('co_embedding_2', 'RNA Co-embedding Space', **embedding_attrs)

# Arrows for MaxFuse Step with minimum lengths and label distance
dot.edge('rna_data', 'maxfuse', label='',  minlen='2')
# dot.edge('maxfuse', 'vae_contrastive_2', label=' RNA Co-Embeddings\nOutputs', minlen='2', labeldistance='2')
dot.edge('maxfuse', 'co_embedding_2', label=' RNA Co-Embeddings\nOutputs', minlen='2', labeldistance='2')
dot.edge('co_embedding_2', 'vae_contrastive_2', label=' RNA Co-Embeddings\nOutputs', minlen='2', labeldistance='2')
dot.edge('protein_data', 'maxfuse', label='',  minlen='2')
dot.edge('protein_data', 'cn_data', label=' Calculate CN', minlen='2', labeldistance='2')
dot.edge('maxfuse', 'co_embedding', label=' Protein Co-Embeddings Outputs', minlen='2', labeldistance='2')

# VAE with Integrated Contrastive Learning
dot.node('vae_contrastive', 'VAE with Integrated\nContrastive Learning', **model_attrs)
dot.node('vae_contrastive_2', 'VAE with Integrated\nContrastive Learning', **model_attrs)
dot.node('latent_space', 'CN Aware RNA Latent Space -\nCan Be Used For Classification tasks', **embedding_attrs_2)


# Plate for Latent Space and Contrastive Learning
with dot.subgraph(name='cluster_plate') as plate:
    plate.attr(label='Train VAE with \nContrastive Learning', color='black', style='dashed')
    # plate.node('contrastive', 'Contrastive Loss\non CN and Cell Type', **process_attrs)

    # Edges within the Plate with added minlen and labeldistance for clarity
    plate.edge('co_embedding', 'vae_contrastive', label='Input Co-embedding', minlen='2', labeldistance='2')
    plate.edge('vae_contrastive', 'latent_space_2', label='Encode To', minlen='2', labeldistance='2')
    # Invisible node for plate connection and forcing bottom placement
    plate.node('plate_rep', '', shape='point')
    plate.node('latent_space_2', 'Proteomic CN Aware Latent Space', **embedding_attrs_2)

dot.edge('vae_contrastive_2', 'latent_space', label='Encodes to', minlen='2', labeldistance='2')
# plate.edge('contrastive', 'latent_space', label='Refines', minlen='2', labeldistance='2')

# dot.edge('latent_space', 'cn_pred')  # Invisible edge to control rank

# Set rank for final node to be at the bottom
with dot.subgraph() as bottom:
    bottom.attr(rank='sink')
    # bottom.node('cn_pred', 'Final Model for CN Prediction\n(Input: RNA, Output: CN)', **model_attrs)
    dot.edge('plate_rep', 'vae_contrastive_2', label='Use Trained Encoder',  minlen='2')
    # dot.edge('cn_data', 'cn_pred', label=' Validate predictions\nwith ground truth CN labels', minlen='2', labeldistance='2')

# Additional edges for contrastive learning context
dot.edge('cn_data', 'vae_contrastive', label=' Train using CN\n as ground truth\n for contrastive learning', minlen='2', labeldistance='2')
# dot.edge('cn_data', 'latent_space', label=' Train using CN\nas ground truth', minlen='2', labeldistance='2')

# dot.edge('latent_space', 'contrastive', label=' Applies to Latent Space', minlen='2', labeldistance='2')

# Render the graph
dot.save(f'vae_contrastive_pipeline.{dot._format}')
dot.render('vae_contrastive_pipeline', view=True)
