from graphviz import Digraph

# Initialize Digraph with increased spacing
dot = Digraph(format="png")
dot.attr(rankdir="TB", size="10", nodesep="1.2", ranksep="0.1")  # Increase spacing
dot.attr(dpi="300")  # Set high DPI for better resolution

# Define node styles
data_attrs = {"shape": "ellipse", "style": "filled", "fillcolor": "lightgrey"}
process_attrs = {"shape": "box", "style": "filled", "fillcolor": "lightblue"}
alignment_attrs = {"shape": "hexagon", "style": "filled", "fillcolor": "lightgreen"}
embedding_attrs = {"shape": "parallelogram", "style": "filled", "fillcolor": "lightyellow"}
final_output_attrs = {
    "shape": "oval",
    "style": "filled",
    "fillcolor": "lightcoral",
}  # Final output node style

# Data Nodes
dot.node("rna_data", "scRNA Data", **data_attrs)
dot.node("protein_data", "scProtein Data", **data_attrs)

# Archetype Vector Bases (Non-Aligned)
dot.node("rna_non_aligned_archetypes", "Non-Aligned RNA\nArchetype Basis", **process_attrs)
dot.node("protein_non_aligned_archetypes", "Non-Aligned Protein\nArchetype Basis", **process_attrs)
dot.node(
    "non_aligned_rna_embedding", "Non-Aligned RNA Cell\nArchetype Embeddings", **embedding_attrs
)
dot.node(
    "non_aligned_protein_embedding",
    "Non-Aligned Protein Cell\nArchetype Embeddings",
    **embedding_attrs
)
dot.edge("rna_non_aligned_archetypes", "non_aligned_rna_embedding")
dot.edge("protein_non_aligned_archetypes", "non_aligned_protein_embedding")
dot.edge(
    "non_aligned_rna_embedding",
    "align_non_aligned_basis",
    label="NNLS Archetype Linear Combination",
)
dot.edge("non_aligned_protein_embedding", "align_non_aligned_basis")

# Edges from data to non-aligned vector basis
dot.edge("rna_data", "rna_non_aligned_archetypes", label="Extract Polygon Vertices (PCHA)")
dot.edge("protein_data", "protein_non_aligned_archetypes", label="Extract Polygon Vertices (PCHA)")
dot.node("major_cell_types", "Major Cell Types Ground Truth Labels\nRNA+Protein", **data_attrs)
dot.edge("major_cell_types", "align_non_aligned_basis")

# Alignment Node for Non-Aligned Basis
dot.node(
    "align_non_aligned_basis",
    "Align Non-Aligned Vector Bases\nfor Comparable Representation",
    **alignment_attrs
)

# RNA and Protein Archetype Embeddings
# dot.node('aligned_rna_embedding', 'Aligned RNA Cell\nArchetype Embeddings', **embedding_attrs)
# dot.node('aligned_protein_embedding', 'Aligned Protein Cell\nArchetype Embeddings', **embedding_attrs)

# Edges from alignment to aligned basis
# dot.edge('align_non_aligned_basis', 'aligned_rna_embedding', label='NNLS Archetype Linear Combination')
# dot.edge('align_non_aligned_basis', 'aligned_protein_embedding')

# Final output node: Match similar cells across modalities
dot.node("matched_cells", "Modality-Agnostic\nCell Matching", **final_output_attrs)
dot.edge("align_non_aligned_basis", "matched_cells", label="Aligned Archetype Embeddings")
# dot.edge('aligned_protein_embedding', 'matched_cells')

# Render the graph
dot.render("archetype_matching_pipeline", view=True)

# Additional node label suggestions in comments:
# 1. "Integrated Cell Matching Across Modalities"
# 2. "Cross-Modality Cell Alignment"
# 3. "Unified Cell Comparison Across Modalities"
# 4. "Modality-Agnostic Cell Matching"
