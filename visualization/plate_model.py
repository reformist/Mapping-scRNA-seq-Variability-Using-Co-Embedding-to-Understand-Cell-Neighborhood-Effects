from graphviz import Digraph

g = Digraph("DualVAE_PlateModel", format="png")
g.attr("graph", rankdir="LR")

# Global parameters outside the plate
g.node("thetaRNA", label="θ_RNA", shape="box", style="filled,rounded", fillcolor="#f5f5f5")
g.node("thetaProt", label="θ_Protein", shape="box", style="filled,rounded", fillcolor="#f5f5f5")

# Priors on z: Normal(0, I)
g.node("zRNA_dist", label="z_RNA ~ N(0,I)", shape="ellipse", style="dashed")
g.node("zProt_dist", label="z_Protein ~ N(0,I)", shape="ellipse", style="dashed")

# Plate for N cells
with g.subgraph(name="cluster_cells") as c:
    c.attr(label="N cells")
    c.node("zRNA_i", label="z_RNA_i", shape="circle")
    c.node("zProt_i", label="z_Protein_i", shape="circle")
    c.node("X_RNA_i", label="X^RNA_i", shape="box", style="rounded")
    c.node("X_Prot_i", label="X^Protein_i", shape="box", style="rounded")

    # Edges inside the plate
    # Latent to observed
    c.edge("zRNA_i", "X_RNA_i", label="p(X^RNA_i | z_RNA_i, θ_RNA)")
    c.edge("zProt_i", "X_Prot_i", label="p(X^Protein_i | z_Protein_i, θ_Protein)")

    # Coupling factor between z_RNA_i and z_Protein_i
    # This can represent the contrastive alignment or a joint prior/factor.
    c.node(
        "Factor_align",
        label="Contrastive\nAlignment",
        shape="diamond",
        style="rounded,filled",
        fillcolor="#ffd57f",
    )
    c.edge("zRNA_i", "Factor_align")
    c.edge("zProt_i", "Factor_align")

# Connect priors to latent variables (conceptually, showing that each z_i is drawn from the same prior)
g.edge("zRNA_dist", "zRNA_i")
g.edge("zProt_dist", "zProt_i")

# Connect global parameters to the observed nodes to indicate their role in the generative model
g.edge("thetaRNA", "X_RNA_i")
g.edge("thetaProt", "X_Prot_i")

g.render("DualVAE_PlateModel", view=True)
