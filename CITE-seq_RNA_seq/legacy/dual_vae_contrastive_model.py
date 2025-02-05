# based on
# import sleep
from torch.onnx.verification import verify

# import mudata as md
# import muon
# ignore warnings
from bar_nick_utils import plot_latent, compute_pairwise_kl, verify_gradients
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy.sparse import issparse
from scvi.model import SCVI
from scvi.train import TrainingPlan
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
plot_flag = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)
np.random.seed(325)
plot_flag = True
save_dir = '../data/totalVI'

file_path = "/CITE-seq_RNA_seq/adata_rna_subset.h5ad"

adata_rna_subset = sc.read(
    '/CITE-seq_RNA_seq/adata_rna_subset.h5ad')
adata_prot_subset = sc.read(
    '/CITE-seq_RNA_seq/adata_prot_subset.h5ad')


#


# aa = rna_vae_.get_latent_representation()
# bb = protein_vae_.get_latent_representation()

# pca.fit(aa)
# aa_pca = pca.transform(aa)
# plt.subplot(1, 2, 1)
# plt.scatter(aa_pca[:, 0], aa_pca[:, 1], c=rna_vae_.adata.obs['CN'])
# pca.fit(bb)
# bb_pca = pca.transform(bb)
# plt.title('before training, RNA')
# plt.subplot(1, 2, 2)
# plt.scatter(bb_pca[:, 0], bb_pca[:, 1], c=protein_vae_.adata.obs['CN'])
# plt.title('during training, protein')
# plt.show()
# rna_vae_.module.to(device)
# protein_vae_.module.to(device)


class DualVAETrainingPlan(TrainingPlan):
    def __init__(self, rna_module, **kwargs):
        protein_vae = kwargs.pop('protein_vae')
        rna_vae = kwargs.pop('rna_vae')
        contrastive_weight = kwargs.pop('contrastive_weight', 1.0)
        super().__init__(rna_vae.module, **kwargs)
        super().__init__(protein_vae.module, **kwargs)
        # protein_vae.is_trained = True
        # rna_vae.is_trained = True

        self.rna_vae = rna_vae
        self.protein_vae = protein_vae
        self.prev = None
        self.contrastive_weight = contrastive_weight
        self.protein_vae.module.to(device)
        self.rna_vae.module = self.rna_vae.module.to(device)
        self.first_step = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.rna_vae.module.parameters()) + list(self.protein_vae.module.parameters()),
            lr=0.001,
            weight_decay=1e-5,
        )
        return optimizer

    def validation_step(self, batch, batch_idx):
        pass

    #     """Validation step for the model."""
    #     # RNA VAE forward pass
    #     _, _, rna_loss_output = self.forward(batch)
    #
    #     # Protein VAE forward pass
    #     protein_batch = self._get_protein_batch(batch)
    #     _, _, protein_loss_output = self.protein_vae.module.forward(protein_batch)
    #
    #     # Inference outputs
    #     rna_inference_outputs = self.rna_vae.module.inference(
    #         batch["X"], batch_index=batch["batch"], n_samples=1
    #     )
    #     protein_inference_outputs = self.protein_vae.module.inference(
    #         protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
    #     )
    #
    #     # Compute latent distances
    #     matching_rna_protein_latent_distances = torch.distributions.kl_divergence(
    #         rna_inference_outputs["qz"], protein_inference_outputs["qz"]
    #     )
    #
    #     # Compute pairwise distances for RNA and protein
    #     rna_distances = compute_pairwise_kl(
    #         rna_inference_outputs["qz"].mean, rna_inference_outputs["qz"].scale
    #     )
    #     prot_distances = compute_pairwise_kl(
    #         protein_inference_outputs["qz"].mean, protein_inference_outputs["qz"].scale
    #     )
    #
    #     # Compute contrastive loss
    #     cell_neighborhood_info = torch.tensor(
    #         self.rna_vae.adata[batch["labels"].cpu().numpy().squeeze()].obs["CN"].values).to(device)
    #     # index = batch["labels"].detach().cpu().numpy().squeeze()
    #     #
    #     # cell_neighborhood_info = torch.tensor(self.rna_vae.adata[index].obs["CN"].values).to(device)
    #
    #     major_cell_type = torch.tensor(
    #         self.rna_vae.adata[batch["labels"].cpu().numpy().squeeze()].obs["major_cell_types"].values.codes).to(
    #         device).squeeze()
    #
    #     num_cells = cell_neighborhood_info.shape[0]
    #     diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=cell_neighborhood_info.device)
    #
    #     same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
    #     same_major_cell_type = major_cell_type.unsqueeze(0) == major_cell_type.unsqueeze(1)
    #
    #     distances = prot_distances  # or combine with RNA distances if necessary
    #     distances = distances.masked_fill(diagonal_mask, 0)
    #
    #     same_major_type_same_cn_loss = (distances ** 2) * (same_major_cell_type * same_cn_mask).type(torch.bool)
    #     same_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
    #                 same_major_cell_type * ~same_cn_mask).type(torch.bool)
    #     different_major_type_same_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
    #                 ~same_major_cell_type * same_cn_mask).type(torch.bool)
    #     different_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
    #                 ~same_major_cell_type * ~same_cn_mask).type(torch.bool)
    #
    #     cn_loss = (
    #                       same_major_type_same_cn_loss.sum() +
    #                       same_major_type_different_cn_loss.sum() +
    #                       different_major_type_same_cn_loss.sum() +
    #                       different_major_type_different_cn_loss.sum()
    #               ) / (num_cells * (num_cells - 1))
    #
    #     # Total loss
    #     total_loss = (
    #             0.1 * self.contrastive_weight * cn_loss +
    #             0.1 * rna_loss_output.loss +
    #             0.1 * protein_loss_output.loss +
    #             10 * matching_rna_protein_latent_distances.mean()
    #     )
    #
    #     # Log metrics
    #     self.log(
    #         "validation_rna_loss", rna_loss_output.loss, on_epoch=True, sync_dist=self.use_sync_dist
    #     )
    #     self.log(
    #         "validation_protein_loss", protein_loss_output.loss, on_epoch=True, sync_dist=self.use_sync_dist
    #     )
    #     self.log(
    #         "validation_contrastive_loss", cn_loss, on_epoch=True, sync_dist=self.use_sync_dist
    #     )
    #     self.log(
    #         "validation_total_loss", total_loss, on_epoch=True, sync_dist=self.use_sync_dist
    #     )
    #     self.log(
    #         "validation_matching_latent_distances", matching_rna_protein_latent_distances.mean(), on_epoch=True,
    #         sync_dist=self.use_sync_dist
    #     )
    #
    #     # Compute and log metrics
    #     self.compute_and_log_metrics(rna_loss_output, self.val_metrics, "validation")
    #     self.compute_and_log_metrics(protein_loss_output, self.val_metrics, "validation")
    #     return total_loss

    def training_step(self, batch, batch_idx):
        rna_batch = self._get_rna_batch(batch)
        kl_weight = 2  # maybe make sure this is proper
        self.loss_kwargs.update({"kl_weight": kl_weight})
        self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, rna_loss_output = self.rna_vae.module(rna_batch, loss_kwargs=self.loss_kwargs)
        protein_batch = self._get_protein_batch(batch)
        _, _, protein_loss_output = self.protein_vae.module(protein_batch, loss_kwargs=self.loss_kwargs)
        rna_inference_outputs = self.rna_vae.module.inference(
            rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        )
        index = rna_batch["labels"]
        assert len(set(self.rna_vae.adata[index].obs['CN'].values)) != 1

        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )
        # here we assume that the cells have been aligned in the same order to their best match across modalities
        matching_rna_protein_latent_distances = torch.distributions.kl_divergence(rna_inference_outputs["qz"],
                                                                                  protein_inference_outputs["qz"])
        rna_distances = compute_pairwise_kl(rna_inference_outputs["qz"].mean,
                                            rna_inference_outputs["qz"].scale)
        prot_distances = compute_pairwise_kl(protein_inference_outputs["qz"].mean,
                                             protein_inference_outputs["qz"].scale)

        if self.first_step:  # plot histogram of distances prot and rna on top of each other after each epoch
            plt.hist(rna_inference_outputs["qz"].loc.detach().cpu().numpy().flatten(), bins=100, alpha=0.5)
            plt.hist(protein_inference_outputs["qz"].loc.detach().cpu().numpy().flatten(), bins=100, alpha=0.5)
            plt.hist(matching_rna_protein_latent_distances.detach().cpu().numpy().flatten(), bins=100, alpha=0.5)
            plt.show()
            plt.hist(prot_distances.detach().cpu().numpy().flatten(), bins=100, alpha=0.5)
            plt.hist(rna_distances.detach().cpu().numpy().flatten(), bins=100, alpha=0.5)
            plt.show()
            self.first_step = False
        distances = prot_distances + rna_distances
        if self.global_step>10 and self.global_step % 100 == 0:
            verify_gradients(self.rna_vae.module,self.protein_vae)

            sns.histplot(prot_distances.detach().cpu().flatten(), label='prot')
            sns.histplot(rna_distances.detach().cpu().flatten(), label='rna')
            plt.legend()
            plt.show()
            plot_latent(rna_inference_outputs["qz"].mean.clone().detach().cpu().numpy(),
                        protein_inference_outputs["qz"].mean.clone().detach().cpu().numpy(),
                        self.rna_vae.adata, self.protein_vae.adata, index=rna_batch["labels"])
            plt.show()
        if self.prev is None:
            self.prev = torch.sum(prot_distances)
        else:
            if self.prev == torch.sum(prot_distances):
                raise ValueError('The model is not training')
        cell_neighborhood_info = torch.tensor(self.rna_vae.adata[index].obs["CN"].values).to(device)
        major_cell_type = torch.tensor(self.rna_vae.adata[index].obs["major_cell_types"].values.codes).to(
            device).squeeze()

        num_cells = cell_neighborhood_info.shape[0]
        diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=cell_neighborhood_info.device)
        # this will give us each row represents a item in the array, and each col is whether it is the same as the items in that index of the col
        # this way we get for each cell(a row) which other cells (index of each item in the row, which is the col) are matching
        # so if we got 1,2,1, we will get [[1,0,1],[0,1,0],[1,0,1]]
        same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
        same_major_cell_type = major_cell_type.unsqueeze(0) == major_cell_type.unsqueeze(1)
        if self.first_step:  # show the mask only for the first batch to make sure it is working as expected
            pass
            # self.first_step = False
            # plt.imshow(same_cn_mask.cpu().numpy())
            # 
            # plt.figure()
            # plt.subplot(2,1,1)
            # plot_torch_normal(rna_inference_outputs["qz"].mean[0][0].item(),rna_inference_outputs["qz"].scale[0][0].item())
            # plot_torch_normal(protein_inference_outputs["qz"].mean[0][0].item(),protein_inference_outputs["qz"].scale[0][0].item())
            # plt.title(f'kl_div score is {kl_div_vec[0][0].item()}')
            # plt.show()
            # # plot the second item in the batch
            # plt.subplot(2,1,2)
            # plot_torch_normal(rna_inference_outputs["qz"].mean[1][0].item(),rna_inference_outputs["qz"].scale[1][0].item())
            # plot_torch_normal(protein_inference_outputs["qz"].mean[1][0].item(),protein_inference_outputs["qz"].scale[1][0].item())
            # plt.title(f'kl_div score is {kl_div_vec[1][0].item()}')
            # plt.show()

        distances = distances.masked_fill(diagonal_mask, 0)

        same_major_type_same_cn_mask = (same_major_cell_type * same_cn_mask).type(torch.bool)
        same_major_type_different_cn_mask = (same_major_cell_type * ~same_cn_mask).type(torch.bool)
        different_major_type_same_cn_mask = (~same_major_cell_type * same_cn_mask).type(torch.bool)
        different_major_type_different_cn_mask = (~same_major_cell_type * ~same_cn_mask).type(torch.bool)

        same_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
        same_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)
        different_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
        different_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)

        same_major_type_same_cn_loss = (distances ** 2) * same_major_type_same_cn_mask
        same_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * same_major_type_different_cn_mask
        different_major_type_same_cn_loss = ((10 - distances).clamp(min=0) ** 2) * different_major_type_same_cn_mask
        different_major_type_different_cn_loss = ((10 - distances).clamp(
            min=0) ** 2) * different_major_type_different_cn_mask
        # for debugging only: # todo remove this same_cn_loss, it is not valid
        same_cn_loss = (distances ** 2) * same_cn_mask
        same_major_type_loss = (distances ** 2) * same_major_cell_type
        # end of debugging
        positive_loss = same_major_type_same_cn_loss

        negative_loss = different_major_type_different_cn_loss + different_major_type_same_cn_loss + 2 * same_major_type_different_cn_loss
        cn_loss = (positive_loss.sum() + negative_loss.sum()) / (num_cells * (num_cells - 1))

        matching_loss = 10000 * matching_rna_protein_latent_distances.mean()
        reconstruction_loss = rna_loss_output.loss * 0.1 + protein_loss_output.loss * 0.1
        contrastive_loss = 1000 * self.contrastive_weight * cn_loss
        total_loss = (
                reconstruction_loss
                + contrastive_loss
                + matching_loss
        )
        # Log losses
        self.log("train_rna_reconstruction_loss", rna_loss_output.loss, on_epoch=True, on_step=False)
        self.log("train_protein_reconstruction_loss", protein_loss_output.loss, on_epoch=True, on_step=False)
        self.log("train_contrastive_loss", contrastive_loss, on_epoch=True, on_step=False)
        self.log("train_matching_rna_protein_loss", matching_loss, on_epoch=True, on_step=False)
        self.log("train_total_loss", total_loss, on_epoch=True, on_step=False)
        if self.current_epoch % 50 == 0:
            rna_vae.save(save_dir, prefix=f'batch_{batch_idx}_', save_anndata=False, overwrite=True)
        return total_loss

    def _get_protein_batch(self, batch):
        indices = batch['labels'].detach().cpu().numpy().flatten()  # Assuming batch contains indices
        protein_data = self.protein_vae.adata[indices]
        protein_batch = {
            'X': torch.tensor(protein_data.X.A if issparse(protein_data.X) else protein_data.X).to(device),
            'batch': torch.tensor(protein_data.obs['_scvi_batch'].values, dtype=torch.long).to(device),
            'labels': indices,
        }
        return protein_batch

    def _get_rna_batch(self, batch):
        indices = batch['labels'].detach().cpu().numpy().flatten()
        rna_data = self.rna_vae.adata[indices]
        rna_batch = {
            'X': torch.tensor(rna_data.X.A if issparse(rna_data.X) else rna_data.X).to(device),
            'batch': torch.tensor(rna_data.obs['_scvi_batch'].values, dtype=torch.long).to(device),
            'labels': indices,
        }
        return rna_batch


# adata_rna_subset = adata_rna_subset.copy()
# adata_prot_subset = adata_prot_subset.copy()
# adata_prot_subset = adata_prot_subset[np.random.permutation(adata_prot_subset.n_obs)].copy()
# adata_rna_subset = adata_rna_subset[np.random.permutation(adata_rna_subset.n_obs)].copy()
SCVI.setup_anndata(
    adata_rna_subset,
    labels_key="index_col",
)

SCVI.setup_anndata(
    adata_prot_subset,
    labels_key="index_col",
)

# Initialize VAEs
rna_vae = scvi.model.SCVI(adata_rna_subset, gene_likelihood="nb", n_hidden=128, n_layers=1)
protein_vae = scvi.model.SCVI(adata_prot_subset, gene_likelihood="nb", n_hidden=50, n_layers=1)
initial_weights = {name: param.clone() for name, param in rna_vae.module.named_parameters()}

rna_vae._training_plan_cls = DualVAETrainingPlan
protein_vae._training_plan_cls = DualVAETrainingPlan
protein_vae.module.to('cpu')
rna_vae.module.to('cpu')
rna_vae.is_trained ,protein_vae.is_trained =True,True
latent_rna_before = rna_vae.get_latent_representation().copy()
latent_prot_before = protein_vae.get_latent_representation().copy()
rna_vae.is_trained,protein_vae.is_trained  = False, False

rna_vae.train(  # for debug only!!! mess up the real training
    check_val_every_n_epoch=99,
    max_epochs=2,
    early_stopping=False,
    early_stopping_patience=70,
    early_stopping_monitor="train_total_loss",
    batch_size=256,
    shuffle_set_split=True,
    plan_kwargs={'protein_vae': protein_vae,
                 'rna_vae': rna_vae,
                 'contrastive_weight': 10.0,
                 }
)

# Save final weights
final_weights = {name: param.clone() for name, param in rna_vae.module.named_parameters()}
# Compare weights
for name in initial_weights:
    if torch.equal(initial_weights[name].detach().cpu(), final_weights[name].detach().cpu()):
        print(f"Parameter {name} did not change.")
    else:
        print(f"Parameter {name} was updated.")

# %% mdtraining_plan
# %%
fig, ax = plt.subplots(1, 1)
for key in rna_vae.history.keys():
    if 'loss' in key:
        norm_loss = (rna_vae.history[key] - rna_vae.history[key].min()) / (rna_vae.history[key].max() - rna_vae.history[key].min())
        label = f'{key} min: {rna_vae.history[key].values.min():.0f} max: {rna_vae.history[key].values.max():.0f}'
        ax.plot(norm_loss, label=label)
ax.legend()
plt.show()


SCVI_LATENT_KEY = "X_scVI"
# assert protein_vae == rna_vae.training_plan.protein_vae
protein_vae.module.to('cpu')
rna_vae.module.to('cpu')
protein_vae.is_trained = True
rna_vae.is_trained = True

# latent_rna = rna_vae.get_latent_representation()
# latent_prot = protein_vae.get_latent_representation()
# adata_rna_subset.obsm[SCVI_LATENT_KEY] = latent_rna
# adata_prot_subset.obsm[SCVI_LATENT_KEY] = latent_prot

# sc.pl.pca(adata_prot_subset, color=["CN", "cell_types"],title=['Latent space, CN Protein', 'Latent space, minor cell types Protein'])
# sc.pp.neighbors(adata_prot_subset, use_rep=basis)
# sc.tl.umap(adata_prot_subset)
# sc.pl.umap(adata_prot_subset, color=["CN", "cell_types"], title=['Latent space, CN Protein', 'Latent space, minor cell types Protein'])
# pca on aa and plot

pca.fit(latent_rna_before)
latent_rna_pca = pca.transform(latent_rna_before)
plt.subplot(1, 2, 1)
plt.scatter(latent_rna_pca[:, 0], latent_rna_pca[:, 1], c=adata_rna_subset.obs['CN'])
pca.fit(latent_prot_before)
latent_prot_pca = pca.transform(latent_prot_before)
plt.title('RNA ')
plt.subplot(1, 2, 2)
plt.scatter(latent_prot_pca[:, 0], latent_prot_pca[:, 1], c=adata_prot_subset.obs['CN'])
plt.title('proteins Before training')
plt.show()

# # plot pca original data
pca.fit(adata_rna_subset.X)
rna_pca = pca.transform(adata_rna_subset.X)
plt.subplot(1, 2, 1)
plt.scatter(rna_pca[:, 0], rna_pca[:, 1], c=adata_rna_subset.obs['CN'])
pca.fit(adata_prot_subset.X)
prot_pca = pca.transform(adata_prot_subset.X)
plt.subplot(1, 2, 2)
plt.scatter(prot_pca[:, 0], prot_pca[:, 1], c=adata_prot_subset.obs['CN'])
plt.title('Original data')
plt.show()

rna_vae.module.to(device)
protein_vae.module.to(device)
rna_vae.module.eval()
protein_vae.module.eval()
rna = rna_vae.get_latent_representation()
protein = protein_vae.get_latent_representation()

rna_pca = pca.fit_transform(rna)
plt.subplot(1, 2, 1)
plt.scatter(rna_pca[:, 0], rna_pca[:, 1], c=adata_rna_subset.obs['CN'])
protein_pca = pca.fit_transform(protein)
plt.title('RNA after training')
plt.subplot(1, 2, 2)
plt.scatter(protein_pca[:, 0], protein_pca[:, 1], c=adata_prot_subset.obs['CN'])
plt.title('protein After training')
plt.show()
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
rna_tsne = tsne.fit_transform(rna)
plt.subplot(1, 2, 1)
plt.scatter(rna_tsne[:, 0], rna_tsne[:, 1], c=adata_rna_subset.obs['CN'])
protein_tsne = tsne.fit_transform(protein)
plt.title('RNA after training')
plt.subplot(1, 2, 2)
plt.scatter(protein_tsne[:, 0], protein_tsne[:, 1], c=adata_prot_subset.obs['CN'])
plt.title('protein After training')
plt.show()
