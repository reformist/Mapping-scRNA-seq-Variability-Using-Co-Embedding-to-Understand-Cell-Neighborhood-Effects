# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: archetypegen
#     language: python
#     name: python3
# ---

# %%
# !pip install zadu

# %%
from zadu import zadu

# %%
from zadu import zadu

hd, ld = load_datasets()
spec = [{
    "id"    : "tnc",
    "params": { "k": 20 },
}, {
    "id"    : "snc",
    "params": { "k": 30, "clustering_strategy": "dbscan" }
}]

scores = zadu.ZADU(spec, hd).measure(ld)
print("T&C:", scores[0])
print("S&C:", scores[1])

# %% [markdown]
# #### explanation of ZADU: used to evaluate dimensionality reduction embeddings. we hope this will be useful because we use UMAP to understand mixing between RNA and Protein modalities along with other things  

# %%
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
hd = fetch_openml('mnist_784', version=1, cache=True).data.to_numpy()[::7]
ld = TSNE().fit_transform(hd)

# %%
# directly accessing functions
from zadu.measures import *

mrre = mean_relative_rank_error.measure(hd, ld, k=20)
pr  = pearson_r.measure(hd, ld)
nh  = neighborhood_hit.measure(ld, label, k=20)

# %%
mrre

# %%
pr

# %%
from zadu import zadu
from zaduvis import zaduvis
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml


hd = fetch_openml('mnist_784', version=1, cache=True).data.to_numpy()[::7]
ld = TSNE().fit_transform(hd)

## Computing local pointwise distortions
spec = [{
    "id": "tnc",
    "params": {"k": 25}
},{
    "id": "snc",
    "params": {"k": 50}
}]
zadu_obj = zadu.ZADU(spec, hd, return_local=True)


# %%
scores, local_list = zadu_obj.measure(ld)

tnc_local = local_list[0]
snc_local = local_list[1]

local_trustworthiness = tnc_local["local_trustworthiness"]
local_continuity = tnc_local["local_continuity"]
local_steadiness = snc_local["local_steadiness"]
local_cohesiveness = snc_local["local_cohesiveness"]

fig, ax = plt.subplots(1, 4, figsize=(50, 12.5))
zaduvis.checkviz(ld, local_trustworthiness, local_continuity, ax=ax[0])
zaduvis.reliability_map(ld, local_trustworthiness, local_continuity, k=10, ax=ax[1])
zaduvis.checkviz(ld, local_steadiness, local_cohesiveness, ax=ax[2])
zaduvis.reliability_map(ld, local_steadiness, local_cohesiveness, k=10, ax=ax[3])
