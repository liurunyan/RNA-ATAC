# %%
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse 
import mudata as md
import os

os.chdir("/data/home/scxl504/run/GAT/sc")
dataset = "LUNG-CITE"


# %% [markdown]
# ### Load data

# %%

rna_adata = sc.read_h5ad(os.path.join("data", "raw", dataset, f"{dataset}_RNA.h5ad"))
# peaks_adata = sc.read_h5ad(os.path.join("data", "raw", dataset, f"{dataset}_Peaks.h5ad"))
adt_adata = sc.read_h5ad(os.path.join("data", "raw", dataset, f"{dataset}_ADT.h5ad"))
latent = np.load(
    os.path.join("results", "embedding", f"{dataset}_latent_embedding.npy")
)


# %% [markdown]
# ### Create mudata

# %%
latent_adata = sc.AnnData(
    X=latent,
    obs=rna_adata.obs.copy(),
)

latent_adata.obs_names = rna_adata.obs_names.copy()
latent_adata.var_names = [
    f"latent_{i}" for i in range(latent_adata.n_vars)
]

# create MuData
mdata = md.MuData(
    {
        "rna": rna_adata,
        # "peaks": peaks_adata,
        "adt": adt_adata,
        "latent": latent_adata,
    }
)

# mdata.update()
print(mdata)

# %% [markdown]
# ### Visualization

# %%
# RNA 
# RNA PCA
sc.pp.pca(
    mdata["rna"],
    n_comps=50
)

# RNA neighbors
sc.pp.neighbors(
    mdata["rna"],
    n_neighbors=15,
    metric="cosine",
    use_rep="X_pca"
)

# RNA UMAP
sc.tl.umap(
    mdata["rna"],
    random_state=42
)

# # Plot
# sc.pl.umap(
#     mdata["rna"],
#     color="celltype",
#     frameon=False,
#     title=f"{dataset} RNA",
# )

fig = sc.pl.umap(
    mdata["rna"],
    color="celltype",
    frameon=False,
    title=f"{dataset} RNA",
    return_fig=True,
)
if isinstance(fig, tuple):
    fig = fig[0]
out_dir = os.path.join("results", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{dataset}_RNA_umap.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# %%
# Peaks 
# sc.pp.pca(
#     mdata["peaks"],
#     n_comps=50
# )

# sc.pp.neighbors(
#     mdata["peaks"],
#     metric="cosine",
#     n_neighbors=15,
#     use_rep="X_pca"
# )

# sc.tl.umap(
#     mdata["peaks"],
#     random_state=42
# )

# # sc.pl.umap(
# #     mdata["peaks"],
# #     color="celltype",
# #     frameon=False,
# #     title=f"{dataset} Peaks",
# # )
# fig = sc.pl.umap(
#     mdata["peaks"],
#     color="celltype",
#     frameon=False,
#     title=f"{dataset} Peaks",
#     return_fig=True,
# )
# if isinstance(fig, tuple):
#     fig = fig[0]
# out_dir = os.path.join("results", "figures")
# os.makedirs(out_dir, exist_ok=True)
# out_path = os.path.join(out_dir, f"{dataset}_Peaks_umap.png")
# fig.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.close(fig)

sc.pp.pca(
    mdata["adt"],
    n_comps=50
)

sc.pp.neighbors(
    mdata["adt"],
    metric="cosine",
    n_neighbors=15,
    use_rep="X_pca"
)

sc.tl.umap(
    mdata["adt"],
    random_state=42
)

# sc.pl.umap(
#     mdata["peaks"],
#     color="celltype",
#     frameon=False,
#     title=f"{dataset} Peaks",
# )
fig = sc.pl.umap(
    mdata["adt"],
    color="celltype",
    frameon=False,
    title=f"{dataset} ADT",
    return_fig=True,
)
if isinstance(fig, tuple):
    fig = fig[0]
out_dir = os.path.join("results", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{dataset}_ADT_umap.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# %%
# compute UMAP for latent embedding
# sc.pp.neighbors(mdata["latent"], metric="cosine", n_neighbors=15) 
# # metric can be changed to "euclidean" or "cosine"
# sc.tl.umap(mdata["latent"], random_state=42)
# # visualize UMAP for latent embedding
# sc.pl.umap(
#     mdata["latent"],
#     color="celltype",
#     frameon=False,
#     title="Latent",
# )


