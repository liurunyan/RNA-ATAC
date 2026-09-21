# Visualize a precomputed joint RNA+ATAC latent embedding (.npy) on a Seurat object.
# Replaces the separate SCTransform+PCA (RNA) and TF-IDF+LSI (ATAC) reduction
# steps with a single custom embedding, then runs UMAP on it for visualization.

library(Seurat)
library(Signac)
library(dplyr)
library(ggplot2)
library(RcppCNPy)  # for npyLoad(); alternative: reticulate::import("numpy")$load()

# %%
base_path <- "/data/run01/scxl504/GAT/sc"
initial_path <- file.path(base_path, "data", "raw")
file_source <- "PBMC-Multiome"
file_name <- "PBMC-Multiome.Rds"

# Path to the joint low-dimensional embedding produced by train_graph_ae.py
# (see results/embedding/<DATASET_NAME>_latent_embedding.npy in that pipeline).
embedding_path <- file.path(base_path, "results", "embedding", "PBMC-Multiome_latent_embedding.npy")

set.seed(42)  # fixed seed so the subsample below is reproducible run to run

# %%
file_path <- file.path(initial_path, file_source, file_name)
obj_input <- readRDS(file_path)
print(obj_input)


obj <- obj_input  # use all cells if the embedding was computed on the full dataset

# %%
# Load the precomputed joint embedding and align it with the Seurat object.
embedding <- npyLoad(embedding_path)
message("Embedding dims: ", nrow(embedding), " x ", ncol(embedding))

if (nrow(embedding) != ncol(obj)) {
  stop(
    "Row count mismatch: embedding has ", nrow(embedding), " rows but obj has ",
    ncol(obj), " cells. The embedding was likely generated from a different ",
    "cell subsample/order than this script's sample(). Re-generate the embedding ",
    "from this exact cell set, or save+join by barcode instead of relying on order."
  )
}


colnames(embedding) <- paste0("latent_", seq_len(ncol(embedding)))
rownames(embedding) <- colnames(obj)

obj[["joint"]] <- CreateDimReducObject(
  embeddings = embedding,
  key = "joint_",
  assay = DefaultAssay(obj)
)

obj <- FindNeighbors(
  obj,
  reduction = "joint",
  dims = 1:ncol(embedding),
  verbose = FALSE
)

obj <- RunUMAP(
  obj,
  reduction = "joint",
  dims = 1:ncol(embedding),
  reduction.name = "umap.joint",
  reduction.key = "jointUMAP_",
  seed.use = 42
)


# %%
p_joint <- DimPlot(obj, reduction = "umap.joint", group.by = "celltype", label = TRUE) +
  ggtitle("Joint RNA+ATAC latent UMAP") +
  NoLegend()
p_joint
ggsave("p_joint.png", plot = p_joint, width = 8, height = 6, dpi = 300)