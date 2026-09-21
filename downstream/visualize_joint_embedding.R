

library(Seurat)
library(Signac)
library(dplyr)
library(ggplot2)
library(RcppCNPy)

# ============================================================
# 1. Paths
# ============================================================

base_path <- "/data/run01/scxl504/GAT/sc"

initial_path <- file.path(
  base_path,
  "data",
  "raw"
)

file_source <- "PBMC-Multiome"
file_name <- "PBMC-Multiome.Rds"

embedding_path <- file.path(
  base_path,
  "results",
  "embedding",
  "PBMC-Multiome_latent_embedding.npy"
)

# ============================================================
# 2. Load Seurat object
# ============================================================

file_path <- file.path(
  initial_path,
  file_source,
  file_name
)

obj <- readRDS(file_path)

print(obj)

cat(
  "Number of cells:",
  ncol(obj),
  "\n"
)

# ============================================================
# 3. Load latent embedding
# ============================================================

embedding <- npyLoad(embedding_path)

cat(
  "Embedding dimensions:",
  nrow(embedding),
  "x",
  ncol(embedding),
  "\n"
)

# ============================================================
# 4. Check number of cells
# ============================================================

if (nrow(embedding) != ncol(obj)) {

  stop(
    "Number of cells does not match: embedding has ",
    nrow(embedding),
    " cells, but Seurat object has ",
    ncol(obj),
    " cells."
  )
}

# ============================================================
# 5. Set row/column names
#
# embedding:
#   cells × latent dimensions
# ============================================================

rownames(embedding) <- colnames(obj)

colnames(embedding) <- paste0(
  "latent_",
  seq_len(ncol(embedding))
)

# ============================================================
# 6. Create DimReduc object
# ============================================================

obj[["joint"]] <- CreateDimReducObject(
  embeddings = embedding,
  key = "joint_",
  assay = DefaultAssay(obj)
)

# ============================================================
# 7. FindNeighbors
#
# Match Scanpy:
#
# sc.pp.neighbors(
#     adata_latent,
#     n_neighbors=15
# )
#
# Seurat equivalent:
#   k.param = 15
#
# cosine = FALSE -> Euclidean distance
# ============================================================

obj <- FindNeighbors(
  obj,
  reduction = "joint",
  dims = 1:ncol(embedding),
  k.param = 15,
  annoy.metric = "euclidean",
  verbose = FALSE
)

# ============================================================
# 8. RunUMAP
#
# Match Scanpy as closely as possible:
#
# sc.tl.umap(
#     adata_latent,
#     random_state=42
# )
#
# Scanpy default:
#   min_dist = 0.5
#   spread   = 1.0
#
# Seurat uses uwot by default.
# ============================================================

set.seed(42)

obj <- RunUMAP(
  obj,
  reduction = "joint",
  dims = 1:ncol(embedding),

  # Same neighborhood size
  n.neighbors = 15,

  # Scanpy defaults
  min.dist = 0.5,
  spread = 1.0,

  # Use Euclidean metric
  metric = "euclidean",

  # Random seed
  seed.use = 42,

  reduction.name = "umap.joint",
  reduction.key = "jointUMAP_",

  verbose = TRUE
)

# ============================================================
# 9. Plot
# ============================================================

p_joint <- DimPlot(
  obj,
  reduction = "umap.joint",
  group.by = "celltype",
  label = TRUE
) +
  ggtitle(
    "Joint RNA+ATAC latent UMAP (Seurat)"
  ) +
  NoLegend()

print(p_joint)

# ============================================================
# 10. Save figure
# ============================================================

ggsave(
  filename = file.path(
    base_path,
    "PBMC-Multiome_Seurat_UMAP.png"
  ),
  plot = p_joint,
  width = 8,
  height = 6,
  dpi = 300
)

