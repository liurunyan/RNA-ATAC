# %%
library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)

# %%
base_path <- "/data/run01/scxl504/GAT/sc"
initial_path <- file.path(base_path, "data", "raw")
file_source <- "PBMC-Multiome"
# file_name <- "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
file_name <- "PBMC-Multiome.Rds"
K <- 50

# %%
# file_path <- file.path(initial_path, file_source, paste0(file_name, ".Rds"))
file_path <- file.path(initial_path, file_source, file_name)
obj_input <- readRDS(file_path)
# obj_input <- Read10X_h5(file_path)

# message("Loaded object of class: ", paste(class(obj_input), collapse = ", "))
print(obj_input)


# %%
if (inherits(obj_input, "Seurat")) {
  cells <- colnames(obj_input)
} else {
  cells <- colnames(obj_input[[1]])
}
n_half <- length(cells) %/% 10
half_cells <- sample(cells, n_half)

if (inherits(obj_input, "Seurat")) {
  obj_input_half <- subset(obj_input, cells = half_cells)
} else {
  obj_input_half <- lapply(obj_input, function(x) {
    if (!is.null(colnames(x))) x[, half_cells, drop = FALSE] else x
  })
}
obj_input_half 


# %%
# Extract multiome data with safe handling
if (inherits(obj_input_half, "Seurat")) {
  rna_counts <- tryCatch(
    GetAssayData(obj_input_half, assay = "RNA", layer = "counts"),
    error = function(e) GetAssayData(obj_input_half, assay = "RNA")
  )
  atac_counts <- tryCatch(
    GetAssayData(obj_input_half, assay = "Peaks", layer = "counts"),
    error = function(e) GetAssayData(obj_input_half, assay = "Peaks")
  )
} else {
  rna_counts <- obj_input_half$RNA
  atac_counts <- obj_input_half$Peaks
}

# If Peaks is an Assay/ChromatinAssay, extract counts matrix (Seurat v5 uses 'layer')
if (inherits(atac_counts, c("Assay", "ChromatinAssay"))) {
  counts_mat <- tryCatch(
    GetAssayData(atac_counts, layer = "counts"),
    error = function(e) {
      message("GetAssayData(layer=) failed, trying default GetAssayData()\nError: ", e$message)
      GetAssayData(atac_counts)
    }
  )
} else {
  counts_mat <- atac_counts
}

if (is.null(rownames(counts_mat))) {
  stop("ATAC counts have no rownames; expected 'chr:start-end' rownames.")
}

atac_counts <- counts_mat
rm(counts_mat)
gc()

# %%
obj <- CreateSeuratObject(counts = rna_counts)
obj@meta.data <- obj_input_half@meta.data
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

# Now add in the ATAC-seq data
# we'll only use peaks in standard chromosomes
grange.counts <- StringToGRanges(rownames(atac_counts), sep = c(":", "-"))
grange.use <- seqnames(grange.counts) %in% standardChromosomes(grange.counts) 
atac_counts <- atac_counts[as.vector(grange.use), ]
# annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
# seqlevelsStyle(annotations) <- 'UCSC'
# genome(annotations) <- "hg38"

# frag.file <- "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz"
# frag.file <- file.path(initial_path, file_source, "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz")
chrom_assay <- CreateChromatinAssay(
   counts = atac_counts,
   sep = c(":", "-"),
  #  genome = 'hg38',
  #  fragments = frag.file,
   min.cells = 10,
  #  annotation = annotations
 )
obj[["ATAC"]] <- chrom_assay

VlnPlot(obj, features = c("nCount_ATAC", "nCount_RNA","percent.mt"), ncol = 3,
  log = TRUE, pt.size = 0) + NoLegend()
rm(chrom_assay, annotations, atac_counts)
gc()


# %%
# processed_dir <- file.path(base_path, "data", "processed", file_source)
# if (!dir.exists(processed_dir)) {
#   dir.create(processed_dir, recursive = TRUE)
# }

# saveRDS(obj, file = file.path(processed_dir, "obj_processed.rds"))
# message("Saved processed Seurat object to: ", file.path(processed_dir, "obj_processed.rds"))


# %%
# RNA analysis
DefaultAssay(obj) <- "RNA"
obj <- SCTransform(obj, verbose = FALSE) %>%
                  RunPCA() %>% 
                  RunUMAP(dims = 1:K, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')
obj

# %%
p1 <- DimPlot(obj, reduction = "umap.rna", group.by = "celltype", label = TRUE) + 
              ggtitle("RNA UMAP") + 
              NoLegend()
p1
ggsave("p1.png", plot = p1, width = 8, height = 6, dpi = 300)

# %%


# %%
DefaultAssay(obj) <- "ATAC"
obj <- RunTFIDF(obj)
obj <- FindTopFeatures(obj, min.cutoff = 'q0')
obj <- RunSVD(obj)
obj <- RunUMAP(obj, reduction = 'lsi', dims = 2:K, reduction.name = "umap.atac", reduction.key = "atacUMAP_")

gc()

# %%
p2 <- DimPlot(obj, reduction = "umap.atac", group.by = "celltype", label = TRUE) + 
              ggtitle("ATAC UMAP") +
              NoLegend()
p2
ggsave("p2.png", plot = p2, width = 8, height = 6, dpi = 300)



# Joint UMAP of RNA and ATAC embeddings
rna_umap <- as.data.frame(Embeddings(obj, reduction = "umap.rna"))
atac_umap <- as.data.frame(Embeddings(obj, reduction = "umap.atac"))

rna_umap$cell <- rownames(rna_umap)
rna_umap$modality <- "RNA"
atac_umap$cell <- rownames(atac_umap)
atac_umap$modality <- "ATAC"

colnames(rna_umap)[1:2] <- c("UMAP_1", "UMAP_2")
colnames(atac_umap)[1:2] <- c("UMAP_1", "UMAP_2")

joint_umap <- rbind(rna_umap, atac_umap)

p3 <- ggplot(joint_umap, aes(x = UMAP_1, y = UMAP_2, color = modality)) +
        geom_point(alpha = 0.7, size = 0.8) +
        theme_classic(base_size = 12) +
        labs(title = "Joint RNA/ATAC UMAP", color = "Modality") +
        scale_color_manual(values = c("RNA" = "#1f77b4", "ATAC" = "#ff7f0e"))
ggsave("p3.png", plot = p3, width = 8, height = 6, dpi = 300)
# %% [markdown]
# ### Downstream Analysis

# %%




