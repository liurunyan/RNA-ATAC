file_name <- "LUNG-CITE"
clean_assay <- TRUE

message(sprintf("File name: %s", file_name))
message(sprintf("Final setting: clean_assay = %s", clean_assay))


createCleanAssay <- function(seurat_object, assay_name) {

    if (!assay_name %in% Assays(seurat_object)) {
        stop(sprintf("Assay '%s' not found in Seurat object.", assay_name),
             call. = FALSE)
    }
    DefaultAssay(seurat_object) <- assay_name

    # clean_object <- DietSeurat(
    #     seurat_object,
    #     assays      = assay_name,  
    #     counts      = TRUE,        
    #     data        = TRUE,        
    #     scale.data  = FALSE,       
    #     features    = NULL,        
    #     dimreducs   = NULL,        
    #     graphs      = NULL,       
    #     misc        = FALSE        
    # )

    clean_object <- DietSeurat(
    seurat_object,
    assays = assay_name,
    layers = c("counts", "data"),
    features = NULL,
    dimreducs = NULL,
    graphs = NULL,
    misc = FALSE
    )

    VariableFeatures(clean_object[[assay_name]]) <- character(0)

    if ("neighbors" %in% slotNames(clean_object)) {
        clean_object@neighbors <- list()
    }
    clean_object@commands <- list()
    clean_object@tools    <- list()

    DefaultAssay(clean_object) <- assay_name

    clean_object
}



processSeuratFile <- function(file_name, overwrite = TRUE, delete_h5s = TRUE, download_channel = 63, clean_assay=TRUE) {
  # Load required libraries
  library(Seurat)
  library(SeuratDisk)
  library(SeuratObject)

  # Construct the source file path
  rds_file <- file.path("data", "raw", file_name, sprintf("%s.Rds", file_name))

  # Load original Seurat object from an RDS file
  object <- readRDS(rds_file)
  print(object)


  # Distinguish output naming based on whether clean assays are used.
  if (isTRUE(clean_assay)) {
    output_file_name <- sprintf("%s_clean", file_name)
    message(sprintf("clean_assay = TRUE; file_name used for output: %s", output_file_name))
  } else {
    output_file_name <- file_name
    message(sprintf("clean_assay = FALSE; file_name used for output: %s", output_file_name))
  }
  output_file_name <- file_name

  # Loop over each assay (modality) in the Seurat object
  assay_names <- names(object@assays)
  for (assay in assay_names) {
  message(sprintf("Processing assay: %s", assay))

  if (isTRUE(clean_assay)) {
    object_to_convert <- createCleanAssay(object, assay)
    message(sprintf("Using clean assay for '%s'; output file prefix: %s", assay, output_file_name))
  } else {
    object_to_convert <- object
    DefaultAssay(object_to_convert) <- assay   # still set default on the copy
    message(sprintf("Using original Seurat object for '%s'; output file prefix: %s", assay, output_file_name))
  }

  # Remove the line: output_file_name <- file_name

  prefix<- file.path("data", "raw", file_name, sprintf("%s_%s", output_file_name, assay))
  h5s_file  <- sprintf("%s.h5Seurat", prefix)

  # save the single-assay object, not the full original
  SaveH5Seurat(object_to_convert, filename = h5s_file, overwrite = overwrite)
  Convert(h5s_file, dest = "h5ad", overwrite = overwrite)
}
  # for (assay in assay_names) {
  #   message(sprintf("Processing assay: %s", assay))

  #   # Set the current assay as the default so that it becomes the primary (.X) in conversion
  #   # DefaultAssay(object) <- assay

  #   if (isTRUE(clean_assay)) {
  #     # Create a clean assay to avoid issues with scale.data and variable features.
  #     object_to_convert <- createCleanAssay(object, assay)
  #     message(sprintf("Using clean assay for '%s'; output file prefix: %s", assay, output_file_name))
  #   } else {
  #     # Keep the original Seurat object for conversion if clean-assay mode is disabled.
  #     object_to_convert <- object
  #     message(sprintf("Using original Seurat object for '%s'; output file prefix: %s", assay, output_file_name))
  #   }


  #   # Ensure the target assay is the default before saving/converting.
  #   DefaultAssay(object) <- assay

  #   # Construct file prefixes and filenames for this assay.
  #   # The h5Seurat file is saved in data/ and the h5ad file is intended to be in data/raw/
  #   prefix_h5s <- file.path("data", "raw", file_name, sprintf("%s_%s", output_file_name, assay))

  #   h5s_file <- sprintf("%s.h5Seurat", prefix_h5s)

  #   prefix_h5ad <- file.path("data", "raw", file_name, sprintf("%s_%s", output_file_name, assay))
  #   h5ad_file <- sprintf("%s.h5ad", prefix_h5ad)

  #   # Save the Seurat object as an h5Seurat file for this assay.
  #   SaveH5Seurat(object_to_convert, filename = h5s_file, overwrite = overwrite)

  #   # Convert the h5Seurat file to an h5ad file for this assay.
  #   # Note: Convert() writes the h5ad file in the same directory as the h5Seurat file.
    
  #   Convert(h5s_file, dest = "h5ad", overwrite = overwrite)
  #   # Determine the converted file name (it will have the same base as h5s_file, but with a .h5ad extension)
  #   converted_file <- sub("\\.h5Seurat$", ".h5ad", h5s_file)
  #   }
}

# Call the function with the provided (or default) file name.
processSeuratFile(file_name = file_name, overwrite = TRUE, delete_h5s = TRUE, download_channel = 63, clean_assay = clean_assay)




