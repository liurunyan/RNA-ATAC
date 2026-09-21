import os
import sys
import torch
import scanpy as sc
import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph
from preprocess import preprocess_scRNA, preprocess_ADT, preprocess_scATAC, preprocess_Peaks, remove_lsi_key
DATASET_NAME = "LUNG-CITE"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BASE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

INPUT_DIR = os.path.join(BASE_DATA_DIR, "raw", DATASET_NAME)
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "processed", DATASET_NAME)
OUTPUT_PATH = os.path.join( OUTPUT_DIR, f"{DATASET_NAME}_processed.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("========================================")
print("Dataset :", DATASET_NAME)
print("Input   :", INPUT_DIR)
print("Device  :", DEVICE)
print("Output  :", OUTPUT_PATH)
print("========================================")


dataset_config = {
    "LUNG-CITE": {
        "modalities": ["ADT", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad",
        "pca_components": {
            "ADT": 52
        }
    },
    "PBMC-Multiome": {
        "modalities": ["Peaks", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad",
    }
}


def create_hetero_graph(data_dict, config, device):
    """Create PyG HeteroData object with KNN graphs"""
    processed = {m: {'x': torch.tensor(data_dict[m].X, dtype=torch.float)} 
                for m in config["modalities"]}
    data = HeteroData(processed)
    
    # Create cell node features
    data['cell'].x = torch.cat([data[m].x for m in config["modalities"]], dim=1)
    
    # Create KNN graphs
    data = data.to(device)
    for m in config["modalities"]:
        data['cell', m, 'cell'].edge_index = knn_graph(
            data[m].x, k=10, cosine=True, num_workers=16
        )
    return data.cpu()

def load_dataset(dataset_name, base_dir, device):
    config = dataset_config[dataset_name]
    data_dict = {}
    
    for modality in config["modalities"]:
        # Get filename pattern from config
        fname = config["file_pattern"].format(
            dataset=dataset_name,
            modality=config.get("modality_map", {}).get(modality, modality))
        
        input_path = os.path.join(base_dir, fname)
        
        # Preprocessing
        if modality == "RNA":
            data_dict[modality] = preprocess_scRNA(input_path)
        elif modality == "ADT":
            data_dict[modality] = preprocess_ADT(
                input_path,
                n_pcs=config.get("pca_components", {}).get(modality, None))
        elif modality == "ATAC":
            try:
                data_dict[modality] = preprocess_scATAC(input_path)
            except ValueError:
                remove_lsi_key(input_path)
                data_dict[modality] = preprocess_scATAC(input_path)
        elif modality == "Peaks":
            try:
                data_dict[modality] = preprocess_Peaks(input_path)
            except ValueError:
                remove_lsi_key(input_path)
                data_dict[modality] = preprocess_Peaks(input_path)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    return create_hetero_graph(data_dict, config, device), data_dict

# return create_hetero_graph(data_dict, config, device), data_dict
hetero_data, data_dict = load_dataset(
    DATASET_NAME,
    INPUT_DIR,
    DEVICE
)

print("\nHeteroData:")
print(hetero_data)

print("\nNode types:")
print(hetero_data.node_types)

print("\nEdge types:")
print(hetero_data.edge_types)

print("\nCell feature shape:")
print(hetero_data["cell"].x.shape)

# for edge_type in hetero_data.edge_types:
#     print(
#         edge_type,
#         hetero_data[edge_type].edge_index.shape
#     )

torch.save(
    hetero_data,
    OUTPUT_PATH
)

print("\nSaved:")
print(OUTPUT_PATH)