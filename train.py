import os
import math
import random
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from config import Config, get_default_config

from model.model import GraphAELightningModule


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GraphAETrainer(GraphAELightningModule):
    """Extended Lightning module with optimizer configuration for training."""

    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers,
                 learning_rate, total_epochs, warmup_epochs=5):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            modalities=modalities,
            num_layers=num_layers
        )
        self.learning_rate = learning_rate
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            else:
                progress = (current_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }


def load_data(dataset_name: str, base_data_dir: str, device: torch.device):
    """Load preprocessed heterogeneous graph data."""
    graph_path = os.path.join(base_data_dir, dataset_name, f"{dataset_name}_processed.pt")
    print(f"Loading data from: {graph_path}")
    loaded_data = torch.load(graph_path)
    hetero_data = loaded_data.to(device)
    print(f"Data loaded successfully. Number of cells: {hetero_data['cell'].x.size(0)}")
    return hetero_data


def create_dataloader(hetero_data, modalities, batch_size: int = 512):
    """Create a NeighborLoader for mini-batch training."""
    num_cells = hetero_data['cell'].x.size(0)
    cell_idx = torch.arange(num_cells, device=hetero_data['cell'].x.device)

    neighbor_loader = NeighborLoader(
        hetero_data,
        num_neighbors={
            ('cell', m, 'cell'): [5, 5] for m in modalities
        },
        input_nodes=('cell', cell_idx),
        batch_size=batch_size
    )
    print(f"DataLoader created with batch size: {batch_size}")
    return neighbor_loader


def train_model(model, dataloader, n_epochs: int, checkpoint_dir: str):
    """Train the model using PyTorch Lightning Trainer."""
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=checkpoint_dir,
        filename='graph_ae-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    print(f"Starting training for {n_epochs} epochs...")
    trainer.fit(model, train_dataloaders=dataloader)
    print("Training completed!")
    return trainer


def save_embeddings(model, hetero_data, dataset_name: str, output_dir: str):
    """Generate and save latent embeddings."""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        hetero_data = hetero_data.to(model.device)
        z = model(hetero_data)

        # Optionally compute edge predictions
        pos_edge_index = list(hetero_data.edge_index_dict.values())[0]
        pred_edge_probs = model.model.decode(z, pos_edge_index)

    # Save embeddings in different formats
    latent_embedding = z.detach().cpu().numpy()

    npy_path = os.path.join(output_dir, f"{dataset_name}_latent_embedding.npy")
    np.save(npy_path, latent_embedding)

    pt_path = os.path.join(output_dir, f"{dataset_name}_latent_embedding.pt")
    torch.save(z.detach().cpu(), pt_path)

    print(f"Latent embedding shape: {latent_embedding.shape}")
    print(f"Saved latent embedding to: {npy_path}")
    print(f"Saved latent embedding to: {pt_path}")

    return z


def main():
    """Main training pipeline."""
    # Set random seed for reproducibility
    # Load configuration
    cfg = get_default_config()

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)

    # # Configuration
    # DATASET_NAME = "LUNG-CITE"
    # BASE_DATA_DIR = os.path.join("data", "processed")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # MODALITIES = ['Peaks', 'RNA']
    # MODALITIES = ['ADT', 'RNA']

    # # Hyperparameters
    # BATCH_SIZE = 256
    # HIDDEN_CHANNELS = 512
    # LATENT_CHANNELS = 512
    # NUM_LAYERS = 2
    # LEARNING_RATE = 1e-3
    # N_EPOCHS = 500  # Change to 500 for full training
    # WARMUP_EPOCHS = 3
    # Load data
    hetero_data = load_data(cfg.data.dataset_name, cfg.data.base_data_dir, DEVICE)

    # Create dataloader
    dataloader = create_dataloader(hetero_data, cfg.data.modalities, cfg.data.batch_size)

    # Print sample batch information
    for batch in dataloader:
        print(f"Sample batch: {batch}")
        break

    # Initialize model with optimizer configuration
    in_channels = hetero_data['cell'].x.size(1)
    model = GraphAETrainer(
        in_channels=in_channels,
        hidden_channels=cfg.model.hidden_channels,
        latent_channels=cfg.model.latent_channels,
        modalities=cfg.data.modalities,
        num_layers=cfg.model.num_layers,
        learning_rate=cfg.training.learning_rate,
        total_epochs=cfg.training.n_epochs,
        warmup_epochs=cfg.training.warmup_epochs
    )
    print(f"Model initialized with {in_channels} input channels")

    # Train model
    trainer = train_model(model, dataloader,  cfg.training.n_epochs, cfg.output.checkpoint_dir)

    # Save embeddings
    z = save_embeddings(model, hetero_data, cfg.data.dataset_name, cfg.output.embedding_dir)

    print("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()



