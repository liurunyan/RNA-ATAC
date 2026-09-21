import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import HeteroConv, GCNConv, GATv2Conv
from torch_geometric.utils import negative_sampling


class HeteroGraphAE(nn.Module):
    """Heterogeneous Graph Autoencoder model (without variational reparameterization)."""

    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers=2, **kwargs):
        super().__init__()
        # First heterogeneous convolution layer.
        self.conv1 = HeteroConv({
            ('cell', m, 'cell'): GCNConv(in_channels, hidden_channels) for m in modalities
        }, aggr='sum')
        self.bn1 = nn.ModuleDict({'cell': nn.BatchNorm1d(hidden_channels)})

        # Intermediate layers (if num_layers > 1).
        self.layers = nn.ModuleList([
            HeteroConv({
                ('cell', m, 'cell'): GATv2Conv(hidden_channels, hidden_channels // 8, heads=8)
                for m in modalities
            }, aggr='sum')
            for _ in range(num_layers - 1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.ModuleDict({'cell': nn.BatchNorm1d(hidden_channels)}) for _ in range(num_layers - 1)
        ])

        # Single heterogeneous convolution to compute the latent representation.
        self.z_conv = HeteroConv({
            ('cell', m, 'cell'): GCNConv(hidden_channels, latent_channels) for m in modalities
        }, aggr='sum')

        # Save modalities and create decoders for each modality.
        self.modalities = modalities
        self.decoder_dict = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(latent_channels, 1),
                nn.Sigmoid()
            ) for m in modalities
        })

    def encode(self, data):
        """Encode graph data into latent representation."""
        x_dict = {'cell': data['cell'].x}
        # First layer: conv -> batch norm -> activation.
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: self.bn1[k](x) for k, x in x_dict.items()}
        x_dict = {k: F.silu(x) for k, x in x_dict.items()}

        # Intermediate layers.
        for bn_layer, layer in zip(self.bn_layers, self.layers):
            x_dict = layer(x_dict, data.edge_index_dict)
            x_dict = {k: bn_layer[k](x) for k, x in x_dict.items()}
            x_dict = {k: F.silu(x) for k, x in x_dict.items()}

        # Compute the latent representation.
        z_dict = self.z_conv(x_dict, data.edge_index_dict)
        return z_dict

    def decode(self, z, edge_index):
        """
        Compute dot-product scores and pass them through a modality-specific decoder.
        Returns a dictionary mapping each modality to its predicted edge probabilities.
        """
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
       
        edge_product = z_src * z_dst
        predicted_edges = {}
        for m in self.modalities:
            # Each modality's decoder can learn its own transformation.
            predicted_edges[m] = self.decoder_dict[m](edge_product)
        return predicted_edges

    def forward(self, data):
        z_dict = self.encode(data)
        # We assume that we are working with 'cell' nodes.
        z = z_dict['cell']
        return z


class GraphAELightningModule(pl.LightningModule):
    """PyTorch Lightning module for training the Heterogeneous Graph Autoencoder."""

    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers,
                 num_clusters, clustering_weight):
        super().__init__()
        self.save_hyperparameters(ignore=['modalities'])

        self.model = HeteroGraphAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            modalities=modalities,
            num_layers=num_layers
        )

        # Initialize clustering parameters.
        self.num_clusters = num_clusters
        self.clustering_weight = clustering_weight
        # Learnable cluster centers (initialized randomly).
        self.cluster_centers = torch.nn.Parameter(torch.randn(num_clusters, latent_channels))

    def forward(self, data):
        return self.model(data)

    def compute_clustering_loss(self, z):
        """
        Compute clustering loss as the mean L2 distance from each latent embedding
        to its nearest cluster center.
        z: Tensor of shape [num_nodes, latent_channels]
        """
        distances = torch.cdist(z, self.cluster_centers, p=2)  # shape: [num_nodes, num_clusters]
        min_distances, _ = torch.min(distances, dim=1)
        return torch.mean(min_distances)

    def training_step(self, batch, batch_idx):
        # Obtain latent embeddings.
        z = self.model(batch)

        # Calculate reconstruction loss per modality.
        total_recon_loss = 0.0
        modality_losses = {}
        # Iterate over each edge type that connects 'cell' to 'cell'.
        for key, pos_edge_index in batch.edge_index_dict.items():
            if key[0] == 'cell' and key[2] == 'cell':
                modality = key[1]  # extract modality from the edge key
                # Sample negative edges for this modality.
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=z.size(0),
                    num_neg_samples=pos_edge_index.size(1)
                )
                # Use the decoder to get predictions.
                pos_pred = self.model.decode(z, pos_edge_index)[modality]
                neg_pred = self.model.decode(z, neg_edge_index)[modality]
                preds = torch.cat([pos_pred, neg_pred], dim=0)
                labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
                modality_loss = F.binary_cross_entropy(preds, labels)
                modality_losses[modality] = modality_loss
                total_recon_loss += modality_loss

        # Compute clustering loss.
        cluster_loss = self.compute_clustering_loss(z)

        # Total loss: reconstruction loss (summed over modalities) + weighted clustering loss.
        loss = total_recon_loss + self.clustering_weight * cluster_loss

        batch_size = batch['cell'].x.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("cluster_loss", cluster_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for m, l in modality_losses.items():
            self.log(f"{m}_recon_loss", l, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

