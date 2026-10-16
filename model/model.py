import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.utils import negative_sampling


class HeteroGraphAE(nn.Module):
    """Heterogeneous Graph Autoencoder with decreasing hidden dimensions across layers."""

    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers=2, **kwargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.modalities = modalities
        
        # # Parse hidden_channels: int
        # step = hidden_channels // (num_layers + 1)
        # self.hidden_dims = [hidden_channels - i * step for i in range(num_layers)]    
        # print(f"Hidden dimensions for each layer: {self.hidden_dims}")
        if num_layers == 1:
                # Single layer case
                self.hidden_dims = [hidden_channels]
        else:
                # Multi-layer case: exponentially decrease from hidden_channels to close to latent_channels
                ratio = (hidden_channels / latent_channels) ** (1.0 / num_layers)
                self.hidden_dims = []
                current_dim = hidden_channels
                for i in range(num_layers):
                    self.hidden_dims.append(int(current_dim))
                    current_dim = current_dim / ratio
        
        # Create multiple heterogeneous convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer: in_channels -> hidden_dims[0]
        self.convs.append(HeteroConv({
            ('cell', m, 'cell'): GCNConv(in_channels, self.hidden_dims[0]) for m in modalities
        }, aggr="sum"))
        self.bns.append(nn.ModuleDict({'cell': nn.BatchNorm1d(self.hidden_dims[0])}))
        
        # Subsequent layers: hidden_dims[i-1] -> hidden_dims[i]
        for i in range(1, num_layers):
            self.convs.append(HeteroConv({
                ('cell', m, 'cell'): GCNConv(self.hidden_dims[i-1], self.hidden_dims[i]) for m in modalities
            }, aggr="sum"))
            self.bns.append(nn.ModuleDict({'cell': nn.BatchNorm1d(self.hidden_dims[i])}))

        # Final layer to compute latent representation: last hidden_dim -> latent_channels
        # self.z_conv = HeteroConv({
        #     ('cell', m, 'cell'): GCNConv(self.hidden_dims[-1], latent_channels) for m in modalities
        # }, aggr='sum')

        # for attention: aggr=none 
        self.z_conv = HeteroConv({
            ('cell', m, 'cell'): GCNConv(self.hidden_dims[-1], latent_channels) for m in modalities
        }, aggr=none)

        # Create decoders for each modality
        self.decoder_dict = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(latent_channels, 1),
                nn.Sigmoid()
            ) for m in modalities
        })

    def encode(self, data):
        """Encode graph data into latent representation."""
        x_dict = {'cell': data['cell'].x}
    
        # Apply multiple conv layers with batch norm and activation
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, data.edge_index_dict)
            x_dict = {k: self.bns[i][k](x) for k, x in x_dict.items()}
            x_dict = {k: F.silu(x) for k, x in x_dict.items()}
        print("aaaaaaa:", x_dict['cell'].shape())
        exit()
        # Compute the latent representation
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
            predicted_edges[m] = self.decoder_dict[m](edge_product)
        return predicted_edges

    def forward(self, data):
        z_dict = self.encode(data)
        z = z_dict['cell']
        return z


class Attention(Module):
    """
    General Attention Fusion Module (Inspired by SpatialGlue), used for intra- or inter-modality fusion.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Attention, self).__init__()                    
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Trainable parameters: W_omega and U_omega (used to compute attention weights)
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Use Xavier uniform initialization for stable training
        init.xavier_uniform_(self.w_omega)
        init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        # Stack the two input embeddings [N, 2, D]
        emb_stack = torch.stack([emb1, emb2], dim=1)

        # 1. Compute attention score V = tanh(E * W_omega)
        v = torch.tanh(torch.matmul(emb_stack, self.w_omega))
        # 2. Compute vu = V * U_omega [N, 2, 1]
        vu = torch.matmul(v, self.u_omega)
        # 3. Compute Softmax weights Alpha across modality dimension 1
        # squeeze(2) removes the last dimension of size 1 [N, 2]
        alpha = F.softmax(vu.squeeze(2) + 1e-6, dim=1)

        # 4. Weighted Sum: emb_combined = sum(E * Alpha)
        # unsqueeze(-1) adds a dimension at the end [N, 2, 1] for broadcast multiplication
        emb_combined = torch.sum(emb_stack * alpha.unsqueeze(-1), dim=1)

        # Return the fused embedding and attention weights
        return emb_combined, alpha


class GraphAE(pl.LightningModule):
    """PyTorch Lightning module for training the Heterogeneous Graph Autoencoder."""

    def __init__(self, in_channels, hidden_channels, latent_channels, 
                 modalities, num_layers):
        super().__init__()
        self.save_hyperparameters(ignore=['modalities'])

        self.model = HeteroGraphAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            modalities=modalities,
            num_layers=num_layers
        )

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        # Obtain latent embeddings.
        z = self.model(batch)

        # Calculate reconstruction loss per modality.
        total_recon_loss = 0.0
        modality_losses = {}
        
        for key, pos_edge_index in batch.edge_index_dict.items():
            if key[0] == 'cell' and key[2] == 'cell':
                modality = key[1]
                
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=z.size(0),
                    num_neg_samples=pos_edge_index.size(1)
                )
                
                pos_pred = self.model.decode(z, pos_edge_index)[modality]
                neg_pred = self.model.decode(z, neg_edge_index)[modality]
                preds = torch.cat([pos_pred, neg_pred], dim=0)
                labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
                modality_loss = F.binary_cross_entropy(preds, labels)
                modality_losses[modality] = modality_loss
                total_recon_loss += modality_loss

        loss = total_recon_loss

        batch_size = batch['cell'].x.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for m, l in modality_losses.items():
            self.log(f"{m}_recon_loss", l, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss