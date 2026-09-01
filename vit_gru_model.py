# =============================================================================
# vit_gru_model.py
# Brain Tumor Detection - Hybrid ViT + GRU Model Architecture
# =============================================================================
# This file defines the core deep learning model combining:
#   1. Vision Transformer (ViT) for spatial feature extraction from MRI patches
#   2. GRU (Gated Recurrent Unit) for sequential pattern learning across patches
#   3. Fully-connected head for 4-class tumor classification
# =============================================================================

import torch                          # Core PyTorch framework
import torch.nn as nn                 # Neural network modules
import timm                           # Library for pretrained Vision Transformers


class HybridViTGRU(nn.Module):
    """
    Hybrid Vision Transformer + GRU model for brain tumor classification.

    Architecture Flow:
        Input MRI Image (B, 3, 224, 224)
            ↓
        ViT Backbone → Patch Embeddings (B, num_patches, embed_dim)
            ↓
        GRU Layer → Sequential Features (B, hidden_size)
            ↓
        Dropout → FC Layer → Output (B, num_classes=4)

    Where B = batch size
    """

    def __init__(
        self,
        num_classes: int = 4,          # glioma, meningioma, pituitary, no_tumor
        vit_model_name: str = "vit_base_patch16_224",  # pretrained ViT variant
        pretrained: bool = True,        # use ImageNet pretrained weights
        gru_hidden_size: int = 256,     # number of GRU hidden units
        gru_num_layers: int = 2,        # stacked GRU layers
        dropout_rate: float = 0.3,      # dropout for regularization
    ):
        """
        Initialize the Hybrid ViT-GRU model.

        Args:
            num_classes:      Number of output classes (4 for this project)
            vit_model_name:   timm model name for Vision Transformer backbone
            pretrained:       Whether to load ImageNet pretrained ViT weights
            gru_hidden_size:  Hidden dimension of GRU layers
            gru_num_layers:   Number of stacked GRU layers
            dropout_rate:     Dropout probability for regularization
        """
        super(HybridViTGRU, self).__init__()  # initialize parent nn.Module

        # ── 1. Vision Transformer Backbone ──────────────────────────────────
        # Load pretrained ViT; we keep it as a feature extractor
        # features_only=False → we'll extract intermediate representations manually
        self.vit = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,          # remove ViT's classification head
        )

        # Determine the embedding dimension of the ViT
        # vit_base_patch16_224 → embed_dim = 768
        self.embed_dim = self.vit.embed_dim  # e.g., 768

        # ── 2. GRU Layer ────────────────────────────────────────────────────
        # ViT splits a 224×224 image into 14×14 = 196 patches (+ 1 cls token = 197)
        # We treat each patch embedding as a time step in the sequence
        # GRU learns long-range dependencies across patches
        self.gru = nn.GRU(
            input_size=self.embed_dim,    # each patch embedding = 768-dim vector
            hidden_size=gru_hidden_size,  # GRU hidden state size = 256
            num_layers=gru_num_layers,    # 2 stacked GRU layers
            batch_first=True,             # input shape: (batch, seq_len, features)
            dropout=dropout_rate if gru_num_layers > 1 else 0.0,  # inter-layer dropout
            bidirectional=False,          # unidirectional GRU
        )

        # ── 3. Classification Head ───────────────────────────────────────────
        # After GRU, take the last hidden state → dropout → FC → 4 classes
        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),   # FC: 256 → 128
            nn.ReLU(inplace=True),              # non-linear activation
            nn.Dropout(p=dropout_rate),         # regularization
            nn.Linear(128, num_classes),        # FC: 128 → 4 (final output)
        )

        # ── 4. Layer Normalization ───────────────────────────────────────────
        # Normalize GRU output for training stability
        self.layer_norm = nn.LayerNorm(gru_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
               - batch_size: number of images in the batch
               - 3: RGB channels
               - 224×224: image dimensions

        Returns:
            logits: Raw class scores of shape (batch_size, num_classes=4)
                    Apply softmax externally for probabilities
        """

        # ── Step 1: Extract ViT Patch Embeddings ────────────────────────────
        # ViT processes image as sequence of flattened patches
        # patch_tokens shape: (batch_size, num_patches + 1, embed_dim)
        #   = (B, 197, 768) for vit_base_patch16_224
        # The +1 is the [CLS] token prepended by ViT
        patch_tokens = self.vit.forward_features(x)
        # shape: (B, 197, 768)

        # ── Step 2: Remove [CLS] token, keep only patch tokens ──────────────
        # Index [1:] removes the class token at position 0
        # We want pure spatial patch embeddings for GRU sequence learning
        patch_tokens = patch_tokens[:, 1:, :]
        # shape: (B, 196, 768)  ← 196 patches from 14×14 grid

        # ── Step 3: Pass Patch Sequence Through GRU ─────────────────────────
        # GRU processes the 196 patch embeddings as a temporal sequence
        # gru_out: all hidden states for each time step (B, 196, 256)
        # hidden:  final hidden state (num_layers, B, 256)
        gru_out, hidden = self.gru(patch_tokens)
        # gru_out shape: (B, 196, 256)
        # hidden shape:  (2, B, 256) for 2-layer GRU

        # ── Step 4: Extract Final Hidden State ──────────────────────────────
        # Take the last layer's final hidden state as the sequence summary
        # hidden[-1] → shape: (B, 256)
        final_hidden = hidden[-1]
        # shape: (B, 256)

        # ── Step 5: Normalize and Apply Dropout ─────────────────────────────
        final_hidden = self.layer_norm(final_hidden)   # normalize
        final_hidden = self.dropout(final_hidden)      # regularize

        # ── Step 6: Classification ───────────────────────────────────────────
        # Pass through FC layers to get class logits
        logits = self.classifier(final_hidden)
        # shape: (B, 4)

        return logits

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings from ViT (used for Grad-CAM).

        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            Patch embeddings (B, 196, 768) without [CLS] token
        """
        patch_tokens = self.vit.forward_features(x)
        return patch_tokens[:, 1:, :]  # remove CLS token


def create_model(
    num_classes: int = 4,
    pretrained: bool = True,
    device: str = "cpu",
) -> HybridViTGRU:
    """
    Factory function to create and initialize the Hybrid ViT-GRU model.

    Args:
        num_classes: Number of tumor classes (default: 4)
        pretrained:  Use pretrained ViT weights (default: True)
        device:      Target device - 'cuda' or 'cpu'

    Returns:
        Initialized HybridViTGRU model moved to specified device
    """
    # Instantiate the model
    model = HybridViTGRU(
        num_classes=num_classes,
        vit_model_name="vit_base_patch16_224",
        pretrained=pretrained,
        gru_hidden_size=256,
        gru_num_layers=2,
        dropout_rate=0.3,
    )

    # Move model to target device (GPU if available)
    model = model.to(device)

    print(f"[Model] HybridViTGRU created on device: {device}")
    print(f"[Model] ViT embed dim: {model.embed_dim}")
    print(f"[Model] GRU hidden size: 256, layers: 2")
    print(f"[Model] Output classes: {num_classes}")

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total trainable parameters: {total_params:,}")

    return model


def get_model_summary(model: HybridViTGRU, device: str = "cpu") -> None:
    """
    Print a detailed model summary with tensor shapes at each stage.

    Args:
        model:  The HybridViTGRU model instance
        device: Device to run the dummy forward pass on
    """
    print("\n" + "=" * 60)
    print("       HYBRID ViT-GRU MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"\nInput  Shape : (B, 3, 224, 224)")
    print(f"  ↓ ViT Backbone (vit_base_patch16_224)")
    print(f"Patch Tokens : (B, 197, 768)  [196 patches + 1 CLS]")
    print(f"  ↓ Remove CLS token")
    print(f"Patch Seq    : (B, 196, 768)")
    print(f"  ↓ GRU (input=768, hidden=256, layers=2)")
    print(f"GRU Output   : (B, 196, 256)")
    print(f"Final Hidden : (B, 256)       [last time step]")
    print(f"  ↓ LayerNorm + Dropout")
    print(f"  ↓ FC(256→128) + ReLU + Dropout")
    print(f"  ↓ FC(128→4)")
    print(f"Output       : (B, 4)         [class logits]")
    print("=" * 60)

    # Run dummy forward pass to verify shapes
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224).to(device)  # batch of 2
        output = model(dummy_input)
        print(f"\n[Verification] Input:  {dummy_input.shape}")
        print(f"[Verification] Output: {output.shape}")
        print(f"[Verification] Forward pass successful ✓")
    print("=" * 60 + "\n")


# =============================================================================
# Quick test: run this file directly to verify model loads correctly
# =============================================================================
if __name__ == "__main__":
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = create_model(num_classes=4, pretrained=True, device=device)

    # Print architecture summary
    get_model_summary(model, device=device)

    print("vit_gru_model.py loaded successfully ✓")
