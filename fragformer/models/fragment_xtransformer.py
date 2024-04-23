from __future__ import annotations

import torch
import torch.nn as nn
from x_transformers import TransformerWrapper


class FragmentXTransformer(TransformerWrapper):
    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        """
        FragmentXTransformer extends TransformerWrapper to perform token-level and sequence-level classification tasks.
        """
        super().__init__(num_tokens=num_tokens,
                         max_seq_len=max_seq_len,
                         **kwargs)
        
        self.hidden_dim = self.attn_layers.dim
        self.num_tokens = num_tokens

        if checkpoint_path:
            self.load_model(checkpoint_path)


    def forward(self, x, **kwargs):
        """
        Forward pass of the FragmentXTransformer.

        Args:
            x (torch.Tensor): Tokenized input tensor of shape (batch_size, sequence_length).
            **kwargs: Additional keyword arguments for the TransformerWrapper.

        Returns:
            output_dict (dict): Dictionary containing logits for token-level classification tasks.
        """

        x = super().forward(x, **kwargs)
        output_dict = {}
        if 'return_embeddings' in kwargs:
            output_dict['embeddings'] = x
        elif 'return_logits_and_embeddings' in kwargs:
            assert isinstance(x, tuple)
            output_dict['mlm_logits'] = x[0]
            output_dict['embeddings'] = x[1]
        else:
            output_dict['mlm_logits'] = x

        return output_dict

    def load_model(self, checkpoint_path: str):
        """Locate state dict in lightning checkpoint and load into model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Remove "model." or "transformer." prefix from state dict keys and remove any keys containing 'to_cls'
        try:
            state_dict = {
                key.replace("model.", "").replace("transformer.", ""): value
                for key, value in state_dict.items()
                if "to_cls" not in key
            }
        except Exception as e:
            print("Error loading state dict. Please check the checkpoint file.")
            raise e


        self.load_state_dict(state_dict)
