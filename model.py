# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4

# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import Tokenizer

class ModelConfig:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        block_size: int = 64,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        num_local_experts: int = 4,
        num_experts_per_tok: int = 2,
        intermediate_size_expert: int = 256,
        intermediate_size_shared: int = 256
    ):
        """
        Configuration for the Mixture of Experts (MoE) Transformer.

        Parameters
        ----------
        d_model : int
            Embedding dimension.
        n_layers : int
            Number of Transformer blocks.
        n_heads : int
            Number of attention heads.
        block_size : int
            Maximum context length (sequence length).
        vocab_size : int
            Vocabulary size (number of unique tokens).
        rms_norm_eps : float
            Epsilon value for RMS normalization.
        rope_theta : float
            Theta parameter for Rotary Positional Embedding (RoPE).
        num_local_experts : int
            Number of experts per MoE layer.
        num_experts_per_tok : int
            Number of experts to route each token to (Top-K).
        intermediate_size_expert : int
            Hidden size within each expert MLP.
        intermediate_size_shared : int
            Hidden size within the shared MLP.
        """

        # Transformer hyperparameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        
        # RoPE hyperparameters
        self.rope_theta = rope_theta
        
        # MoE hyperparameters
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_size_expert = intermediate_size_expert
        self.intermediate_size_shared = intermediate_size_shared

        # Derived hyperparameters
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads # Dimension of keys/queries/values per head
        self.expert_dim = intermediate_size_expert # Alias for clarity
        self.shared_expert_dim = intermediate_size_shared # Alias for clarity
        
        # Device configuration
        # Set the device (GPU 'cuda' if available, else CPU) for tensor operations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RoPE(nn.Module):
    def __init__(self, d_k: int, rope_theta: float = 10000.0, device: str = 'cpu'):
        """
        Rotary Positional Embedding (RoPE) module. We precompute the inverse frequencies (`inv_freq`) based on `rope_theta` and the dimension of each attention head (`d_k`). The actual rotation angles (`freqs_cis`) depend on the token positions and are calculated dynamically within the forward pass.

        Formulas:
        $$ \theta_i = \frac{1}{\rm{ropetheta}^{2i / d_k}} $$ where $i \in [0, 1, ..., d_k/2 - 1]$. We precompute `inv_freq` which corresponds to this $\theta_i$.

        Parameters
        ----------
        d_k : int
            Dimension of embeddings in each attention head.
        rope_theta : float
            Base theta value for computing frequencies.
        device : str
            Device to run the RoPE calculations on (default is 'cpu').
        """

        super().__init__()
        self.device = device
        self.rope_freq_indices = torch.arange(0, d_k, 2, dtype=torch.float)
        self.inv_freq = 1.0 / (rope_theta ** (self.rope_freq_indices / d_k))
        self.inv_freq = self.inv_freq.to(device)  # Move to the specified device

    def calc_frequencies(self, T: int, B: int) -> torch.Tensor:
        """
        Calculate the frequencies of RoPE for an input-batch based on the precomputed inverse frequencies.

        Parameters
        ----------
        T : int
            Sequence length (number of tokens).

        B : int
            Batch size.

        Returns
        -------
        freqs_cis : torch.Tensor
            Frequencies tensor of shape (B, T, d_k/2). (cis format: cos + i*sin)
        """
        
        # Create position IDs (0 to T-1)
        position_ids = torch.arange(T).unsqueeze(0) # Shape: (1, T)
        position_ids = position_ids.to(self.device)  # Move to the specified device
        
        # Expand inv_freq for batch and sequence length
        # inv_freq shape: (d_k/2) -> (1, d_k/2, 1) -> (B, d_k/2, 1)
        inv_freq_expanded = self.inv_freq.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        
        # position_ids shape: (1, T) -> (B, 1, T)
        pos_ids_expanded = position_ids.expand(B, -1).unsqueeze(1).float()
        
        # Calculate frequencies: (B, d_k/2, 1) @ (B, 1, T) -> (B, d_k/2, T)
        with torch.autocast(device_type=self.device, enabled=False): # RoPE often done in float32
            freqs = (inv_freq_expanded.float() @ pos_ids_expanded).transpose(1, 2) # (B, T, d_k/2)
            # Convert to complex numbers (cis format: cos + i*sin)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # (B, T, d_k/2)
        
        return freqs_cis
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RoPE

        Parameters
        ----------
            x: torch.Tensor
                Input tensor

            freqs_cis: torch.Tensor
                Frequencies tensor of shape (B, T, d_k/2) for RoPE

        Returns
        -------
            torch.Tensor

        """

        B, T, n_heads, _ = x.shape  # B = batch_size, T = sequence_length, n_heads = number of heads, d_k = dimension of each head
        
        # Inline apply_rotary_emb logic
        # Reshape Q or K for complex multiplication: (B, T, n_heads, d_k/2, 2)
        x_rope = x.float().reshape(B, T, n_heads, -1, 2)
        
        # View as complex: (B, T, n_heads, d_k/2)
        x_complex = torch.view_as_complex(x_rope)
        
        # Reshape freqs_cis for broadcasting: (B, T, 1, d_k/2)
        freqs_cis_bthd = freqs_cis.unsqueeze(2)
        
        # Apply rotation
        x_rotated = x_complex * freqs_cis_bthd
        
        # Convert back to real: (B, T, n_heads, d_k/2, 2)
        x_out_real = torch.view_as_real(x_rotated)
        
        # Flatten last two dimensions: (B, T, n_heads, d_k)
        return x_out_real.flatten(3).type_as(x)

class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_local_experts: int,
        num_experts_per_tok: int,
        expert_dim: int,
        shared_expert_dim: int
    ):
        """
        Mixture of Experts (MoE) layer.

        Parameters
        ----------
        d_model : int
            Input embedding dimension.
        num_local_experts : int
            Number of experts in this layer.
        num_experts_per_tok : int
            Number of experts each token is routed to.
        expert_dim : int
            Hidden layer size of each expert MLP.
        shared_expert_dim : int
            Hidden layer size of shared MLP.
        """
        
        super().__init__()

        self.num_experts_per_tok = num_experts_per_tok
        
        # 1. Router
        self.router_linear = nn.Linear(d_model, num_local_experts, bias=False)

        # 2. Experts (Weights as Parameters)
        # Gate/Up Projection Weight: (num_experts, d_model, 2 * expert_dim)
        self.gate_up_w = nn.Parameter(torch.empty(num_local_experts, d_model, 2 * expert_dim))
        nn.init.normal_(self.gate_up_w, mean=0.0, std=0.02) # Example initialization

        # Down Projection Weight: (num_experts, expert_dim, d_model)
        self.down_w = nn.Parameter(torch.empty(num_local_experts, expert_dim, d_model))
        nn.init.normal_(self.down_w, mean=0.0, std=0.02) # Example initialization

        # 3. Shared Expert (Standard MLP layers)
        self.shared_gate = nn.Linear(d_model, shared_expert_dim, bias=False)
        self.shared_up = nn.Linear(d_model, shared_expert_dim, bias=False)
        self.shared_down = nn.Linear(shared_expert_dim, d_model, bias=False)

        # Activation function (used inline)
        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MoE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input.
        """

        B, T, C = x.shape  # B = batch_size, T = sequence_length, C = d_model
        
        # --- Router Logits ---
        # Input x_norm: (B, T, C) -> Router Output: (B, T, num_experts)
        router_logits = self.router_linear(x)

        # --- Expert Selection (Top-K) ---
        # Get top-k experts and their routing weights (logits)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        # Apply sigmoid to make weights sum to <= k (treating experts independently)
        # Or use Softmax if weights should sum to 1 across the k experts
        # Reference script uses sigmoid: B, T, k
        routing_weights = torch.sigmoid(routing_weights)

        # --- Prepare for Expert Calculation ---
        # Flatten B and T dimensions to treat each token independently
        x_flat = x.view(-1, C) # (B*T, C)
        selected_experts_flat = selected_experts.view(-1) # (B*T*k)
        routing_weights_flat = routing_weights.view(-1)   # (B*T*k)

        # Create token indices corresponding to the selected experts
        # token_idx goes from 0 to B*T-1
        # expert_idx is the index of the expert selected (0 to num_experts-1)
        token_idx = torch.arange(B * T, device=x.device).repeat_interleave(self.num_experts_per_tok) # (B*T*k)
        expert_idx = selected_experts_flat # (B*T*k)

        # Gather the hidden states for each token * expert combination
        # Input x_flat: (B*T, C)
        # Index token_idx: (B*T*k)
        # Output expert_inputs: (B*T*k, C)
        expert_inputs = x_flat[token_idx]

        # --- Run Experts ---
        # Get expert weights for the selected experts
        # moe_expert_gate_up_proj[i]: (num_experts, C, 2*expert_dim)
        # Index expert_idx: (B*T*k)
        # Output gate_up_w_selected: (B*T*k, C, 2*expert_dim)
        gate_up_w_selected = self.gate_up_w[expert_idx]
        # moe_expert_down_proj[i]: (num_experts, expert_dim, C)
        # Output down_w_selected: (B*T*k, expert_dim, C)
        down_w_selected = self.down_w[expert_idx]

        # Perform batched matrix multiplication (BMM)
        # expert_inputs: (B*T*k, C) -> unsqueeze for BMM: (B*T*k, 1, C)
        # gate_up_w_selected: (B*T*k, C, 2*expert_dim)
        # Output gate_up_states: (B*T*k, 1, 2*expert_dim)
        gate_up_states = torch.bmm(expert_inputs.unsqueeze(1), gate_up_w_selected)

        # Split gate and up states
        gate, up = gate_up_states.chunk(2, dim=-1) # Each: (B*T*k, 1, expert_dim)

        # Apply activation function (SiLU) and gating
        activated_states = self.activation_fn(gate) * up # (B*T*k, 1, expert_dim)

        # Down projection
        # activated_states: (B*T*k, 1, expert_dim)
        # down_w_selected: (B*T*k, expert_dim, C)
        # Output expert_outputs_flat: (B*T*k, 1, C) -> squeeze -> (B*T*k, C)
        expert_outputs_flat = torch.bmm(activated_states, down_w_selected).squeeze(1) # (B*T*k, C)

        # Weight the outputs by the routing weights
        # expert_outputs_flat: (B*T*k, C)
        # routing_weights_flat: (B*T*k) -> unsqueeze -> (B*T*k, 1)
        expert_outputs_weighted = expert_outputs_flat * routing_weights_flat.unsqueeze(-1) # (B*T*k, C)

        # --- Combine Expert Outputs ---
        # Sum the outputs for each token (across the k selected experts)
        # Need to scatter-add the weighted outputs back to the original token positions
        # Initialize output tensor: (B*T, C)
        combined_expert_outputs = torch.zeros_like(x_flat) # (B*T, C)
        # Scatter Add:
        # combined_expert_outputs: tensor to add into
        # dim=0: dimension along which to index
        # index=token_idx.unsqueeze(-1).expand(-1, C): indices to scatter to (needs C dimension)
        # src=expert_outputs_weighted: values to add
        combined_expert_outputs.scatter_add_(0, token_idx.unsqueeze(-1).expand(-1, C), expert_outputs_weighted)

        # --- Run Shared Expert ---
        shared_gate_val = self.shared_gate(x) # (B, T, shared_expert_dim)
        shared_up_val = self.shared_up(x)     # (B, T, shared_expert_dim)
        shared_activated = self.activation_fn(shared_gate_val) * shared_up_val # (B, T, shared_expert_dim)
        shared_output = self.shared_down(shared_activated)      # (B, T, C)

        # --- Combine MoE and Shared Expert ---
        # Reshape combined_expert_outputs back to (B, T, C)
        moe_output = combined_expert_outputs.view(B, T, C)
        # Add shared expert output
        final_moe_output = moe_output + shared_output

        return final_moe_output  # (B, T, C)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, rope: RoPE, device: str = 'cpu'):
        """
        Multi-Head Attention with Rotary Positional Embeddings (RoPE).

        Parameters
        ----------
        d_model : int
            Input embedding dimension.
        n_heads : int
            Number of attention heads.
        block_size : int
            Maximum sequence length.
        rope : RoPE
            Rotary positional embedding module.
        device : str
            Device to run the attention calculations on (default is 'cpu').
        """

        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size
        self.rope = rope
        self.device = device

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask for attention
        self.causal_mask = self.create_causal_mask(block_size).to(device)


    def create_causal_mask(self, block_size: int) -> torch.Tensor:
        """
        Create a causal mask for attention to prevent attending to future tokens. # Creating the lower triangular mask for causal self-attention. Values are 1 where attention is allowed, 0 where it's masked.

        Parameters
        ----------
        block_size : int
            Context length (sequence length) for the mask.

        Returns
        -------
        torch.Tensor
            Causal mask of shape (1, 1, block_size, block_size).
        """
        
        # Shape: (1, 1, block_size, block_size) for broadcasting with (B, n_heads, T, T)
        causal_mask = torch.tril(torch.ones(block_size, block_size))
        
        return causal_mask.view(1, 1, block_size, block_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention with RoPE.
        B: Batch size, T: Sequence length, C: Embedding dimension

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C)

        freqs_cis : torch.Tensor
            Frequencies tensor of shape (B, T, d_k/2) for RoPE.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, C) after applying multi-head attention.
        """
        
        B, T, C = x.shape  # B = batch_size, T = sequence_length, C = d_model
        
        # QKV projection
        qkv = self.qkv_proj(x) # (B, T, 3*C)

        # Split heads
        qkv = qkv.view(B, T, self.n_heads, 3 * self.d_k) # Reshape before permute

        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, n_heads, d_k)

        # --- Apply RoPE ---
        q = self.rope(q, freqs_cis)  # (B, T, n_heads, d_k)
        k = self.rope(k, freqs_cis)  # (B, T, n_heads, d_k)

        # Permute for attention calculation: (B, n_heads, T, d_k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3) # Permute v as well

        # --- Scaled Dot-Product Attention ---
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_k ** -0.5)
        
        # Apply Causal Mask
        attn_scores = attn_scores.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        attention_weights = F.softmax(attn_scores, dim=-1) # (B, n_heads, T, T)
        # Handle potential NaNs if a row in softmax is all -inf (e.g., first token)
        attention_weights = torch.nan_to_num(attention_weights)

        # Attention output
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        attn_output = attention_weights @ v
        # Concatenate heads: -> (B, T, n_heads, d_k) -> (B, T, C)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output  # (B, T, C)

class MoETransformer(nn.Module):
    """
    Mixture of Experts Transformer model.

    This class defines a Transformer architecture enhanced with MoE (Mixture of Experts)
    layers. It includes token embeddings, rotary positional embeddings (RoPE), RMSNorm,
    and stacked layers of attention and MoE blocks.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
    
        self.config = config
        
        # Embedding layer
        self.token_embedding_table = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.d_model)
        
        # Rotary Positional Embedding (RoPE)
        self.rope = RoPE(d_k=config.d_k, rope_theta=config.rope_theta, device=config.device)

        # RMSNorm layers
        self.rms_norm = nn.ModuleList([
            nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            for _ in range(config.n_layers + 1)
        ])
        
        # Multi Head Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(
                d_model=config.d_model, 
                n_heads=config.n_heads, 
                block_size=config.block_size, 
                rope=self.rope, 
                device=config.device)
            for _ in range(config.n_layers)
        ])

        # Mixture of Experts (MoE) layers
        self.moe_layers = nn.ModuleList([
            MoELayer(
                d_model=config.d_model,
                num_local_experts=config.num_local_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                expert_dim=config.intermediate_size_expert,
                shared_expert_dim=config.intermediate_size_shared
            )
            for _ in range(config.n_layers)
        ])

        # Final output layer
        self.final_output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def generate(self, seed_text: str, tokenizer: Tokenizer = None, max_new_tokens: int = 200) -> str:
        """
        Generate text based on a seed input.

        Parameters
        ----------
        seed_text : str
            Initial text to start generation. 
            If `tokenizer` is provided, this should be a string that can be tokenized.
            If `tokenizer` is None, this should be a list of integer token indices.
        tokenizer : Tokenizer, optional
            Tokenizer to convert text to token indices. If None, it is assumed that
            `seed_text` is already in token index format.
        max_new_tokens : int
            Maximum number of new tokens to generate.

        Returns
        -------
        str or list
            If `tokenizer` is provided, this will be a string.
            If `tokenizer` is None, this will be a list of integer token indices.
        """

        if tokenizer is not None:
            # Convert seed text to token indices using the tokenizer
            seed_ids = tokenizer.encode(seed_text)
        else:
            # make sure seed_text is a list of integer token indices
            if not all(isinstance(i, int) for i in seed_text):
                raise ValueError("If tokenizer is None, seed_text must be a list of integer token indices.")

        generated_sequence = torch.tensor([seed_ids], dtype=torch.long, device=self.config.device)

        # Disable gradient calculations
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # --- 1. Prepare Input Context ---
                # Ensure context doesn't exceed block_size
                current_context = generated_sequence[:, -self.config.block_size:]
                # B_gen, T_gen = current_context.shape
                # C_gen = self.config.d_model

                # --- 2. Forward Pass ---
                logits_gen = self.forward(current_context) # (B_gen, T_gen, vocab_size)

                # --- 3. Get Logits for Last Time Step ---
                logits_last_token = logits_gen[:, -1, :] # Shape: (B_gen, vocab_size)

                # --- 4. Apply Softmax ---
                probs = F.softmax(logits_last_token, dim=-1) # Shape: (B_gen, vocab_size)

                # --- 5. Sample Next Token ---
                next_token = torch.multinomial(probs, num_samples=1) # Shape: (B_gen, 1)

                # --- 6. Append Sampled Token ---
                generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        print("Generation loop finished.")

        # Convert generated token indices back to text if tokenizer is provided
        if tokenizer is not None:
            generated_text = tokenizer.decode(generated_sequence[0].tolist())
            return generated_text
        else: # If tokenizer is None, return the raw token indices
            return generated_sequence[0].tolist()

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE Transformer (inspired by Llama4).
        B: Batch size, T: Sequence length, C: Embedding dimension

        Parameters
        ----------
        xb : torch.Tensor
            Input tensor of shape (B, T).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, vocab_size).
        """

        B, T = xb.shape # B = batch_size, T = block_size
        
        # Embedding layer
        x = self.token_embedding_table(xb)  # (B, T, C)
        
        # Prepare RoPE frequencies
        # Note: RoPE frequencies depend on the sequence length and batch size
        freqs_cis = self.rope.calc_frequencies(T=T, B=B)  # (B, T, d_k/2)

        # --- Transformer Blocks Loop ---
        for i in range(self.config.n_layers):
            # Residual connection starts here
            residual_attn = x

            # --- Input RMSNorm ---
            x = self.rms_norm[i](x)

            # --- Multi-Head Attention (MHA) ---
            attn_output = self.attention_layers[i](x, freqs_cis)  # (B, T, C)

            # --- Add Residual Connection (Attention) ---
            x = residual_attn + attn_output

            # Residual connection before MoE
            residual_moe = x

            # --- Post-Attention RMSNorm ---
            x_norm = self.rms_norm[i](x)

            # --- MoE Block ---
            moe_output = self.moe_layers[i](x_norm)  # (B, T, C)

            # --- Add Residual Connection (MoE/FFN) ---
            x = residual_moe + moe_output

        # --- Final RMSNorm ---
        x_norm = self.rms_norm[-1](x)

        # --- Final Output Layer ---
        logits = self.final_output_layer(x_norm)  # (B, T, vocab_size)

        return logits  # (B, T, vocab_size)