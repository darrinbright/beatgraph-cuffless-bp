import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm_groups(num_channels):
    return 8 if num_channels % 8 == 0 else 1


def init_module_weights(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _ = x.shape
        squeeze = self.pool(x).view(batch_size, channels)
        scale = self.se(squeeze).view(batch_size, channels, 1)
        return x * scale


class AttentionGate1D(nn.Module):
    def __init__(self, gate_channels, skip_channels):
        super().__init__()
        intermediate = max(1, gate_channels // 4)
        self.W_g = nn.Conv1d(gate_channels, intermediate, kernel_size=1, bias=True)
        self.W_x = nn.Conv1d(skip_channels, intermediate, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(group_norm_groups(skip_channels), skip_channels)
        self.psi = nn.Conv1d(intermediate, 1, kernel_size=1, bias=True)

    def forward(self, gate, skip):
        gate = F.interpolate(gate, size=skip.shape[-1], mode="linear", align_corners=False)
        skip = self.norm(skip)
        attention = torch.sigmoid(self.psi(F.relu(self.W_g(gate) + self.W_x(skip))))
        return skip * attention


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout=0.25, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        groups = group_norm_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )
        self.use_se = use_se
        self.se = SEBlock1D(out_channels) if use_se else None

    def forward(self, x):
        x = self.block(x)
        if self.se is not None:
            x = self.se(x)
        return x


class BeatEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(group_norm_groups(32), 32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(group_norm_groups(64), 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(group_norm_groups(embed_dim), embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.conv(x)).squeeze(-1)


class DenseGraphAttention(nn.Module):
    def __init__(self, dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        batch_size, num_nodes, _ = x.shape
        query = self.query(x).view(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(x).view(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(x).view(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores.masked_fill(adjacency.unsqueeze(1) <= 0, -1e4)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        message = torch.matmul(attention, value).permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        return F.elu(self.norm(self.output(message) + x))


class BeatGraphBlock(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.layer1 = DenseGraphAttention(dim=dim, heads=heads)
        self.layer2 = DenseGraphAttention(dim=dim, heads=heads)

    def forward(self, x, adjacency):
        return self.layer2(self.layer1(x, adjacency), adjacency)


class ResidualBPHead(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, out_dim)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        residual = hidden
        hidden = self.fc2(hidden)
        hidden = self.norm(hidden + residual)
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc3(hidden))
        return self.fc4(hidden)


def batch_graph_density(adjacency, beat_mask):
    valid_pairs = beat_mask.unsqueeze(1) & beat_mask.unsqueeze(2)
    edges = (adjacency > 0) & valid_pairs
    return edges.float().sum(dim=(1, 2)) / valid_pairs.float().sum(dim=(1, 2)).clamp(min=1.0)


def build_adjacency(beat_embeddings, beat_mask, morph_threshold):
    batch_size, num_beats, _ = beat_embeddings.shape
    device = beat_embeddings.device
    adjacency = torch.zeros(batch_size, num_beats, num_beats, device=device, dtype=beat_embeddings.dtype)
    identity = torch.eye(num_beats, device=device, dtype=beat_embeddings.dtype).unsqueeze(0)
    adjacency = torch.maximum(adjacency, identity)

    temporal = torch.zeros(num_beats, num_beats, device=device, dtype=beat_embeddings.dtype)
    index = torch.arange(num_beats - 1, device=device)
    temporal[index, index + 1] = 1.0
    temporal[index + 1, index] = 1.0
    adjacency = torch.maximum(adjacency, temporal.unsqueeze(0))

    normalized = F.normalize(beat_embeddings, dim=-1, eps=1e-6)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    threshold = morph_threshold if torch.is_tensor(morph_threshold) else torch.tensor(morph_threshold, device=device)
    morph_edges = similarity > threshold
    morph_edges = morph_edges & ~torch.eye(num_beats, dtype=torch.bool, device=device).unsqueeze(0)
    adjacency = torch.where(morph_edges, torch.ones_like(adjacency), adjacency)

    invalid = ~beat_mask
    adjacency = adjacency.masked_fill(invalid.unsqueeze(-1) | invalid.unsqueeze(-2), 0.0)
    return adjacency


def splat_graph_context(beat_tokens, beat_mask, beat_boundaries, window_len=1248):
    batch_size, max_beats, embed_dim = beat_tokens.shape
    device = beat_tokens.device

    beat_index_map = torch.full((batch_size, window_len), -1, dtype=torch.long, device=device)
    for batch_index in range(batch_size):
        for beat_index, (start, end) in enumerate(beat_boundaries[batch_index]):
            if beat_index >= max_beats or not bool(beat_mask[batch_index, beat_index]):
                break
            start = int(start)
            end = min(int(end), window_len)
            if end <= start:
                continue
            beat_index_map[batch_index, start:end] = beat_index

    valid_mask = beat_index_map >= 0
    beat_index_map = beat_index_map.clamp(min=0)

    index_expanded = beat_index_map.unsqueeze(-1).expand(batch_size, window_len, embed_dim)
    context = torch.gather(beat_tokens, 1, index_expanded)
    context = context * valid_mask.unsqueeze(-1).float()
    return context.permute(0, 2, 1)


class BPMGraphUNet(nn.Module):
    beat_len = 128
    embed_dim = 64

    def __init__(
        self,
        in_channels=3,
        out_dim=2,
        morph_threshold=0.95,
        ppg_channel_index=0,
        beat_channel_indices=None,
        use_se=True,
        use_attention_gates=True,
        use_residual_head=True,
    ):
        super().__init__()
        self.ppg_channel_index = ppg_channel_index
        self.beat_channel_indices = tuple(
            beat_channel_indices if beat_channel_indices is not None else (ppg_channel_index,)
        )
        self.use_attention_gates = use_attention_gates
        self.morph_threshold = nn.Parameter(torch.tensor(float(morph_threshold), dtype=torch.float32))

        self.encoder1 = ConvBlock1D(in_channels, 32, kernel_size=7, use_se=use_se)
        self.encoder2 = ConvBlock1D(32, 64, kernel_size=7, use_se=use_se)
        self.encoder3 = ConvBlock1D(64, 128, kernel_size=5, use_se=use_se)
        self.encoder4 = ConvBlock1D(128, 256, kernel_size=5, use_se=use_se)
        self.pool = nn.MaxPool1d(2)

        self.beat_encoder = BeatEncoder(in_channels=len(self.beat_channel_indices), embed_dim=self.embed_dim)
        self.beat_feature_projection = nn.Linear(4, self.embed_dim)
        self.beat_graph = BeatGraphBlock(dim=self.embed_dim, heads=4)
        self.graph_pool_attention = nn.Linear(self.embed_dim, 1)
        self.graph_fusion = nn.Conv1d(256 + self.embed_dim, 256, kernel_size=1)

        if use_attention_gates:
            self.ag3 = AttentionGate1D(256, 128)
            self.ag2 = AttentionGate1D(128, 64)
            self.ag1 = AttentionGate1D(64, 32)

        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock1D(128 + 128, 128, kernel_size=5, use_se=use_se)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock1D(64 + 64, 64, kernel_size=7, use_se=use_se)
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock1D(32 + 32, 32, kernel_size=7, use_se=use_se)

        self.demographics_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.waveform_head = nn.Conv1d(32, 1, kernel_size=1)
        global_dim = 256 + 64 + self.embed_dim + 32
        if use_residual_head:
            self.blood_pressure_head = ResidualBPHead(global_dim, out_dim=out_dim)
        else:
            self.blood_pressure_head = nn.Sequential(
                nn.Linear(global_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, out_dim),
            )
        self.graph_density = 0.0
        self.apply(init_module_weights)

    def encode(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        bottleneck = skip4
        return bottleneck, (skip3, skip2, skip1)

    def decode(self, bottleneck, skips, graph_context):
        graph_context_down = F.adaptive_avg_pool1d(graph_context, bottleneck.shape[-1])
        x = self.graph_fusion(torch.cat([bottleneck, graph_context_down], dim=1))
        skip3, skip2, skip1 = skips

        def upsample_concat(up_layer, features, skip):
            upsampled = up_layer(features)
            if upsampled.shape[-1] != skip.shape[-1]:
                upsampled = F.interpolate(upsampled, size=skip.shape[-1], mode="linear", align_corners=False)
            return torch.cat([upsampled, skip], dim=1)

        if self.use_attention_gates:
            skip3 = self.ag3(x, skip3)
        x = self.decoder3(upsample_concat(self.up3, x, skip3))
        if self.use_attention_gates:
            skip2 = self.ag2(x, skip2)
        x = self.decoder2(upsample_concat(self.up2, x, skip2))
        if self.use_attention_gates:
            skip1 = self.ag1(x, skip1)
        return self.decoder1(upsample_concat(self.up1, x, skip1))

    def graph_context(self, x, beat_boundaries, beat_features, beat_inputs, beat_mask):
        beat_inputs = beat_inputs[:, :, list(self.beat_channel_indices), :]
        batch_size, max_beats, channels, beat_len = beat_inputs.shape
        flat_beats = beat_inputs.reshape(batch_size * max_beats, channels, beat_len)
        beat_embeddings = self.beat_encoder(flat_beats).view(batch_size, max_beats, self.embed_dim)
        projected_features = self.beat_feature_projection(beat_features)
        beat_embeddings = beat_embeddings + projected_features * beat_mask.unsqueeze(-1).to(projected_features.dtype)

        adjacency = build_adjacency(beat_embeddings, beat_mask, self.morph_threshold)
        beat_tokens = self.beat_graph(beat_embeddings, adjacency)

        pool_logits = self.graph_pool_attention(beat_tokens).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~beat_mask, -1e4)
        pool_weights = torch.softmax(pool_logits, dim=1)
        graph_pool = torch.sum(beat_tokens * pool_weights.unsqueeze(-1), dim=1)

        graph_context = splat_graph_context(
            beat_tokens,
            beat_mask,
            beat_boundaries,
            window_len=x.size(-1),
        )
        self.graph_density = float(batch_graph_density(adjacency, beat_mask).mean().detach().cpu())
        return graph_context, graph_pool

    def forward(self, x, beat_boundaries, beat_features, beat_inputs, beat_mask, demographics):
        bottleneck, skips = self.encode(x)
        graph_context, graph_pool = self.graph_context(
            x,
            beat_boundaries,
            beat_features,
            beat_inputs,
            beat_mask,
        )
        decoded = self.decode(bottleneck, skips, graph_context)
        encoder_pool = bottleneck.mean(dim=-1)
        decoder_pool = torch.cat([decoded.mean(dim=-1), decoded.max(dim=-1).values], dim=1)
        demographics_embed = self.demographics_encoder(demographics)
        global_features = torch.cat([encoder_pool, decoder_pool, graph_pool, demographics_embed], dim=1)
        blood_pressure = self.blood_pressure_head(global_features)
        waveform = self.waveform_head(decoded).squeeze(1)
        return blood_pressure, waveform


def build_model(
    model_type,
    in_channels,
    ppg_channel_index=0,
    beat_channel_indices=None,
    out_dim=2,
    legacy=False,
):
    if model_type != "bpm_graph_unet":
        raise KeyError(f"Unknown model type: {model_type}")
    return BPMGraphUNet(
        in_channels=in_channels,
        out_dim=out_dim,
        ppg_channel_index=ppg_channel_index,
        beat_channel_indices=beat_channel_indices,
        use_se=not legacy,
        use_attention_gates=not legacy,
        use_residual_head=not legacy,
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
