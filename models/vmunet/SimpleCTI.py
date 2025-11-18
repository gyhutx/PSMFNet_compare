class SimpleCTI(nn.Module):
    """
    简化版 Cross-scale Token Interaction
    输入两个不同分辨率特征，自动flatten，做token级融合与交互。
    """

    def __init__(self, in_channels, attn_heads=4, fusion_hidden=None, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.attn_heads = attn_heads
        hidden = fusion_hidden or in_channels

        # Token交互: 多头自注意力
        self.token_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=attn_heads, batch_first=True)

        # 通道融合: 1x1卷积
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)

        if use_cnn:
            # 可选：CNN局部增强
            self.cnn_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x_main, x_aux):
        """
        x_main: [B, C, H, W]
        x_aux:  [B, C, H, W] 或 [B, C, H', W']
        """
        B, C, H, W = x_main.shape

        # 若x_aux与x_main分辨率不同，resize到一致
        if x_aux.shape[2:] != (H, W):
            x_aux = F.interpolate(x_aux, size=(H, W), mode='bilinear', align_corners=False)

        # flatten到token序列
        x1_token = x_main.flatten(2).transpose(1, 2)  # [B, N, C]
        x2_token = x_aux.flatten(2).transpose(1, 2)   # [B, N, C]

        # 拼接两个token序列，做self-attention交互
        tokens = torch.cat([x1_token, x2_token], dim=1)  # [B, 2N, C]
        attn_out, _ = self.token_attn(tokens, tokens, tokens)  # [B, 2N, C]

        # 拆分还原
        out1 = attn_out[:, :H*W, :].transpose(1, 2).view(B, C, H, W)
        out2 = attn_out[:, H*W:, :].transpose(1, 2).view(B, C, H, W)

        # 拼接+通道融合
        fused = torch.cat([out1, out2], dim=1)  # [B, 2C, H, W]
        fused = self.fusion_conv(fused)         # [B, C, H, W]

        # 可选：CNN增强
        if self.use_cnn:
            fused = self.cnn_block(fused)

        return fused