# I'm considering replacing the forward function for the PyraT-CAiT Stage with this forward function.
# Will have to see if it makes a difference.

def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Passes the input triplet through the i-th Stage of a PyraT-CAiT model.

    Inputs:
        anchor: a batch of anchor images.
        positive: a batch of positive images (similar to the anchor images).
        negative: a batch of negative images (dissimilar from the anchor images).

    Returns:
        z_anchor: the embedded anchor batch.
        z_positive: the embedded positive batch.
        z_negative: the embedded negative batch.
    '''
    
    anchor, positive, negative = self._embed(anchor, positive, negative)

    for block in self.transformer_stack:
        anchor = block(anchor)
        positive = block(positive)
        negative = block(negative)

    positive_ca = self.positive_cross_attn(anchor, positive)
    negative_ca = self.negative_cross_attn(anchor, negative)

    # assume additional learnable parameters "alpha" and "beta"

    anchor_pure = anchor.clone()
    anchor_mixed = anchor + alpha * positive_ca - beta * negative_ca # pull (anchor, positive) together, push (anchor, negative) apart

    positive = positive + alpha * positive_ca - beta * negative_ca # pull (positive, anchor) together, push (positive, negative) apart
    negative = negative - beta * (positive_ca + negative_ca) # push (negative, positive) and (negative, anchor) apart
    
    z_anchor = self._reshape_output(anchor_mixed) if not self.add_cls else anchor_pure

    z_positive = self._reshape_output(positive) if not self.add_cls else positive
    z_negative = self._reshape_output(negative) if not self.add_cls else negative

    return z_anchor, z_positive, z_negative