class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.expand = nn.Linear(d_model, d_ff)
        self.contract = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.contract(torch.relu(self.expand(x)))
