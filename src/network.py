import torch
import torch.nn as nn

class DeepEDMD(nn.Module):
    def __init__(self, state_dim=3, input_dim=3, latent_dim=16, hidden_layers=[32, 64]):
        super(DeepEDMD, self).__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        # 논문: [x, Output_NN] 형태 (Lifted state에 원래 state 포함)
        # 신경망 출력 차원 = L - n (전체 차원 - 원래 차원)
        net_out_dim = latent_dim - state_dim
        
        enc_layers = []
        in_dim = state_dim
        for h in hidden_layers:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, net_out_dim))
        self.encoder_net = nn.Sequential(*enc_layers)
        
        # --- Decoder ---
        dec_layers = []
        in_dim = latent_dim
        # Decoder 구조: L -> 128 -> 64 -> n
        dec_hidden = [128, 64] 
        for h in dec_hidden:
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, state_dim))
        self.decoder_net = nn.Sequential(*dec_layers)
        
        # --- Koopman Matrix (Learnable) ---
        # A: (L, L), B: (L, m)
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(input_dim, latent_dim, bias=False)
        
        # 초기화: A는 단위행렬 근처, B는 작은 값으로 시작하면 안정적
        self.A.weight.data = torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)

    def encode(self, x):
        # x: (Batch, state_dim)
        psi_nn = self.encoder_net(x)      # (Batch, L-n)
        psi = torch.cat([x, psi_nn], dim=1) # (Batch, L)
        return psi

    def decode(self, psi):
        # psi: (Batch, L)
        return self.decoder_net(psi)

    def forward(self, x, u):
        # One-step prediction (학습 루프에서는 직접 A, B 사용하므로 잘 안씀)
        psi = self.encode(x)
        psi_next = self.A(psi) + self.B(u)
        x_next = self.decode(psi_next)
        return x_next