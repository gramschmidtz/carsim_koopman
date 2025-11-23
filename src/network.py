import torch
import torch.nn as nn

class DeepEDMD(nn.Module):
    def __init__(self, state_dim=3, input_dim=3, latent_dim=16):
        super(DeepEDMD, self).__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        # 논문 텍스트: "structure chosen as [n 32 64 L-n L-n]"
        # 즉, Input(n) -> 32 -> 64 -> (L-n) -> Output(L-n)
        
        net_out_dim = latent_dim - state_dim
        
        self.encoder_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, net_out_dim), 
            nn.ReLU(),
            nn.Linear(net_out_dim, net_out_dim) # 마지막 층 (Activation 없음 or 논문 구현에 따라 다름)
            # 논문에서 마지막 층 Activation 언급이 없으면 보통 Linear입니다.
            # 하지만 [L-n, L-n] 표기는 층이 하나 더 있다는 뜻이므로 Linear를 추가했습니다.
        )
        
        # --- Decoder ---
        # 논문 텍스트: "structure of the decoder was set as [L 128 64 32 n]"
        # 즉, Input(L) -> 128 -> 64 -> 32 -> Output(n)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        
        # --- Koopman Matrix (Learnable) ---
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(input_dim, latent_dim, bias=False)
        
        # 가중치 초기화 (논문 내용 반영: uniform distribution [-w, w])
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 논문: "weights were initialized with a uniform distribution ... [-w, w] for w = 1/sqrt(a)"
                # a is the number of inputs to the layer (fan_in)
                input_dim = m.weight.size(1)
                w = 1.0 / (input_dim ** 0.5)
                nn.init.uniform_(m.weight, -w, w)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -w, w)

    def encode(self, x):
        psi_nn = self.encoder_net(x)      
        psi = torch.cat([x, psi_nn], dim=1) 
        return psi

    def decode(self, psi):
        return self.decoder_net(psi)

    def forward(self, x, u):
        psi = self.encode(x)
        psi_next = self.A(psi) + self.B(u)
        x_next = self.decode(psi_next)
        return x_next