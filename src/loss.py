import torch
import torch.nn.functional as F

def calculate_loss(model, x_seq, u_seq, p_steps):
    """
    x_seq: (Batch, p+1, n)
    u_seq: (Batch, p, m)
    """
    batch_size = x_seq.shape[0]
    
    # 1. 초기 상태 인코딩
    x0 = x_seq[:, 0, :]
    psi = model.encode(x0) # (Batch, L)
    
    loss_pred = 0.0
    loss_recon = 0.0
    loss_lin = 0.0
    
    # Reconstruction Loss (t=0)
    x0_hat = model.decode(psi)
    loss_recon += F.mse_loss(x0_hat, x0)
    
    # p-step rollout
    for t in range(p_steps):
        u_t = u_seq[:, t, :]
        x_target = x_seq[:, t+1, :]
        
        # Linear Dynamics in Latent Space
        # psi_{k+1} = A * psi_k + B * u_k
        psi_next_lin = model.A(psi) + model.B(u_t)
        
        # 2. Linear Evolution Loss (Latent Space consistency)
        # 실제 다음 state를 인코딩한 것과 선형 예측한 것의 차이
        psi_target = model.encode(x_target)
        loss_lin += F.mse_loss(psi_next_lin, psi_target)
        
        # 3. Prediction Loss (Physical Space)
        x_next_pred = model.decode(psi_next_lin)
        loss_pred += F.mse_loss(x_next_pred, x_target)
        
        # 다음 스텝 준비
        psi = psi_next_lin

    # 평균
    loss_pred /= p_steps
    loss_lin /= p_steps
    
    # 논문 가중치 (alpha) - 예시값
    total_loss = 1.0 * loss_pred + 0.3 * loss_lin + 1.0 * loss_recon
    
    return total_loss, loss_pred.item()