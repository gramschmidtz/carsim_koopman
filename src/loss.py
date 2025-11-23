import torch
import torch.nn.functional as F

def calculate_loss(model, x_seq, u_seq, p_steps, alphas):
    """
    x_seq: (Batch, p+1, n)
    u_seq: (Batch, p, m)
    alphas: Dictionary containing weights (alpha1 ~ alpha6)
    """
    batch_size = x_seq.shape[0]
    
    # 1. 초기 상태 인코딩
    x0 = x_seq[:, 0, :]
    psi = model.encode(x0) # (Batch, L)
    
    loss_pred = 0.0  # L_x_x
    loss_lin = 0.0   # L_x_o
    loss_inf = 0.0   # L_inf (New)
    
    # Reconstruction Loss (L_o_x)
    x0_hat = model.decode(psi)
    loss_recon = F.mse_loss(x0_hat, x0)
    
    # p-step rollout
    for t in range(p_steps):
        u_t = u_seq[:, t, :]
        x_target = x_seq[:, t+1, :]
        
        # --- Linear Dynamics in Latent Space ---
        # psi_{k+1} = A * psi_k + B * u_k
        psi_next_lin = model.A(psi) + model.B(u_t)
        
        # --- L_x_o: Latent Consistency ---
        # 실제 다음 state를 인코딩한 값 vs 선형 예측 값 비교
        psi_target = model.encode(x_target)
        loss_lin += F.mse_loss(psi_next_lin, psi_target)
        
        # --- Prediction in Physical Space ---
        x_next_pred = model.decode(psi_next_lin)
        
        # --- L_x_x: Prediction MSE ---
        loss_pred += F.mse_loss(x_next_pred, x_target)
        
        # --- L_inf: Infinity Norm Loss (New) ---
        # 논문 Eq (18): 각 배치의 오차 절댓값 중 최댓값의 평균
        # (Batch, n) -> (Batch,) -> Scalar Mean
        error_inf = torch.max(torch.abs(x_next_pred - x_target), dim=1)[0].mean()
        loss_inf += error_inf
        
        # 다음 스텝 준비
        psi = psi_next_lin

    # 평균 계산 (p steps)
    loss_pred /= p_steps
    loss_lin /= p_steps
    loss_inf /= p_steps
    
    # --- L2 Regularization (New) ---
    # 논문 Eq (20): alpha5 * ||theta_e||^2 + alpha6 * ||theta_d||^2
    l2_enc = sum(p.pow(2.0).sum() for p in model.encoder_net.parameters())
    l2_dec = sum(p.pow(2.0).sum() for p in model.decoder_net.parameters())
    
    # --- Total Weighted Loss ---
    total_loss = (alphas['alpha1'] * loss_recon +
                  alphas['alpha2'] * loss_pred +
                  alphas['alpha3'] * loss_lin +
                  alphas['alpha4'] * loss_inf +
                  alphas['alpha5'] * l2_enc +
                  alphas['alpha6'] * l2_dec)
    
    return total_loss, loss_pred.item()