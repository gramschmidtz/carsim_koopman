import torch
import torch.nn.functional as F

def calculate_loss(model, x_seq, u_seq, p_steps, alphas):
    """
    x_seq: (Batch, p+1, n)
    u_seq: (Batch, p, m)
    alphas: Dictionary containing weights (alpha1 ~ alpha6)
    """
    
    # 1. 초기 상태 인코딩 (Prediction을 위한 시작점)
    # 논문 표기상 Psi_0에 해당
    x0 = x_seq[:, 0, :]
    psi = model.encode(x0) # (Batch, L)
    
    loss_pred = 0.0  # L_x_x (Eq 15)
    loss_lin = 0.0   # L_x_o (Eq 16)
    loss_recon = 0.0 # L_o_x (Eq 17) - 수정됨: 루프 내 누적
    loss_inf = 0.0   # L_inf (Eq 18) - 수정됨: Recon 항 추가
    
    # p-step rollout
    for t in range(p_steps):
        u_t = u_seq[:, t, :]       # 현재 입력
        x_target = x_seq[:, t+1, :] # 다음 실제 상태 (Target)
        
        # --- [A] Prediction Stream (Linear Evolution) ---
        # psi_{k+1} = A * psi_k + B * u_k
        psi_next_lin = model.A(psi) + model.B(u_t)
        
        # x_{k+1}_pred = Decoder(psi_{k+1})
        x_next_pred = model.decode(psi_next_lin)
        
        # --- [B] Reconstruction Stream ---
        # 실제 다음 상태(x_target)를 인코딩 -> 디코딩해서 잘 복원되는지 확인
        psi_target = model.encode(x_target)      # Target의 Latent
        x_target_recon = model.decode(psi_target) # Target의 복원값
        
        # ====================================================
        # Loss Calculation (논문 수식 매핑)
        # ====================================================
        
        # 1. L_x_x (Eq 15): Prediction MSE
        # 예측된 상태 vs 실제 상태
        loss_pred += F.mse_loss(x_next_pred, x_target)
        
        # 2. L_x_o (Eq 16): Latent Consistency
        # 선형 예측된 Latent vs 실제 Latent
        loss_lin += F.mse_loss(psi_next_lin, psi_target)
        
        # 3. L_o_x (Eq 17): Reconstruction MSE [수정됨]
        # 타겟의 복원값 vs 실제 타겟 (전체 시퀀스에 대해 수행)
        loss_recon += F.mse_loss(x_target_recon, x_target)
        
        # 4. L_inf (Eq 18): Infinity Norm Loss [수정됨]
        # Part 1: Reconstruction Inf Error ( || x - Recon(x) ||_inf )
        inf_recon = torch.max(torch.abs(x_target - x_target_recon), dim=1)[0].mean()
        
        # Part 2: Prediction Inf Error ( || x - Pred(x) ||_inf )
        inf_pred = torch.max(torch.abs(x_target - x_next_pred), dim=1)[0].mean()
        
        loss_inf += (inf_recon + inf_pred)
        
        # 다음 스텝 준비
        psi = psi_next_lin

    # 평균 계산 (p steps)
    loss_pred /= p_steps
    loss_lin /= p_steps
    loss_recon /= p_steps # [수정됨] 이제 0이 아님
    loss_inf /= p_steps
    
    # --- L2 Regularization (Eq 20) ---
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