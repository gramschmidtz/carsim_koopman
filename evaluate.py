import torch
import matplotlib.pyplot as plt
import numpy as np
from src.network import DeepEDMD
from src.dataset import CarSimDataset
import os

def evaluate():
    DEVICE = torch.device("cpu") # 검증은 CPU로도 충분
    DATA_DIR = "./data"
    
    # 1. 모델 및 통계량 로드 (보안 옵션 해제 포함)
    checkpoint = torch.load("saved_models/nndmd_carsim.pt", map_location=DEVICE, weights_only=False)
    
    stats = checkpoint['stats']
    X_mean, X_std = stats['X_mean'], stats['X_std']
    
    model = DeepEDMD(state_dim=3, input_dim=3, latent_dim=16).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 2. 테스트 데이터 로드
    print(f"Loading test data from {DATA_DIR}...")
    test_dataset = CarSimDataset(DATA_DIR, p_window=100, mode='test') 
    
    # 랜덤한 샘플 하나 추출
    if len(test_dataset) == 0:
        print("Test dataset is empty. Check data directory or indices.")
        return

    idx = np.random.randint(0, len(test_dataset))
    x_seq, u_seq = test_dataset[idx] 
    
    # 배치 차원 추가: (T, 3) -> (1, T, 3)
    x_seq = x_seq.unsqueeze(0).to(DEVICE) 
    u_seq = u_seq.unsqueeze(0).to(DEVICE) 
    
    # 3. Prediction Rollout
    pred_traj = []
    
    with torch.no_grad():
        # 초기 상태 인코딩 (t=0)
        psi = model.encode(x_seq[:, 0, :])
        pred_traj.append(model.decode(psi)) 
        
        steps = u_seq.shape[1]
        for t in range(steps):
            u_t = u_seq[:, t, :]
            # Linear Evolution
            psi = model.A(psi) + model.B(u_t)
            x_next = model.decode(psi)
            pred_traj.append(x_next)
            
    # [수정된 부분] 리스트를 Tensor로 합치기
    # 결과 Shape 예상: (T, 3)
    pred_traj = torch.cat(pred_traj, dim=0).cpu().numpy()
    
    # 정답 데이터 (배치 차원 제거)
    true_traj = x_seq.squeeze(0).cpu().numpy()
    
    # 4. 역정규화 (Denormalization)
    # X_std, X_mean은 (3,) 형태이므로 Broadcasting 되어 계산됨
    pred_traj = pred_traj * X_std + X_mean
    true_traj = true_traj * X_std + X_mean
    
    # 5. 시각화
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['Vx (m/s)', 'Vy (m/s)', 'Yaw Rate (rad/s)']
    
    # 시간 축 (Steps)
    t_axis = np.arange(pred_traj.shape[0])

    for i in range(3):
        axs[i].plot(t_axis, true_traj[:, i], 'k-', label='Ground Truth', linewidth=2)
        axs[i].plot(t_axis, pred_traj[:, i], 'r--', label='NNDMD Prediction', linewidth=2)
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
        
    plt.suptitle(f"CarSim Validation (Sample #{idx})")
    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()