import torch
import matplotlib.pyplot as plt
import numpy as np
from src.network import DeepEDMD
from src.dataset import CarSimDataset
import os

def evaluate():
    DEVICE = torch.device("cpu") # 검증은 CPU로도 충분
    DATA_DIR = "./data"
    
    # 1. 모델 및 통계량 로드
    checkpoint = torch.load("saved_models/nndmd_carsim.pt", map_location=DEVICE)
    
    stats = checkpoint['stats']
    X_mean, X_std = stats['X_mean'], stats['X_std']
    
    model = DeepEDMD(state_dim=3, input_dim=3, latent_dim=16).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 2. 테스트 데이터 로드 (하나의 긴 궤적 가져오기 위해 직접 로드)
    # dataset 클래스 활용하되, window 없이 로드하거나 첫번째 샘플 길게 가져옴
    test_dataset = CarSimDataset(DATA_DIR, p_window=100, mode='test') 
    
    # 랜덤한 샘플 하나 추출
    idx = np.random.randint(0, len(test_dataset))
    x_seq, u_seq = test_dataset[idx] # (101, 3), (100, 3) (Normalized)
    
    x_seq = x_seq.unsqueeze(0).to(DEVICE) # (1, T, 3)
    u_seq = u_seq.unsqueeze(0).to(DEVICE) # (1, T, 3)
    
    # 3. Prediction Rollout
    pred_traj = []
    
    with torch.no_grad():
        # 초기 상태 인코딩
        psi = model.encode(x_seq[:, 0, :])
        pred_traj.append(model.decode(psi)) # t=0
        
        steps = u_seq.shape[1]
        for t in range(steps):
            u_t = u_seq[:, t, :]
            psi = model.A(psi) + model.B(u_t)
            x_next = model.decode(psi)
            pred_traj.append(x_next)
            
    pred_traj = torch.cat(pred_traj, dim=1).squeeze(0).numpy() # (T, 3)
    true_traj = x_seq.squeeze(0).numpy()
    
    # 4. 역정규화 (Denormalization) -> 원래 단위로 변환
    pred_traj = pred_traj * X_std + X_mean
    true_traj = true_traj * X_std + X_mean
    
    # 5. 시각화
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['Vx (m/s)', 'Vy (m/s)', 'Yaw Rate (rad/s)']
    
    for i in range(3):
        axs[i].plot(true_traj[:, i], 'k-', label='Ground Truth')
        axs[i].plot(pred_traj[:, i], 'r--', label='NNDMD Prediction')
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)
        
    plt.suptitle("CarSim Validation Result (Open-loop Prediction)")
    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()