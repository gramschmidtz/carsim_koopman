import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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

    # idx = np.random.randint(0, len(test_dataset))
    idx = 15684
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

def evaluate_top_k_samples(top_k=10):
    DEVICE = torch.device("cpu")
    DATA_DIR = "./data"
    
    # 1. 모델 및 통계량 로드 (보안 경고 우회 옵션 포함)
    model_path = os.path.join("saved_models", "nndmd_carsim.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    stats = checkpoint['stats']
    X_mean, X_std = stats['X_mean'], stats['X_std']
    
    model = DeepEDMD(state_dim=3, input_dim=3, latent_dim=16).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 2. 테스트 데이터 로드
    print(f"Loading test data from {DATA_DIR}...")
    test_dataset = CarSimDataset(DATA_DIR, p_window=100, mode='test') 
    
    if len(test_dataset) == 0:
        print("Test dataset is empty.")
        return

    print(f"Total Test Samples: {len(test_dataset)}")
    print(f"Calculating Error for all samples to find Top {top_k}...")

    # 모든 샘플의 변수별 MSE를 저장할 리스트
    # shape: (Num_Samples, 3) -> [[mse_vx, mse_vy, mse_r], ...]
    all_mse_per_state = [] 
    
    # 3. 전체 데이터 순회하며 Error 계산
    for idx in tqdm(range(len(test_dataset))):
        x_seq, u_seq = test_dataset[idx] 
        
        x_seq = x_seq.unsqueeze(0).to(DEVICE) # (1, T, 3)
        u_seq = u_seq.unsqueeze(0).to(DEVICE) # (1, T, 3)
        
        # Rollout Prediction
        pred_traj = []
        with torch.no_grad():
            psi = model.encode(x_seq[:, 0, :])
            pred_traj.append(model.decode(psi)) 
            
            steps = u_seq.shape[1]
            for t in range(steps):
                u_t = u_seq[:, t, :]
                psi = model.A(psi) + model.B(u_t)
                x_next = model.decode(psi)
                pred_traj.append(x_next)
        
        pred_traj_tensor = torch.cat(pred_traj, dim=0) # (T, 3)
        true_traj_tensor = x_seq.squeeze(0)            # (T, 3)
        
        # 변수별 MSE 계산 (Time 축으로 평균) -> (3,)
        mse_per_dim = torch.mean((pred_traj_tensor - true_traj_tensor) ** 2, dim=0).cpu().numpy()
        all_mse_per_state.append(mse_per_dim)

    # 4. 통계 계산 (MSE -> RMSE 변환 후 통계 산출)
    # (N, 3) 배열로 변환
    all_mse_array = np.array(all_mse_per_state) 
    
    # [중요] 먼저 모든 샘플을 RMSE 단위로 변환 (Element-wise sqrt)
    all_rmse_array = np.sqrt(all_mse_array) 
    
    # 그 다음, RMSE 데이터셋을 기준으로 통계 산출
    avg_rmse = np.mean(all_rmse_array, axis=0) # 평균
    std_rmse = np.std(all_rmse_array, axis=0)  # 표준편차 (RMSE 값들의 분산도)
    min_rmse = np.min(all_rmse_array, axis=0)  # 최소 오차
    max_rmse = np.max(all_rmse_array, axis=0)  # 최대 오차
    
    print("\n" + "="*80)
    print(f"📊 Test Set Statistics (RMSE Unit) per State Variable")
    print(f"{'Metric':<15} | {'Vx (m/s)':<18} | {'Vy (m/s)':<15} | {'Yaw Rate (rad/s)':<15}")
    print("-" * 80)
    print(f"{'Avg RMSE':<15} | {avg_rmse[0]:.6f}           | {avg_rmse[1]:.6f}        | {avg_rmse[2]:.6f}")
    print(f"{'Std Dev':<15} | {std_rmse[0]:.6f}           | {std_rmse[1]:.6f}        | {std_rmse[2]:.6f}")
    print(f"{'Min RMSE':<15} | {min_rmse[0]:.6f}           | {min_rmse[1]:.6f}        | {min_rmse[2]:.6f}")
    print(f"{'Max RMSE':<15} | {max_rmse[0]:.6f}           | {max_rmse[1]:.6f}        | {max_rmse[2]:.6f}")
    print("="*80 + "\n")

    # 5. 랭킹 산정 (Sample별 평균 RMSE 기준)
    ranking_results = []
    for idx, rmse_per_dim in enumerate(all_rmse_array):
        # 3개 변수의 평균 RMSE를 해당 샘플의 대표 점수로 사용
        sample_mean_rmse = np.mean(rmse_per_dim)
        ranking_results.append((sample_mean_rmse, idx))

    # 오름차순 정렬 (에러 작은 순)
    ranking_results.sort(key=lambda x: x[0])
    top_samples = ranking_results[:top_k]

    print(f"=== 🏆 Top {top_k} Best Samples (Lowest Avg RMSE) ===")
    print(f"{'Rank':<5} | {'Sample Index':<12} | {'RMSE':<10}")
    print("-" * 40)
    for rank, (error, sample_idx) in enumerate(top_samples):
        print(f"{rank+1:<5} | #{sample_idx:<11} | {error:.6f}")

    # 6. 가장 좋은 샘플(Rank 1) 시각화
    best_idx = top_samples[0][1]
    best_error = top_samples[0][0]
    
    print(f"\nVisualizing Rank 1 Sample: #{best_idx}")
    plot_sample(test_dataset, model, best_idx, best_error, X_mean, X_std, DEVICE)

def plot_sample(dataset, model, idx, error_val, X_mean, X_std, device):
    x_seq, u_seq = dataset[idx]
    x_seq = x_seq.unsqueeze(0).to(device)
    u_seq = u_seq.unsqueeze(0).to(device)

    pred_traj = []
    with torch.no_grad():
        psi = model.encode(x_seq[:, 0, :])
        pred_traj.append(model.decode(psi))
        for t in range(u_seq.shape[1]):
            psi = model.A(psi) + model.B(u_seq[:, t, :])
            pred_traj.append(model.decode(psi))
            
    pred_traj = torch.cat(pred_traj, dim=0).cpu().numpy()
    true_traj = x_seq.squeeze(0).cpu().numpy()
    
    # 역정규화 (Denormalization)
    pred_traj = pred_traj * X_std + X_mean
    true_traj = true_traj * X_std + X_mean
    
    # 시각화
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    labels = ['Vx (m/s)', 'Vy (m/s)', 'Yaw Rate (rad/s)']
    t_axis = np.arange(pred_traj.shape[0])

    for i in range(3):
        axs[i].plot(t_axis, true_traj[:, i], 'k-', label='Ground Truth', linewidth=2)
        axs[i].plot(t_axis, pred_traj[:, i], 'r--', label='NNDMD Prediction', linewidth=2)
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
        
    plt.suptitle(f"Best Validation Sample (Index #{idx})\nAvg RMSE across states: {error_val:.5f}")
    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # evaluate_top_k_samples(top_k=10)
    evaluate()