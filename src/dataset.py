import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset

class CarSimDataset(Dataset):
    def __init__(self, data_dir, p_window=40, mode='train'):
        self.p = p_window
        self.data_list = []
        
        # 파일 로드 (1~40번)
        # Train: 1~35, Test: 36~40 (논문 비율과 유사하게 설정)
        file_indices = range(1, 36) if mode == 'train' else range(36, 41)
        
        print(f"Loading {mode} data from {data_dir}...")

        all_X = []
        all_U = []

        for i in file_indices:
            file_name = f"carsim_v6_{i}.mat"
            file_path = os.path.join(data_dir, file_name)
            
            if not os.path.exists(file_path):
                continue
            
            try:
                mat = scipy.io.loadmat(file_path)
                # 데이터 키 찾기 (보통 'data' 또는 파일명과 유사함. 여기서는 values만 가져옴)
                # mat 파일 구조에 따라 수정 필요할 수 있음. 보통 list(mat.keys())[-1] 사용
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                raw = mat[key] # (Time, 30)
                
                # --- [전처리] 단위 변환 ---
                # Readme 기준:
                # 5: Local Vx (km/h) -> idx 4
                # 6: Local Vy (km/h) -> idx 5
                # 7: Yaw Rate (deg/s) -> idx 6
                # 28: Steer (deg) -> idx 27
                # 29: Throttle (0-1) -> idx 28
                # 30: Brake (MPa) -> idx 29
                
                # State: [vx, vy, r] (m/s, m/s, rad/s)
                vx = raw[:, 4] / 3.6
                vy = raw[:, 5] / 3.6
                r  = np.deg2rad(raw[:, 6])
                
                # Input: [steer, throttle, brake] (rad, -, -)
                steer = np.deg2rad(raw[:, 27])
                th = raw[:, 28]
                br = raw[:, 29]
                
                X = np.stack([vx, vy, r], axis=1).astype(np.float32)
                U = np.stack([steer, th, br], axis=1).astype(np.float32)
                
                all_X.append(X)
                all_U.append(U)
                
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        # 정규화 (Normalization) - 전체 데이터 기준 통계량 산출
        # 실제로는 Train 데이터 통계만으로 Test도 정규화해야 함 (간략화됨)
        self.X_mean, self.X_std = self._get_stats(all_X)
        self.U_mean, self.U_std = self._get_stats(all_U)
        
        self.samples = []
        for X, U in zip(all_X, all_U):
            # Z-score Normalization
            X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
            U_norm = (U - self.U_mean) / (self.U_std + 1e-6)
            
            # 윈도우 슬라이싱
            n_steps = X.shape[0]
            for t in range(0, n_steps - self.p - 1, 5): # Stride 5로 데이터 생성
                x_seq = X_norm[t : t + self.p + 1]
                u_seq = U_norm[t : t + self.p]
                self.samples.append((x_seq, u_seq))

    def _get_stats(self, data_list):
        concat = np.concatenate(data_list, axis=0)
        return concat.mean(axis=0), concat.std(axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, u_seq = self.samples[idx]
        return torch.from_numpy(x_seq), torch.from_numpy(u_seq)