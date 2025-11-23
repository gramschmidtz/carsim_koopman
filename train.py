import torch
from torch.utils.data import DataLoader
from src.dataset import CarSimDataset
from src.network import DeepEDMD
from src.loss import calculate_loss
import os
from tqdm import tqdm

def main():
    # 설정
    DATA_DIR = "./data" 
    EPOCHS = 50
    BATCH_SIZE = 64
    P_WINDOW = 41
    LR = 1e-3 # 논문 Table II: 10^-4지만 학습 속도 위해 조정 가능
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 논문 Table II 기반 가중치 설정
    ALPHAS = {
        "alpha1": 1.0,   # Reconstruction (L_o_x)
        "alpha2": 1.0,   # Prediction (L_x_x)
        "alpha3": 0.3,   # Linearity (L_x_o)
        "alpha4": 1e-9,  # Infinity Norm (L_inf)
        "alpha5": 1e-9,  # Encoder L2 (Weight Decay)
        "alpha6": 1e-9   # Decoder L2 (Weight Decay)
    }
    
    # 1. 데이터셋 준비
    train_dataset = CarSimDataset(DATA_DIR, p_window=P_WINDOW, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Train data size: {len(train_dataset)}")
    
    # 2. 모델 초기화
    model = DeepEDMD(state_dim=3, input_dim=3, latent_dim=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 3. 학습 루프
    model.train()
    for epoch in range(EPOCHS):
        total_loss_sum = 0
        pred_err_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x_seq, u_seq in pbar:
            x_seq = x_seq.to(DEVICE)
            u_seq = u_seq.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 수정된 calculate_loss 호출 (ALPHAS 전달)
            loss, p_err = calculate_loss(model, x_seq, u_seq, P_WINDOW, ALPHAS)
            
            loss.backward()
            optimizer.step()
            
            total_loss_sum += loss.item()
            pred_err_sum += p_err
            
            pbar.set_postfix({'TotalLoss': loss.item(), 'PredMSE': p_err})
            
        avg_loss = total_loss_sum / len(train_loader)
        print(f"Epoch {epoch+1} Done. Avg Total Loss: {avg_loss:.6f}")
        
    # 4. 모델 저장
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    save_dict = {
        'model_state': model.state_dict(),
        'stats': {
            'X_mean': train_dataset.X_mean, 'X_std': train_dataset.X_std,
            'U_mean': train_dataset.U_mean, 'U_std': train_dataset.U_std
        }
    }
    torch.save(save_dict, "saved_models/nndmd_carsim.pt")
    print("Model saved to saved_models/nndmd_carsim.pt")

if __name__ == "__main__":
    main()