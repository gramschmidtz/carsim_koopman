import torch
from torch.utils.data import DataLoader
from src.dataset import CarSimDataset
from src.network import DeepEDMD
from src.loss import calculate_loss
import os
from tqdm import tqdm

def main():
    # 설정
    DATA_DIR = "./data" # .mat 파일 경로
    EPOCHS = 50
    BATCH_SIZE = 64
    P_WINDOW = 20 # Multi-step prediction steps
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        total_loss = 0
        pred_err = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x_seq, u_seq in pbar:
            x_seq = x_seq.to(DEVICE)
            u_seq = u_seq.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Loss 계산
            loss, p_err = calculate_loss(model, x_seq, u_seq, P_WINDOW)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred_err += p_err
            
            pbar.set_postfix({'Loss': loss.item(), 'PredMSE': p_err})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")
        
    # 4. 모델 저장
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    # 정규화 통계량도 같이 저장해야 나중에 복원 가능
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