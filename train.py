import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import get_loaders
from model import get_resnet18
from utils import calculate_accuracy

#데이터 경로
data_dir = r"C:\Users\AMI-DEEP3\Desktop\chest\chest_xray"

#가중치 저장 경로
save_dir = "weights"

def train_model(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #하이퍼파라미터 설정
    epoch=10
    lr=1e-4
    batch_size=32

    train_loader, val_loader, _ = get_loaders(data_dir,batch_size)
    model = get_resnet18(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr)

    #학습, 검증 시각 그래프
    train_loss_list=[]
    val_loss_list=[]
    train_acc_list=[]
    val_acc_list=[]

    best_val_loss = float("inf")
    early_stop_counter = 0
    patience=3

    for epoch in range(epoch):
        model.train()
        train_loss, train_acc=0, 0

        tbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}]")
        for input, labels in tbar:
            input, labels = input.to(device), labels.to(device)

            outputs = model(input)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc

            #매 배치마다 진행 상화 표시
            tbar.set_postfix(loss=loss.item(), acc=acc)

    #=== validataion ===
        model.eval()
        val_loss, val_acc= 0,0

        #역전파 계산 하지마 val
        with torch.no_grad():
            for input, labels in val_loader:
                input,labels = input.to(device), labels.to(device)

                outputs = model(input)
                loss = criterion(outputs, labels )
                acc = calculate_accuracy(outputs, labels)

                val_loss += loss.item()
                val_acc += acc
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        #정확도를 기준으로 Ealystopping
        if avg_val_loss < best_val_loss:
            best_val_loss=avg_val_loss
            early_stop_counter=0

            #모델 저장도 여기서 진행
            save_path = os.path.join(save_dir, "best_model.pth")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch+1}] Best model saved! Val_Loss={best_val_loss:.4f}")

        else:
            early_stop_counter+=1
            print(f"EarlyStopping patience{early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print("EarlyStopping triggered!")
                break


        print(f"[Epoch {epoch+1}] Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f}")

        train_loss_list.append(train_loss/len(train_loader))
        val_loss_list.append(avg_val_loss)

        train_acc_list.append(train_acc / len(train_loader))
        val_acc_list.append(avg_val_acc)

    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train & val Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train & Val Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model(data_dir)


