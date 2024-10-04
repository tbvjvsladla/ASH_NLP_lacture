import torch
from tqdm import tqdm  # 훈련 진행상황 체크

import wandb #wandb

class ModelTrainer:
    def __init__(self, epoch_step, device='cuda', BC_mode=False,
                        aux=False, wandb=None, iter=False):
        
        self.epoch_step = epoch_step #tqdm 및 epoch출력정보를 스탭별로
        self.device = device #훈련/검증이 어느 디바이스인지(Default=GPU)
        self.BC_mode = BC_mode #이진분류/다중분류인지?(Default=다중분류)
        self.aux = aux #모델에 보조분류기(Auxiliary classifier)가 있는지 확인

        self.wandb = wandb  # wandb 객체를 클래스의 속성으로 저장
        self.iter = iter # 훈련의 이터레이션 결과치를 가져올 때 사용

    def model_train(self, model, data_loader, loss_fn,
                    optimizer_fn, epoch,
                    scheduler_fn=None):
        
        #기록할 모델, 로스함수
        #로그 종류 : All -> "Gradient", "parameters"
        #log_freq : N번 배치(이터레이션)마다 로깅을 수행
        if self.wandb:
            self.wandb.watch(model, loss_fn, log="all", log_freq=10)

        model.train() # 모델을 훈련모드로 설정
        # loss를 계산하기 위한 임시변수 생성
        run_size, run_loss, correct = 0, 0, 0
        his_iter = []

        # epoch_size 일때만 tqdm 진행 바 생성
        if (epoch+1) % self.epoch_step == 0 or epoch == 0:
            progress_bar = tqdm(data_loader)
        else:
            progress_bar = data_loader

        for image, label in progress_bar:
            # 입력된 데이터의 디바이스 이전
            image = image.to(self.device)
            # 다중분류/이진분류에 따라 라벨 데이터 처리
            if self.BC_mode != True:
                label = label.to(self.device)
            else:
                label = label.to(self.device).float().unsqueeze(1)
            
            # 전사 과정 수행
            if self.aux != True: #보조분류기가 없는 기본모드
                output = model(image)
                loss = loss_fn(output, label)
            else: #보조분류기가 있는 경우
                outputs = model(image) #여기서 outputs는 튜플임
                loss = torch.zeros(1).to(self.device)
                for i, output in enumerate(outputs):
                    if i == 0:
                        loss += loss_fn(output, label)
                    else:
                        loss += (1 / len(outputs)) * loss_fn(output, label)

            #backward과정 수행
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

            # 스케줄러가 있을 때 업데이트
            if scheduler_fn is not None:
                scheduler_fn.step()

            # Accuracy 측정 (다중/이진)
            if self.BC_mode != True:
                if self.aux != True: #보조분류기가 없는 기본모드
                    pred = output.argmax(dim=1) #예측값의 idx출력
                else :
                    pred = outputs[0].argmax(dim=1) #예측값의 idx출력
                correct += pred.eq(label).sum().item()
            else:
                if self.aux != True: #보조분류기가 없는 기본모드
                    pred = torch.sigmoid(output) > 0.5
                else:
                    pred = torch.sigmoid(outputs[0]) > 0.5
                correct += pred.eq(label).sum().item()

            #현재까지 수행한 loss값을 얻어냄
            run_loss += loss.item() * image.size(0)

            #Tqdm의 ACC계산을 위한 분모 업데이트
            run_size += image.size(0)

            #tqdm bar에 추가 정보 기입
            if (epoch+1) % self.epoch_step == 0 or epoch == 0:
                iter_loss = run_loss / run_size
                iter_acc = correct / run_size
                if self.iter:
                    his_iter.append((iter_loss, iter_acc))
                desc = (f"[훈련중]로스: {iter_loss:.3f}, "
                        f"정확도: {iter_acc:.3f}")
                progress_bar.set_description(desc)

        batch_acc = correct / len(data_loader.dataset)
        batch_loss = run_loss / len(data_loader.dataset)
        if self.wandb:
            if (epoch+1) % self.epoch_step == 0 or epoch == 0:
                wandb.log({"훈련 epoch": epoch, 
                        "검증 로스": batch_loss,
                        "검증 정확도": batch_acc})
        if self.iter:
            return batch_loss, batch_acc, his_iter
        return batch_loss, batch_acc
    
    def model_evaluate(self, model, data_loader, loss_fn,
                        epoch):
        # 모델을 평가모드 : droupout, BN기능이 꺼짐
        model.eval()

        # gradient 업데이트를 방지
        with torch.no_grad():
            # loss를 계산하기 위한 임시변수 생성
            run_loss, correct = 0, 0

            if (epoch+1) % self.epoch_step == 0 or epoch == 0:
                progress_bar = tqdm(data_loader)
            else:
                progress_bar = data_loader

            for image, label in progress_bar:
                # 입력된 데이터의 디바이스 이전
                image = image.to(self.device)
                # 다중분류/이진분류에 따라 라벨 데이터 처리
                if self.BC_mode != True:
                    label = label.to(self.device)
                else:
                    label = label.to(self.device).float().unsqueeze(1)

                # 전사 과정 수행
                output = model(image)
                loss = loss_fn(output, label)

                # Accuracy 측정 (다중/이진)
                if self.BC_mode != True:
                    pred = output.argmax(dim=1) #예측값의 idx출력
                    correct += pred.eq(label).sum().item()
                else:
                    pred = torch.sigmoid(output) > 0.5
                    correct += pred.eq(label).sum().item()

                #현재까지 수행한 loss값을 얻어냄
                run_loss += loss.item() * image.size(0)

            batch_acc = correct / len(data_loader.dataset)
            batch_loss = run_loss / len(data_loader.dataset)
            if self.wandb:
                if (epoch+1) % self.epoch_step == 0 or epoch == 0:
                    wandb.log({"검증 epoch": epoch, 
                            "검증 로스": batch_loss,
                            "검증 정확도": batch_acc})

            return batch_loss, batch_acc
