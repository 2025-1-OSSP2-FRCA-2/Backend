from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import io
import os
import numpy as np
import torch
from collections import deque
from app.models.efficientNetPrac import VideoEfficientNet, transform

router = APIRouter(prefix="/ws")

# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VideoEfficientNet(pretrained=True).to(device)
# 학습된 가중치 로드
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', 'model.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@router.websocket("/student")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 학생 WebSocket 연결됨")

    # 프레임 버퍼 초기화
    frame_buffer = deque(maxlen=8)  # clip_len=8에 맞춤

    try:
        while True:
            print("데이터 수신 대기 중...")
            data = await websocket.receive_bytes()
            # 이미지 처리
            image = Image.open(io.BytesIO(data)).convert("RGB")
            # 모델 입력 형식으로 변환
            img_tensor = transform(image)
            # 프레임 버퍼에 추가
            frame_buffer.append(img_tensor)
            # 버퍼가 가득 차면 추론
            if len(frame_buffer) == 8:
                # [8, C, H, W] -> [1, C, 8, H, W]
                clip = torch.stack(list(frame_buffer), dim=0)
                clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(clip)
                    prob = torch.sigmoid(logits)
                    preds = prob.gt(0.5).sum(dim=2).squeeze(0).tolist()
                
                emotions = ['boredom', 'confusion', 'engagement', 'frustration']
                results = {emotion: pred for emotion, pred in zip(emotions, preds)}
                print("→ 감정 상태 예측:", results)


    except WebSocketDisconnect:
        print("🔴 WebSocket 연결 종료됨")