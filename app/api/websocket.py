
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import io
import os
import torch
import asyncio
from collections import deque
from app.models.efficientNetPrac import VideoEfficientNet, transform

router = APIRouter(prefix="/ws")

# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VideoEfficientNet(pretrained=True).to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', 'model.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@router.websocket("/student")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 학생 WebSocket 연결됨")

    # 비동기 프레임 큐와 버퍼
    frame_queue: asyncio.Queue = asyncio.Queue()
    frame_buffer = deque(maxlen=8)

    async def receiver():
        """WebSocket에서 들어오는 프레임을 큐에 넣는 태스크"""
        try:
            while True:
                data = await websocket.receive_bytes()
                image = Image.open(io.BytesIO(data)).convert("RGB")
                tensor = transform(image)
                # 큐에 (입력시각, tensor) 튜플로 저장
                await frame_queue.put((asyncio.get_event_loop().time(), tensor))
        except WebSocketDisconnect:
            # 연결 끊김을 알리기 위해 None을 푸시
            await frame_queue.put(None)
            print("🔴 수신 태스크 종료")

    async def inferencer():
        """0.5초마다 큐에서 최신 프레임 8개 뽑아 추론하는 태스크"""
        try:
            while True:
                # 0.5초 간격으로 실행
                await asyncio.sleep(0.5)

                # 큐에서 가능한 모든 아이템 꺼내기
                while not frame_queue.empty():
                    item = await frame_queue.get()
                    if item is None:
                        # 수신이 종료된 신호
                        return
                    _, tensor = item
                    frame_buffer.append(tensor)

                # 버퍼가 꽉 찼을 때만 추론
                if len(frame_buffer) == frame_buffer.maxlen:
                    # [8, C, H, W] → [1, C, 8, H, W]
                    clip = torch.stack(list(frame_buffer), dim=0)
                    clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(clip)
                        prob = torch.sigmoid(logits)
                        preds = prob.gt(0.5).sum(dim=2).squeeze(0).tolist()

                    emotions = ['boredom', 'confusion', 'engagement', 'frustration']
                    results = {e: p for e, p in zip(emotions, preds)}
                    print("→ 감정 상태 예측:", results)

        except Exception as e:
            print("⚠️ 추론 태스크 에러:", e)

    # 수신·추론 태스크 동시 실행
    recv_task = asyncio.create_task(receiver())
    infer_task = asyncio.create_task(inferencer())

    # 둘 중 하나라도 완료될 때까지 대기
    done, pending = await asyncio.wait(
        {recv_task, infer_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 남은 태스크가 있으면 취소
    for task in pending:
        task.cancel()

    print("🔴 WebSocket 처리 종료")
