from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import io
import numpy as np

router = APIRouter(prefix="/ws")

@router.websocket("/student")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 학생 WebSocket 연결됨")

    try:
        while True:
            print("데이터 수신 대기 중...")
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            np_img = np.array(image)

            # [임시 추론] 평균 밝기 계산
            avg_brightness = np.mean(np_img)
            print(f"→ 집중도 (밝기 기준): {avg_brightness:.2f}")

            # TODO: 추후 ML 모델로 추론하고 결과 저장
    except WebSocketDisconnect:
        print("🔴 WebSocket 연결 종료됨")