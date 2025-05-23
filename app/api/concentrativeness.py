import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import io
import os
import numpy as np
import torch
from collections import deque
from app.models.efficientNetPrac import VideoEfficientNet, transform # app.models.efficientNetPrac는 예시 경로입니다.

# APIRouter 인스턴스 생성
router = APIRouter(prefix="/ws")

# 연결된 선생님 WebSocket 클라이언트를 저장할 set
teacher_clients = set()
# 연결된 학생 WebSocket 클라이언트를 저장할 딕셔너리 (학생 ID: WebSocket 객체)
student_connections = {} # 각 학생에게 개별적으로 메시지를 보내기 위함

# ======== 집중도 계산 함수 ========
def calculate_concentration(emotion_results: dict) -> float:
    """
    { 'boredom': 0/1, 'confusion': 0/1, 'engagement': 0/1, 'frustration': 0/1 } 형태
    """
    engagement = emotion_results.get('engagement', 0)
    boredom = emotion_results.get('boredom', 0)
    confusion = emotion_results.get('confusion', 0)
    frustration = emotion_results.get('frustration', 0)

    # 긍정적인 감정의 가중치
    positive_score = engagement * 0.6 # 집중도가 높으면 0.6점 기여
    # 부정적인 감정의 가중치 (1에서 해당 감정값을 빼서, 감정이 낮을수록 높은 점수를 얻게 함)
    negative_score = (1 - boredom) * 0.15 + \
                     (1 - confusion) * 0.15 + \
                     (1 - frustration) * 0.1

    # 총 집중도 점수 (0 ~ 1 사이로 정규화)
    concentration_score = positive_score + negative_score

    # 점수가 0과 1 사이를 벗어나지 않도록 클리핑
    concentration_score = max(0.0, min(1.0, concentration_score))

    return float(f"{concentration_score:.2f}") # 소수점 두 자리로 반환


# ======== WebSocket 엔드포인트 ========
@router.websocket("/teacher")
async def teacher_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    teacher_clients.add(websocket)
    print(f"🔵 선생님 WebSocket 연결됨: {websocket}")
    try:
        while True:
            # 선생님 페이지에서 보낸 메시지 수신 (예: 학생에게 경고 보내기 요청)
            message = await websocket.receive_text()
            print(f"선생님으로부터 받은 메시지: {message}")
            try:
                msg_data = json.loads(message)
                if msg_data.get("type") == "warning" and "student_id" in msg_data and "message" in msg_data:
                    await send_warning_to_student(msg_data["student_id"], msg_data["message"])
            except json.JSONDecodeError:
                print(f"⚠️ 유효하지 않은 JSON 메시지: {message}")

    except WebSocketDisconnect:
        teacher_clients.remove(websocket)
        print(f"🔴 선생님 WebSocket 연결 종료됨: {websocket}")

        # 선생님 연결 종료 시 모든 학생에게 알림
        disconnect_message = json.dumps({"type": "teacher_disconnected", "message": "선생님과의 연결이 끊어졌습니다."})
        for student_id, student_ws in list(student_connections.items()):
            try:
                await student_ws.send_text(disconnect_message)
                print(f"🔴 학생 {student_id}에게 선생님 연결 종료 알림 전송")
            except Exception as e:
                print(f"⚠️ 학생 {student_id}에게 연결 종료 알림 전송 실패: {e}")


@router.websocket("/student/{student_id}") # 학생 ID를 경로 매개변수로 받음
async def student_websocket_endpoint(websocket: WebSocket, student_id: str):
    await websocket.accept()
    student_connections[student_id] = websocket # 학생 ID로 연결 저장
    print(f"🟢 학생 WebSocket 연결됨 (ID: {student_id})")

    frame_buffer = deque(maxlen=8) # clip_len=8에 맞춤

    #학생 집중도 결과 선생님에게 전송
    try:
        while True:
            # 학생으로부터 이미지 데이터 (바이트 또는 Base64 문자열) 수신
            data = await websocket.receive_bytes() # 이미지 바이트를 받도록 유지

            image = Image.open(io.BytesIO(data)).convert("RGB")
            img_tensor = transform(image)
            frame_buffer.append(img_tensor)

            if len(frame_buffer) == 8:
                clip = torch.stack(list(frame_buffer), dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(clip)
                    prob = torch.sigmoid(logits)
                    preds = prob.gt(0.5).sum(dim=2).squeeze(0).tolist() # True/False를 0/1로 변환된 상태

                emotions_binary = ['boredom', 'confusion', 'engagement', 'frustration']
                emotion_results = {emotion: pred for emotion, pred in zip(emotions_binary, preds)}
                print(f"→ 학생 {student_id} 감정 상태 예측:", emotion_results)

                # **감정 결과를 바탕으로 집중도 계산**
                concentration_score = calculate_concentration(emotion_results)
                print(f"→ 학생 {student_id} 집중도: {concentration_score}")

                # **계산된 집중도 결과를 선생님 클라이언트에게 실시간 전송**
                message_for_teachers = json.dumps({
                    "type": "student_data",
                    "student_id": student_id,
                    "emotion_results": emotion_results,
                    "concentration": concentration_score
                })

                for teacher_ws in teacher_clients:
                    try:
                        await teacher_ws.send_text(message_for_teachers)
                    except Exception as e:
                        print(f"⚠️ 선생님에게 학생 {student_id}의 집중도 전송 실패: {e}")

    except WebSocketDisconnect:
        # 학생 연결이 끊어지면 딕셔너리에서 제거
        if student_id in student_connections:
            del student_connections[student_id]
        print(f"🔴 학생 WebSocket 연결 종료됨 (ID: {student_id})")

# **외부에서 호출 가능한 경고 메시지 전송 함수**
async def send_warning_to_student(student_id: str, message: str):
    """
    특정 학생에게 경고 메시지를 WebSocket을 통해 전송합니다.
    """
    student_ws = student_connections.get(student_id)
    if student_ws:
        try:
            await student_ws.send_text(json.dumps({"type": "warning", "message": message}))
            print(f"📢 학생 {student_id}에게 경고 메시지 전송: {message}")
        except Exception as e:
            print(f"⚠️ 학생 {student_id}에게 경고 메시지 전송 실패: {e}")
    else:
        print(f"⚠️ 학생 {student_id}의 WebSocket 연결을 찾을 수 없습니다.")