from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import io
import os
import torch
import asyncio
import json
from collections import deque
from app.models.efficientNetPrac import VideoEfficientNet, transform
from app.api.concentrativeness import calculate_concentration

# WebSocket 라우터 생성, "/ws" 경로로 설정
router = APIRouter(prefix="/api/ws")

# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA 사용 가능 여부 확인
model = VideoEfficientNet(pretrained=True).to(device)  # 사전 학습된 모델 로드
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', 'model.pt')  # 모델 가중치 경로 설정
model.load_state_dict(torch.load(model_path, map_location=device))  # 모델 가중치 로드
model.eval()  # 모델 평가 모드로 설정

# 연결된 선생님 WebSocket 클라이언트를 저장할 set
teacher_clients = set()
# 연결된 학생 WebSocket 클라이언트를 저장할 딕셔너리 (학생 ID: WebSocket 객체)
student_connections = {}
# WebRTC 시그널링용 연결 관리
webrtc_students = {}  # student_id: websocket
webrtc_profs = set()  # 여러 강사 지원 가능

@router.websocket("/webrtc/student/{student_id}")
async def webrtc_student_ws(websocket: WebSocket, student_id: str):
    await websocket.accept()  # WebSocket 연결 수락
    webrtc_students[student_id] = websocket  # 학생 WebSocket 저장
    print(f"🟢 WebRTC 학생 연결: {student_id}")

    try:
        while True:
            msg = await websocket.receive_text()  # 텍스트 메시지 수신
            data = json.loads(msg)  # JSON 데이터 파싱
            # offer/candidate → 모든 강사에게 중계
            for prof_ws in list(webrtc_profs):
                try:
                    await prof_ws.send_text(json.dumps({**data, "from_student_id": student_id}))  # 강사에게 데이터 전송
                except Exception as e:
                    print(f"⚠️ 강사에게 WebRTC 시그널링 전송 실패: {e}")
    except WebSocketDisconnect:
        del webrtc_students[student_id]  # 연결 종료 시 학생 제거
        print(f"🔴 WebRTC 학생 연결 종료: {student_id}")

@router.websocket("/webrtc/prof")
async def webrtc_prof_ws(websocket: WebSocket):
    await websocket.accept()  # WebSocket 연결 수락
    webrtc_profs.add(websocket)  # 강사 WebSocket 저장
    print("🟢 WebRTC 강사 연결")

    try:
        while True:
            msg = await websocket.receive_text()  # 텍스트 메시지 수신
            data = json.loads(msg)  # JSON 데이터 파싱
            # answer/candidate → 해당 학생에게 중계
            to_student_id = data.get("to_student_id")
            if to_student_id and to_student_id in webrtc_students:
                try:
                    await webrtc_students[to_student_id].send_text(msg)  # 학생에게 데이터 전송
                except Exception as e:
                    print(f"⚠️ 학생에게 WebRTC 시그널링 전송 실패: {e}")
    except WebSocketDisconnect:
        webrtc_profs.remove(websocket)  # 연결 종료 시 강사 제거
        print("🔴 WebRTC 강사 연결 종료")

@router.websocket("/prof")
async def teacher_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # WebSocket 연결 수락
    teacher_clients.add(websocket)  # 연결된 선생님 클라이언트 추가
    print(f"🔵 선생님 WebSocket 연결됨: {websocket}")
    
    # 선생님 연결 시 모든 학생에게 알림
    for student_id, student_ws in student_connections.items():
        try:
            await student_ws.send_text(json.dumps({"type": "teacher_connected", "message": "선생님이 입장하셨습니다."}))  # 학생에게 알림 전송
        except Exception as e:
            print(f"⚠️ 학생 {student_id}에게 선생님 연결 알림 전송 실패: {e}")

    try:
        while True:
            message = await websocket.receive_text()  # 메시지 수신
            print(f"선생님으로부터 받은 메시지: {message}")
            try:
                msg_data = json.loads(message)  # JSON 메시지 파싱
                if msg_data.get("type") == "warning" and "student_id" in msg_data and "message" in msg_data:
                    await send_warning_to_student(msg_data["student_id"], msg_data["message"])  # 경고 메시지 전송
            except json.JSONDecodeError:
                print(f"⚠️ 유효하지 않은 JSON 메시지: {message}")

    except WebSocketDisconnect:
        teacher_clients.remove(websocket)  # 연결 종료 시 클라이언트 제거
        print(f"🔴 선생님 WebSocket 연결 종료됨: {websocket}")

        # 선생님 연결 종료 시 모든 학생에게 알림
        disconnect_message = json.dumps({"type": "teacher_disconnected", "message": "선생님과의 연결이 끊어졌습니다."})
        for student_id, student_ws in list(student_connections.items()):
            try:
                await student_ws.send_text(disconnect_message)  # 학생에게 연결 종료 알림 전송
                print(f"🔴 학생 {student_id}에게 선생님 연결 종료 알림 전송")
            except Exception as e:
                print(f"⚠️ 학생 {student_id}에게 연결 종료 알림 전송 실패: {e}")

@router.websocket("/student/{student_id}")
async def student_websocket_endpoint(websocket: WebSocket, student_id: str):
    await websocket.accept()  # WebSocket 연결 수락
    student_connections[student_id] = websocket  # 학생 연결 저장
    print(f"🟢 학생 WebSocket 연결됨 (ID: {student_id})")

    # 학생 연결 시 현재 선생님 연결 상태 알림
    if teacher_clients:
        await websocket.send_text(json.dumps({"type": "teacher_connected", "message": "선생님이 입장하셨습니다."}))  # 학생에게 알림 전송
    else:
        await websocket.send_text(json.dumps({"type": "teacher_disconnected", "message": "선생님이 입장하지 않았습니다."}))  # 학생에게 알림 전송

    frame_buffer = deque(maxlen=8)  # 프레임 버퍼 초기화

    try:
        while True:
            data = await websocket.receive_bytes()  # 이미지 데이터 수신
            image = Image.open(io.BytesIO(data)).convert("RGB")  # 이미지 변환
            img_tensor = transform(image)  # 이미지 텐서 변환
            frame_buffer.append(img_tensor)  # 프레임 버퍼에 추가

            if len(frame_buffer) == 8:
                clip = torch.stack(list(frame_buffer), dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # 클립 생성

                with torch.no_grad():
                    logits = model(clip)  # 모델 예측
                    prob = torch.sigmoid(logits)  # 확률 계산
                    preds = prob.gt(0.5).sum(dim=2).squeeze(0).tolist()  # 예측 결과 변환

                emotions_binary = ['boredom', 'confusion', 'engagement', 'frustration']  # 감정 상태
                emotion_results = {emotion: pred for emotion, pred in zip(emotions_binary, preds)}  # 감정 결과 매핑
                print(f"→ 학생 {student_id} 감정 상태 예측:", emotion_results)

                # 감정 결과를 바탕으로 집중도 계산
                concentration_score = calculate_concentration(emotion_results)
                print(f"→ 학생 {student_id} 집중도: {concentration_score}")

                # 계산된 집중도 결과를 선생님 클라이언트에게 실시간 전송
                message_for_teachers = json.dumps({
                    "type": "student_data",
                    "student_id": student_id,
                    "emotion_results": emotion_results,
                    "concentration": concentration_score
                })

                for teacher_ws in list(teacher_clients):
                    try:
                        await teacher_ws.send_text(message_for_teachers)  # 집중도 결과 전송
                    except Exception as e:
                        print(f"⚠️ 선생님에게 학생 {student_id}의 집중도 전송 실패: {e}")

    except WebSocketDisconnect:
        if student_id in student_connections:
            del student_connections[student_id]  # 연결 종료 시 학생 제거
        print(f"🔴 학생 WebSocket 연결 종료됨 (ID: {student_id})")

async def send_warning_to_student(student_id: str, message: str):
    student_ws = student_connections.get(student_id)  # 학생 WebSocket 가져오기
    if student_ws:
        try:
            await student_ws.send_text(json.dumps({"type": "warning", "message": message}))  # 경고 메시지 전송
            print(f"📢 학생 {student_id}에게 경고 메시지 전송: {message}")
        except Exception as e:
            print(f"⚠️ 학생 {student_id}에게 경고 메시지 전송 실패: {e}")
    else:
        print(f"⚠️ 학생 {student_id}의 WebSocket 연결을 찾을 수 없습니다.")

@router.websocket("/check_connection")
async def check_connection_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()  # WebSocket 연결 수락
        # 선생님 연결 상태 확인
        teacher_connected = len(teacher_clients) > 0
        # 연결 상태 전송
        await websocket.send_text(json.dumps({
            "teacher_connected": teacher_connected
        }))
    except Exception as e:
        print(f"⚠️ 연결 상태 확인 중 오류 발생: {e}")
    finally:
        await websocket.close()  # WebSocket 연결 종료
