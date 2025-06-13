from fastapi import FastAPI, WebSocket  # FastAPI 앱과 WebSocket 기능 import
from app.api.websocket import router as websocket_router          # 우리가 정의한 라우터 모듈 임포트
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 설정용
import uvicorn                          # 서버 실행을 위한 uvicorn

# FastAPI 인스턴스 생성
app = FastAPI() 

# 우리가 정의한 WebSocket 라우터 등록
app.include_router(websocket_router)

# 🔓 CORS 설정: 프론트엔드가 다른 포트(예: 5173)에서 실행될 때 필요한 설정
# → 실제 배포 시에는 allow_origins=["http://yourdomain.com"] 등으로 제한 필요
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],  # Vite 개발 서버.
    allow_origins=["*"],     # 모든 출처 허용 (개발용)
    allow_methods=["*"],     # 모든 HTTP 메서드 허용
    allow_headers=["*"],     # 모든 헤더 허용
)

if __name__ == "__main__":
    # uvicorn으로 서버 실행 (FastAPI 앱은 main.py의 app 객체)
    # --reload: 코드 변경 시 자동으로 서버 재시작 (개발 시 유용)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)