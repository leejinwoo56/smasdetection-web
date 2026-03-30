import os
import uuid
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from analog_detect_video import process_frame

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    """
    영상 파일을 받아서 SMAS 감지 처리 후 결과 영상을 반환하는 API
    1. 업로드된 영상을 uploads/ 폴더에 저장
    2. frame별로 process_frame() 실행
    3. 처리된 영상을 outputs/ 폴더에 저장
    4. 결과 영상 파일을 응답으로 반환
    """

    # 업로드 파일 확장자 확인
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 고유한 파일명 생성 (동시에 여러 요청이 와도 충돌 방지)
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_input.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_output.mp4")

    # 업로드된 파일 저장
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 영상 처리
    try:
        run_processing(input_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

    # 처리된 영상 반환
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="result.mp4"
    )


def run_processing(input_path: str, output_path: str):
    """
    analog_detect_video.py의 process_frame()을 이용해
    영상을 frame별로 처리하고 결과 영상을 저장
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[처리 시작] {width}x{height} @ {fps:.1f}fps, 총 {total} frames")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        print(f"[Frame] {idx}/{total}", end="\r")

        overlay_full = process_frame(frame)
        writer.write(overlay_full)

    cap.release()
    writer.release()
    print(f"\n[완료] 총 {idx}개 frame 처리")


# 나중에 frontend 파일을 serving하기 위한 준비
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
