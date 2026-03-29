# 프로젝트 진행상황

## 프로젝트 개요
MedSAM 기반 SMAS층 감지 웹페이지
- 영상 업로드 → frame 분리 → 밝기 기준 필터링 → 결과 영상 반환
- 목표: GitHub CI/CD + AWS 배포 실습

## 기술 스택
- Backend: Python + FastAPI
- Frontend: HTML / CSS / JavaScript
- 배포: AWS EC2
- CI/CD: GitHub Actions

---

## 진행 단계

### Phase 1: 환경 세팅
- [ ] Python 가상환경(venv) 생성
- [ ] 필요 패키지 설치 (FastAPI, OpenCV 등)
- [ ] requirements.txt 작성
- [ ] GitHub 레포지토리 연결

### Phase 2: Backend 구현
- [ ] FastAPI 기본 서버 구축
- [ ] 영상 업로드 API 엔드포인트
- [ ] frame 분리 로직 구현
- [ ] 밝기 기준 필터링 로직 구현
- [ ] frame → 영상 재조합 로직 구현
- [ ] API 테스트

### Phase 3: Frontend 구현
- [ ] 메인 페이지 레이아웃 (업로드 영역 + 결과 영역)
- [ ] 영상 업로드 기능
- [ ] 원본 / 처리된 영상 나란히 표시
- [ ] 로딩 상태 표시

### Phase 4: GitHub CI/CD 세팅
- [ ] GitHub 레포지토리 생성
- [ ] .gitignore 작성
- [ ] GitHub Actions workflow 파일 작성
- [ ] main 브랜치 push 시 자동 배포 설정

### Phase 5: AWS 배포
- [ ] EC2 인스턴스 생성
- [ ] 서버 환경 세팅 (Python, nginx 등)
- [ ] 앱 배포
- [ ] 도메인/포트 설정
- [ ] 배포 테스트

---

## 현재 상태
**Phase 1 진행 중**

## 완료된 항목
- 프로젝트 기획 완료
- CLAUDE.md 지침 작성 완료
