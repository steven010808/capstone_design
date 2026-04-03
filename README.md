# Multimodal Search & Multi-Stage Recommendation System

## 1. 프로젝트 개요
본 프로젝트는 멀티모달 검색 및 Multi-Stage 추천 시스템 구축을 목표로 한다.  
1주차에서는 전체 과제 구조를 이해하고, 로컬 개발 환경 및 Docker 기반 실행 환경을 구축했으며,
FastAPI / Streamlit / Redis가 연결된 최소 동작 시스템(skeleton)을 완성했다.

## 2. 1주차 목표
- 프로젝트 기본 구조 이해
- Python 가상환경 및 GitHub 저장소 세팅
- FastAPI 검색/추천 API skeleton 구축
- Streamlit 대시보드 skeleton 구축
- Docker Compose 기반 실행 환경 구성
- Redis 연결 및 config 분리
- api / dashboard / redis 3서비스 연결 확인

## 3. 현재 구현 상태
### 완료
- `POST /api/search` API skeleton
- `GET /api/recommend` API skeleton
- `GET /api/system/status` API 추가
- Streamlit 대시보드 기본 탭 구성
  - Search Metrics
  - Recommendation Metrics
  - A/B Test
  - System Status
- Redis 연결 및 key-value 저장/조회 테스트
- Docker Compose로 `api`, `dashboard`, `redis` 동시 실행
- 공통 설정 파일 `configs/config.yaml`
- 공통 유틸
  - `src/common/config.py`
  - `src/common/redis_client.py`

### 진행 중
- `src/search`, `src/recommendation`, `src/simulator` 모듈 구조 정리
- 검색/추천 더미 로직을 각 서비스 모듈로 분리
- README 및 docs 문서화 보완

## 4. 폴더 구조
```text
multimodal_search_recommand_sys/
├─ configs/
│  └─ config.yaml
├─ docker/
│  ├─ Dockerfile.api
│  └─ Dockerfile.dashboard
├─ docs/
├─ src/
│  ├─ common/
│  ├─ dashboard/
│  ├─ serving/
│  ├─ search/
│  ├─ recommendation/
│  └─ simulator/
├─ tests/
├─ requirements.txt
├─ docker-compose.yml
└─ README.md