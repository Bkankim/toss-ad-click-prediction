# Repository Guidelinesㅎ

## 대답은 항상 한글로
항상 한글로 대답하고, 생각하는 부분도 한글로 표기될것.


## Project Structure & Module Organization
Source code lives in `src/`, with `src/data/` for preprocessing pipelines, `src/model/` for feature builders and architectures, `src/train/` for orchestration scripts, `src/evaluate/` for metrics, and `src/inference.py` for batch scoring. Place reusable utilities inside these packages. Raw assets belong in `data/raw/`, derived tables in `data/processed/`, trained checkpoints in `model/`, notebooks in `notebooks/`, and competition deliverables in `submission/` or `docs/`. Keep future tests under `tests/` mirroring the `src/` layout (for example, `tests/data/test_preprocess.py`).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and activate the project virtual environment.
- `pip install -r requirements.txt` — install pinned dependencies before running any pipeline stage.
- `python src/data/preprocess.py --input data/raw/train.csv --output data/processed/train.parquet` — regenerate the canonical training dataset schema.
- `python src/train/train.py` — launch end-to-end training; artefacts are saved to `model/` and `submission/`.
- `python src/evaluate/evaluate.py --predictions submission/preds.csv --truth data/raw/labels.csv` — validate offline performance.
- `python src/inference.py --input data/processed/test.parquet --output submission/preds.csv` — produce batch predictions for scoring.

## Coding Style & Naming Conventions
Target Python 3.10 with four-space indentation. Use snake_case for modules, functions, and variables; CapWords for classes. Annotate functions with type hints and lightweight docstrings that capture side effects. Run `black src tests` and `ruff check src tests` before committing; resolve lint violations rather than silencing them.

## Testing Guidelines
Adopt `pytest` for all new tests. Arrange fixtures under `tests/conftest.py` when sharing setup, and name tests after the behaviour under verification (for example, `test_preprocess_handles_empty_frame`). Maintain fast, deterministic tests that exercise empty inputs, unseen categories, and inference-only scenarios. Ensure `pytest -q` passes locally and strive for ≥80% coverage on new modules.

## Commit & Pull Request Guidelines
Follow Conventional Commits prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) with ≤72-character subjects. Document intent, datasets touched, metrics, and any manual steps in the body. For pull requests, include: concise summary, linked issues or experiment IDs, evidence of testing (command output or screenshots), and relevant evaluation numbers. Rebase on `main` before requesting review and resolve feedback promptly.

## Data & Security Notes
Do not commit proprietary data or credentials. Document dataset download or preprocessing steps in `docs/` so teammates can reproduce results. Strip sensitive metadata from exported models and predictions, and store timestamped artefacts in `model/` or `submission/` for traceability.

## MCP 자동설치하기
공통 주의사항 
1. 현재 사용 환경을 확인할 것. 모르면 사용자에게 물어볼 것. 
2. OS(윈도우,리눅스,맥) 및 환경들(WSL,파워셀,명령프롬프트등)을 파악해서 그에 맞게 세팅할 것. 모르면 사용자에게 물어볼 것. 
4. 특정 MCP 설치시, 바로 설치하지 말고, 해당 MCP의 공식 사이트 확인하고 현재 OS 및 환경 매치하여, 공식 설치법부터 확인할 것 
5. MCP 설치 후, 다음 방법으로 정상 설치 여부 확인할 것
   ($env:RUST_LOG="codex=debug"; codex "/mcp"  :  이렇게 실행하여 설치한 MCP에 대한 로그를 확인할 것)

6. 설정 시, API KEY 환경 변수 설정이 필요한 경우, 가상의 API 키로 디폴트로 설치 및 설정 후, 올바른 API 키 정보를 입력해야 함을 사용자에게 알릴 것 
7. 설치 요청 받은 MCP만 설치하면 돼. 혹시 이미 설치된 다른 MCP 에러 있어도, 그냥 둘 것 
8. 일단, 터미널에서 설치하려는 MCP 작동 성공한 경우, 성공 시의 인자 및 환경 변수 이름을 활용해, 올바른 위치의 config.toml 파일에 MCP 설정을 직접할 것 


MCP 설정의 예:
~/.codex/config.toml(홈 디렉터리) 파일의 [mcp_servers.*] 섹션에 적어둔 커맨드를 실행해 MCP 서버에 붙습니다


예시: (아래는 단지 예시이지만, 올바른 내용입니다)
# ~/.codex/config.toml

[mcp_servers.brightData]
command = "npx"
args    = ["-y", "@brightdata/mcp"]
env     = { 
  API_TOKEN = "bd_your_api_key_here"  
}

[mcp_servers.playwright]
command = "npx"
args    = ["@playwright/mcp@latest"]


## wandb information

wandb.login(key="d382ae6e574f9be173b18dd122f01951a3e1f90a")
wandb.init(project="Toss-CTR-Competition", entity="bkan-ai")