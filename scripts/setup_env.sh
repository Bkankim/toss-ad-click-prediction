#!/usr/bin/env bash

set -euo pipefail

# 로그 메시지를 규격화하여 출력한다.
log() {
  local level="$1";
  shift
  printf '[%s] %s\n' "$level" "$*"
}

# 스크립트 실행 전제 조건을 점검한다.
preflight_checks() {
  if [[ "$(uname -s)" != "Linux" && "$(uname -s)" != "Darwin" ]]; then
    log WARN "이 스크립트는 Linux/macOS 환경을 기준으로 작성되었습니다."
  fi
}

# 사용자가 원하는 Python 바이너리를 결정한다.
resolve_python_bin() {
  local requested="${PYTHON_BIN:-python3.11}"
  if command -v "$requested" >/dev/null 2>&1; then
    echo "$requested"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    log WARN "${requested} 바이너리를 찾을 수 없어 python3로 대체합니다."
    echo python3
    return 0
  fi

  log ERROR "사용 가능한 Python 인터프리터를 찾지 못했습니다. PYTHON_BIN 환경 변수를 확인하세요."
  exit 1
}

# uv가 설치되어 있는지 확인하고 필요시 설치 안내를 출력한다.
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo uv
    return 0
  fi

  if command -v pipx >/dev/null 2>&1; then
    log INFO "pipx를 통해 uv 설치를 시도합니다."
    pipx install uv >/dev/null 2>&1 || true
    if command -v uv >/dev/null 2>&1; then
      echo uv
      return 0
    fi
  fi

  log WARN "uv를 찾을 수 없습니다. requirements.lock.txt를 pip로 설치합니다."
  echo pip
}

# Python 가상환경을 생성하고 활성화한다.
create_venv() {
  local python_bin="$1"
  local venv_dir="$2"

  if [[ -d "$venv_dir" ]]; then
    log INFO "기존 가상환경을 재사용합니다: $venv_dir"
  else
    log INFO "가상환경을 생성합니다: $venv_dir"
    "$python_bin" -m venv "$venv_dir"
  fi

  # shellcheck source=/dev/null
  source "$venv_dir/bin/activate"
  log INFO "가상환경 활성화 완료 (${VIRTUAL_ENV})"
}

# 프로젝트 의존성을 설치한다.
install_dependencies() {
  local package_manager="$1"
  local repo_root="$2"

  if [[ "$package_manager" == "uv" ]]; then
    log INFO "uv sync로 의존성을 설치합니다."
    uv sync
  else
    log INFO "pip로 requirements.lock.txt를 설치합니다."
    pip install --upgrade pip
    pip install -r "$repo_root/requirements.lock.txt"
  fi
}

# LightGBM GPU 설치 여부를 안내한다.
optional_gpu_note() {
  if [[ "${INSTALL_LGBM_GPU:-0}" == "1" ]]; then
    log INFO "GPU 환경이 감지되어 LightGBM GPU 패키지를 추가 설치합니다."
    pip install lightgbm --extra-index-url https://pypi.nvidia.com
  else
    log INFO "GPU 최적화 LightGBM 설치는 건너뜁니다. 필요 시 INSTALL_LGBM_GPU=1로 재실행하세요."
  fi
}

# 샘플 스모크 테스트 실행을 안내한다.
print_completion_message() {
  cat <<'EOF'

환경 구성 완료 ✅
- 가상환경 활성화: source .venv/bin/activate
- 샘플 로딩 테스트: python main.py --check-data
- 루트 용량 확인: du -sh / 2>/dev/null (150GB 한도 관리)

`.env` 파일을 `.env.example`을 참고해 작성 후, Story 1.1의 스모크 테스트 스크립트를 실행하세요.
EOF
}

# 실행 진입점을 정의한다.
main() {
  preflight_checks

  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  local venv_dir="$repo_root/.venv"

  local python_bin
  python_bin="$(resolve_python_bin)"

  create_venv "$python_bin" "$venv_dir"

  local pkg_manager
  pkg_manager="$(ensure_uv)"

  install_dependencies "$pkg_manager" "$repo_root"
  optional_gpu_note
  print_completion_message
}

main "$@"
