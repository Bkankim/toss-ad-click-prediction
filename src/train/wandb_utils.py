from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


@dataclass
# wandb 설치 여부와 자격 상태를 묶어서 보고한다.
class WandbStatus:
    installed: bool
    api_key_available: bool
    project: Optional[str] = None
    entity: Optional[str] = None

    # 상태를 노트북/로그에 남기기 쉬운 딕셔너리로 변환한다.
    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "api_key_available": self.api_key_available,
            "project": self.project,
            "entity": self.entity,
        }


+# wandb가 설치되어 있고 API 키가 준비됐는지 확인한다.
def check_wandb_setup(project: Optional[str] = None, entity: Optional[str] = None) -> WandbStatus:
    api_key = os.getenv("WANDB_API_KEY")
    return WandbStatus(
        installed=_WANDB_AVAILABLE,
        api_key_available=bool(api_key),
        project=project,
        entity=entity,
    )
