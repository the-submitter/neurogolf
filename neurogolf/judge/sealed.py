"""Stable integrity-protected validation seed expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neurogolf.errors import ConfigurationError


class SealedSeedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    algorithm: str = Field(pattern=r"^affine-mod-v1$")
    start: int = Field(ge=0)
    stride: int = Field(gt=0)
    modulus: int = Field(gt=1)
    count: int = Field(gt=0)
    namespace: str

    def base_seeds(self, count: int | None = None) -> list[int]:
        requested = self.count if count is None else count
        if requested > self.count:
            raise ConfigurationError(
                f"Requested {requested} sealed seeds, manifest provides {self.count}"
            )
        return [(self.start + index * self.stride) % self.modulus for index in range(requested)]

    def task_seeds(self, task_id: str, count: int | None = None) -> list[int]:
        result: list[int] = []
        for seed in self.base_seeds(count):
            payload = f"{self.namespace}:{task_id}:{seed}".encode()
            result.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF)
        return result


def load_sealed_manifest(path: Path) -> SealedSeedManifest:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Could not read sealed seed manifest {path}: {error}") from error
    return SealedSeedManifest.model_validate(payload)
