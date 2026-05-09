from __future__ import annotations

from typing import Any, Dict

from config import describe_trainers


TRAINER_CONTRACTS: Dict[str, Dict[str, Any]] = describe_trainers()
TRAINER_RESOURCE_PROFILES: Dict[str, Dict[str, Any]] = {
    str(key): dict(dict(value).get("contract", {}).get("execution_resources", {}))
    for key, value in dict(TRAINER_CONTRACTS).items()
}


def get_trainer_contracts() -> Dict[str, Dict[str, Any]]:
    return describe_trainers()


def get_trainer_resource_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        str(key): dict(dict(value).get("contract", {}).get("execution_resources", {}))
        for key, value in dict(describe_trainers()).items()
    }


__all__ = [
    "TRAINER_CONTRACTS",
    "TRAINER_RESOURCE_PROFILES",
    "get_trainer_contracts",
    "get_trainer_resource_profiles",
]
