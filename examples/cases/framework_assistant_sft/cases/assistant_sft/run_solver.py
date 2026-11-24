"""标准 CLI；资源与 Artifact 仍由共享 Case runtime 管理。"""
from blackbase.project import run_standard_case_cli


def main(argv=None):
    return run_standard_case_cli(__file__, framework="mlblack", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
