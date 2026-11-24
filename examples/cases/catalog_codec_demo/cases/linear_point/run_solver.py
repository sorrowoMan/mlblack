"""标准 Case CLI；日常运行推荐使用 Project 入口获得 L0 grant。"""
from blackbase.project import run_standard_case_cli


def main(argv=None):
    return run_standard_case_cli(__file__, framework="mlblack", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
