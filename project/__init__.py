from .doctor import DoctorDiagnostic, DoctorReport, format_doctor_report, run_project_doctor
from .scaffold import add_case, create_project

create_standard_scaffold = create_project

__all__ = [
    "DoctorDiagnostic",
    "DoctorReport",
    "add_case",
    "create_project",
    "create_standard_scaffold",
    "format_doctor_report",
    "run_project_doctor",
]
