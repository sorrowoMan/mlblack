from .doctor import DoctorDiagnostic, DoctorReport, format_doctor_report, run_project_doctor
from .project_runner import execute_project, run_project
from .scaffold import add_case, add_component, create_project

create_standard_scaffold = create_project

__all__ = [
    "DoctorDiagnostic",
    "DoctorReport",
    "add_case",
    "add_component",
    "create_project",
    "create_standard_scaffold",
    "format_doctor_report",
    "run_project",
    "execute_project",
    "run_project_doctor",
]
