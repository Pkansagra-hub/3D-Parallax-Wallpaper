from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InpaintPlan:
    background_pass: bool = True
    middle_pass: bool = True
    foreground_pass: bool = False


def default_inpaint_plan() -> InpaintPlan:
    return InpaintPlan()
