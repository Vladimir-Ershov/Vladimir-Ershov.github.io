"""
Thesaurus module containing enums and structured output definitions for the VLA system.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ActionType(str, Enum):
    """Enumeration of possible actions the system can perform."""
    MOVE = "move"
    PICK_UP = "pick_up"
    PLACE = "place"
    COUNT = "count"
    POINT_TO = "point_to"
    PUSH = "push"
    NONE = "none"


class SkillType(str, Enum):
    """Enumeration of skills being executed."""
    OBJECT_DETECTION = "object_detection"
    SPATIAL_REASONING = "spatial_reasoning"
    MANIPULATION = "manipulation"
    COUNTING = "counting"
    NAVIGATION = "navigation"


class FailureType(str, Enum):
    """Enumeration of possible failure modes."""
    OBJECT_NOT_FOUND = "object_not_found"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    UNREACHABLE_LOCATION = "unreachable_location"
    PHYSICAL_CONSTRAINT_VIOLATION = "physical_constraint_violation"
    INSUFFICIENT_VISUAL_INFORMATION = "insufficient_visual_information"
    NONE = "none"


class ActionOutput(BaseModel):
    """Structured output for VLA predictions."""
    action: ActionType = Field(description="The primary action to be executed")
    target: str = Field(description="The target object or entity for the action")
    destination: Optional[str] = Field(
        default=None,
        description="The destination location or reference object (if applicable)"
    )
    completion: float = Field(
        ge=0.0,
        le=100.0,
        description="Estimated completion percentage of the task (0-100)"
    )
    current_skill: SkillType = Field(
        description="The current skill being employed to complete the task"
    )
    failure: FailureType = Field(
        default=FailureType.NONE,
        description="Type of failure encountered, or NONE if no failure"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the decision-making process"
    )

    model_config = ConfigDict(use_enum_values=True)
