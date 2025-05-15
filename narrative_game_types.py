from typing import List, TypedDict, Optional, Literal

StoryStage = Literal[
    "EXPOSITION",
    "INCITING_INCIDENT",
    "RISING_ACTION_1",
    "RISING_ACTION_2",
    "CLIMAX_PRELUDE",
    "CLIMAX_CHALLENGE",
    "FALLING_ACTION",
    "RESOLUTION",
]

STORY_STAGES: List[StoryStage] = [
    "EXPOSITION",
    "INCITING_INCIDENT",
    "RISING_ACTION_1",
    "RISING_ACTION_2",
    "CLIMAX_PRELUDE",
    "CLIMAX_CHALLENGE",
    "FALLING_ACTION",
    "RESOLUTION",
]

class NarrativeGameState(TypedDict):
    current_stage: StoryStage
    story_log: List[str]
    user_response: Optional[str]
    attempt_count: int
    hint_level: int
    max_failed_attempts_before_intervention: int
    challenge_passed: bool
    system_message: Optional[str]
    failed_responses: List[str]
    story_plot_points: Optional[List[str]]
    current_stage_plot_point: Optional[str]
    current_challenge_description: Optional[str]
    expected_solution: Optional[str]
    current_hint: Optional[str]
    outcome_description: Optional[str] 