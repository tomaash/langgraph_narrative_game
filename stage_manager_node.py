from narrative_game_types import NarrativeGameState, StoryStage, STORY_STAGES

def stage_manager(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: stage_manager --- Stage: {state.get('current_stage')}")

    story_plot_points = state.get('story_plot_points')
    if not story_plot_points or len(story_plot_points) != len(STORY_STAGES):
        print("Error: story_plot_points not found in state or has incorrect length. Cannot proceed.")
        return { 
            "system_message": "Critical Error: Missing or invalid story plot points.",
            "current_stage": state.get("current_stage", STORY_STAGES[0]),
            "story_log": state.get("story_log", []),
            "user_response": None, "attempt_count": 0, "hint_level": 0,
            "max_failed_attempts_before_intervention": state.get("max_failed_attempts_before_intervention", 3),
            "challenge_passed": False, "failed_responses": [],
            "story_plot_points": story_plot_points,
            "current_stage_plot_point": "ERROR: PLOT POINTS MISSING",
            "current_challenge_description": None, "expected_solution": None,
            "current_hint": None, "outcome_description": None
        }

    if state.get('current_stage') is None:
        print(f"Stage Manager: Initializing game at EXPOSITION.")
        first_stage = STORY_STAGES[0]
        current_plot = story_plot_points[0]
        print(f"Stage Manager: Plot for {first_stage}: {current_plot}")
        return {
            "current_stage": first_stage,
            "system_message": f"Welcome to the story! Beginning with the EXPOSITION. Your goal: {current_plot}",
            "challenge_passed": False,
            "story_log": [],
            "attempt_count": 0,
            "hint_level": 0,
            "max_failed_attempts_before_intervention": state.get('max_failed_attempts_before_intervention', 3),
            "failed_responses": [],
            "story_plot_points": story_plot_points, 
            "current_stage_plot_point": current_plot, 
            "current_challenge_description": None,
            "expected_solution": None,
            "current_hint": None,
            "outcome_description": None,
            "user_response": None,
        }

    current_stage_index = STORY_STAGES.index(state['current_stage'])
    
    if state.get('challenge_passed', False):
        if state['current_stage'] == "RESOLUTION":
            print("Stage Manager: Game successfully completed!")
            final_plot_point = story_plot_points[current_stage_index]
            return {
                **state,
                "system_message": "Congratulations! You have completed the story.",
                "current_stage_plot_point": final_plot_point,
            }

        if current_stage_index + 1 < len(STORY_STAGES):
            next_stage_index = current_stage_index + 1
            next_stage = STORY_STAGES[next_stage_index]
            next_plot = story_plot_points[next_stage_index]
            system_message = f"You have successfully completed the {state['current_stage']} stage. Moving to {next_stage}. Your new goal: {next_plot}"
            print(f"Stage Manager: Advancing from {state['current_stage']} to {next_stage}")
            print(f"Stage Manager: Plot for {next_stage}: {next_plot}")
            return {
                "current_stage": next_stage,
                "system_message": system_message,
                "challenge_passed": False,
                "attempt_count": 0,
                "hint_level": 0,
                "max_failed_attempts_before_intervention": state.get('max_failed_attempts_before_intervention', 3),
                "current_challenge_description": None,
                "expected_solution": None,
                "current_hint": None,
                "outcome_description": None,
                "user_response": None,
                "failed_responses": [],
                "story_plot_points": story_plot_points,
                "current_stage_plot_point": next_plot,
                "story_log": state.get("story_log", [])
            }
        else:
            print("Stage Manager: Attempted to advance beyond RESOLUTION, but it wasn't passed. This is unexpected.")
            return {
                **state,
                "system_message": "Error: Trying to advance beyond the final stage without proper completion."
            }
    else:
        current_plot = story_plot_points[current_stage_index]
        print(f"Stage Manager: Preparing for stage {state['current_stage']} (challenge not yet passed or first setup for this stage).")
        print(f"Stage Manager: Plot for {state['current_stage']}: {current_plot}")
        return {
            **state, 
            "system_message": f"Now entering: {state['current_stage']}. Your goal: {current_plot}. Prepare for the upcoming challenge.",
            "challenge_passed": False, 
            "attempt_count": 0, 
            "hint_level": 0, 
            "current_challenge_description": None, 
            "expected_solution": None, 
            "current_hint": None, 
            "outcome_description": None,
            "user_response": None,
            "failed_responses": [], 
            "current_stage_plot_point": current_plot,
        } 