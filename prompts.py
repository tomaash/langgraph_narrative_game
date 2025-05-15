from typing import List, Optional

def get_challenge_generation_prompt(stage: str, plot_point: str, story_log_context: str) -> str:
    if stage == "RESOLUTION":
        return (
            f"You are a master storyteller. The story has reached its RESOLUTION stage."
            f"The overarching plot point for this final stage was: '{plot_point}'."
            f"The story unfolded as follows (recent events):\n{story_log_context}"
            f"\nBased on the plot point and the story events, write a compelling and satisfying conclusion to the narrative."
            f"\nCONCLUSION_NARRATIVE: [Write 1-3 engaging paragraphs that resolve the story and provide a sense of closure. This conclusion MUST logically follow from the story events and the resolution plot point.]"
            f"\nOutput ONLY the CONCLUSION_NARRATIVE section, clearly labeled."
        )
    else:
        return (
            f"You are a master storyteller and game designer. Your task is to craft the next part of an interactive story. "
            f"The current story stage is: {stage}."
            f"The overarching plot point or goal for this stage is: '{plot_point}'."
            f"{story_log_context}" # This now includes how the previous stage resolved.
            f"\nBased on the RECENT STORY EVENTS and the current STAGE ({stage}) with its PLOT POINT ('{plot_point}'), generate the following. Ensure your narrative directly and logically continues from the last event in RECENT STORY EVENTS:"
            f"\nNARRATIVE: [Write 1-2 engaging paragraphs of narrative that continue the story, leading into a new challenge. This narrative MUST directly follow from the last event in RECENT STORY EVENTS.]"
            f"\nCHALLENGE: [Based on your NARRATIVE, subtly create a situation or question that presents an immediate challenge for the player. This should be a direct consequence of the NARRATIVE you just wrote.]"
            f"\nEXPECTED_SOLUTION: [Describe the general idea or key elements of a good player response to *your* CHALLENGE. This will guide AI evaluation.]"
            f"\nOutput ONLY these three sections, each clearly labeled on a new line. Keep the language engaging."
        )

def get_answer_evaluation_prompt(
    story_log_context: str, 
    challenge_desc: str, 
    # expected_solution_criteria: str, 
    user_response: str, 
    failed_responses: List[str], 
    current_hint_level: int
) -> str:
    hint_directness = "subtle"
    if current_hint_level == 1:
        hint_directness = "direct"
    elif current_hint_level >= 2:
        hint_directness = "very direct"

    return (
        f"You are an AI companion in a narrative game. Evaluate the player's response and provide an outcome description and a new hint."
        f"STORY CONTEXT SO FAR:\n{story_log_context}"
        f"\nCHALLENGE: '{challenge_desc}'."
        # f"\nEXPECTED SOLUTION CRITERIA: '{expected_solution_criteria}'."
        f"\nPLAYER RESPONSE: '{user_response}'."
        f"\nPLAYER'S PREVIOUS FAILED ATTEMPTS THIS CHALLENGE: {failed_responses}"
        f"\nCURRENT HINT LEVEL (0=subtle, 1=direct, 2=very direct): {current_hint_level}"
        f"\nTASKS:"
        f"1. EVALUATION: Determine if the PLAYER RESPONSE is either at least a bit reasonable or creative. In that case PASS. Be very undemanding, almost anything goes. Just don't be a jerk. Only fail if it's some nonsensical chars or trolling or clearly stupid."
        f"2. OUTCOME: Describe what happens in the game world as a result of the player's failed action (1-2 sentences). Make it engaging."
        f"3. only if EVALUATION is FAIL - HINT: Provide a new, {hint_directness} hint. "
        f"\nOUTPUT FORMAT: Use the following labels EXACTLY, each on a new line. "
        f"EVALUATION: [PASS or FAIL]"
        f"OUTCOME: [Your outcome description]"
        f"HINT: [Your new hint, only if FAIL]"
    )

def get_intervention_prompt(challenge_desc: str, story_log_context: str, current_attempt_count: int, user_response: Optional[str]) -> str:
    return (
        f"The player is stuck on a challenge: '{challenge_desc}'.\n"
        f"The story context: '{story_log_context}'.\n"
        f"The player has failed {current_attempt_count} times. Their last attempt was: '{user_response}'.\n"
        f"Generate a short narrative (2-3 sentences) where an external event or another character intervenes to resolve the current situation for the player, allowing the story to progress. "
        f"This intervention should make it clear that the immediate challenge is overcome. Label it clearly as INTERVENTION_NARRATIVE:"
        f"\nINTERVENTION_NARRATIVE: [Your intervention narrative]"
    )

def get_initial_story_plot_prompt(story_setting: str, story_stages: List[str]) -> str:
    stages_for_prompt = "\n".join([f"{stage.upper()}: [Brief plot point or key event for this stage]" for stage in story_stages])
    return (
        f"You are a master storyteller. Your task is to outline a short story plot based on Freytag's Pyramid. "
        f"The story should be set in: '{story_setting}'. "
        f"For each of the following story stages, provide a concise plot point (1-2 sentences) that describes the key event or development for that stage. "
        f"Ensure each plot point clearly flows into the next, creating a coherent narrative arc. "
        f"Format your response *exactly* as follows, with each stage on a new line followed by its plot point:"
        f"\nEXPOSITION: [Your plot point for Exposition]"
        f"\nINCITING_INCIDENT: [Your plot point for Inciting Incident]"
        f"\n(and so on for all stages: {', '.join(story_stages)})"
        f"\nKeep the language engaging but simple. The overall collection of plot points should form a cohesive story outline."
        f"\nExample for EXPOSITION only: EXPOSITION: A young scholar in a quiet monastery discovers a hidden map pointing to a legendary library said to contain all lost knowledge."
        f"\nNow, generate the plot points for all stages for the setting: '{story_setting}'."
    ) 