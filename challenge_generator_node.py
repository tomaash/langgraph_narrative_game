from narrative_game_types import NarrativeGameState
from models import generate_model
from prompts import get_challenge_generation_prompt
from utils import parse_labeled_content

def challenge_generator(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: challenge_generator for stage {state['current_stage']} ---")
    if not generate_model:
        print("Error: generate_model not initialized. Cannot generate challenge.")
        return {
            **state,
            "system_message": "Critical Error: Generation model not available.",
            "current_challenge_description": "Error: LLM offline.",
            "expected_solution": "Error: LLM offline.", # Default, will be None for RESOLUTION handled below
            "current_hint": None, "outcome_description": None
        }

    stage = state['current_stage']
    plot_point = state.get('current_stage_plot_point', "No specific plot point provided for this stage.")
    story_log_context = "\n".join(state.get('story_log', [])[-10:])
    if not story_log_context.strip():
        story_log_context = "The story is just beginning. This is the first scene."
    else:
        story_log_context = f"RECENT STORY EVENTS:\n{story_log_context}"

    prompt = get_challenge_generation_prompt(stage, plot_point, story_log_context)

    print("\n--- Generating Challenge Content via LLM ---")

    try:
        response = generate_model.generate_content(prompt)
        raw_llm_response_text = response.text

        if not response.parts:
            return {
                **state,
                "system_message": "Error generating content (empty/blocked LLM response).",
                "current_challenge_description": "Error: Could not generate content.",
                "expected_solution": "Error: Could not generate solution criteria." if stage != "RESOLUTION" else None,
                "challenge_passed": stage == "RESOLUTION", # Auto-pass for resolution if LLM fails here
            }

        if stage == "RESOLUTION":
            conclusion_narrative = parse_labeled_content(raw_llm_response_text, "CONCLUSION_NARRATIVE")
            if not conclusion_narrative:
                conclusion_narrative = "The story reaches an end, but the details are shrouded in mist."
                print("Warning: Failed to parse CONCLUSION_NARRATIVE from LLM response for RESOLUTION stage.")
            
            updated_story_log = state.get('story_log', []) + [f"STORY CONCLUSION (STAGE {stage}):\n{conclusion_narrative}"]
            return {
                **state,
                "story_log": updated_story_log,
                "current_challenge_description": conclusion_narrative, # Display the conclusion
                "expected_solution": None,
                "challenge_passed": True, # RESOLUTION means challenge is passed
                "system_message": "The story concludes...",
                "user_response": None, # No user response needed for conclusion
                "attempt_count": 0, 
                "hint_level": 0
            }
        else: # For stages other than RESOLUTION
            narrative = parse_labeled_content(raw_llm_response_text, "NARRATIVE")
            challenge_text = parse_labeled_content(raw_llm_response_text, "CHALLENGE")
            solution_criteria = parse_labeled_content(raw_llm_response_text, "EXPECTED_SOLUTION")

            if not all([narrative, challenge_text, solution_criteria]):
                return {
                    **state,
                    "system_message": "Error: Critical failure in parsing LLM response for challenge generation.",
                    "current_challenge_description": (narrative or "Narrative failed.") + "\n" + (challenge_text or "Challenge failed."),
                    "expected_solution": solution_criteria or "Solution criteria failed.",
                    "challenge_passed": False
                }
            
            current_challenge_full_text = f"{narrative}\n\nCHALLENGE: {challenge_text}"
            updated_story_log = state.get('story_log', []) + [f"NARRATIVE FOR STAGE {stage}:\n{narrative}\n\nCHALLENGE: {challenge_text}"]

            return {
                **state,
                "story_log": updated_story_log,
                "current_challenge_description": current_challenge_full_text,
                "expected_solution": solution_criteria,
                "attempt_count": 0,
                "hint_level": 0,
                "challenge_passed": False,
                "system_message": f"Challenge for {stage} generated and ready.",
                "failed_responses": [],
                "user_response": None,
                "current_hint": None, 
                "outcome_description": None
            }

    except Exception as e:
        return {
            **state,
            "system_message": f"Error generating content for {stage}: {e}",
            "current_challenge_description": f"Error: Could not generate content for {stage} due to exception.",
            "expected_solution": "Error: Criteria generation failed." if stage != "RESOLUTION" else None,
            "challenge_passed": stage == "RESOLUTION" # Auto-pass for resolution if LLM fails here
        } 