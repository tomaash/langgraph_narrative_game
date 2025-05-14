from typing import List, TypedDict, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # Example checkpointing
import os
from dotenv import load_dotenv
import google.generativeai as genai
import random # Added for random setting selection
import re # For parsing LLM output

# Import configurations
from config import GEMINI_API_KEY, GENERATE_MODEL_NAME, EVALUATE_MODEL_NAME
# Import models
from models import generate_model, evaluate_model
# Import prompts
from prompts import (
    get_challenge_generation_prompt,
    get_answer_evaluation_prompt,
    get_intervention_prompt,
    get_initial_story_plot_prompt
)

# Load environment variables from .env file
# load_dotenv() # Moved to config.py

# Access your API key
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Moved to config.py

# Configure the Gemini API key
# if GEMINI_API_KEY: # Logic moved to models.py
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         print(f"Gemini API Key configured successfully.")
#         # do not change!
#         # GENERATE_MODEL_NAME = os.getenv("GENERATE_MODEL_NAME", "gemini-1.5-flash-latest") # Model for story generation # Moved to config.py
#         # EVALUATE_MODEL_NAME = os.getenv("EVALUATE_MODEL_NAME", "gemini-1.5-flash-latest") # Model for user input evaluation # Moved to config.py
# 
#         generate_model = genai.GenerativeModel(GENERATE_MODEL_NAME)
#         evaluate_model = genai.GenerativeModel(EVALUATE_MODEL_NAME) # General purpose model
#         print(f"Gemini model '{GENERATE_MODEL_NAME}' initialized for generation tasks.")
#         print(f"Gemini model '{EVALUATE_MODEL_NAME}' initialized for evaluation and detailed content tasks.")
#     except Exception as e:
#         print(f"Error configuring Gemini API or initializing model: {e}")
#         generate_model = None
#         evaluate_model = None
# else:
#     print("Error: GEMINI_API_KEY not found. Make sure it's set in your .env file. Core LLM features will be skipped.")
#     generate_model = None
#     evaluate_model = None

# 1. State Definition
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

# Define the sequence of stages
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

    # New fields for structured story and challenges
    story_plot_points: Optional[List[str]] # Parsed plot points for each stage from initial summary
    current_stage_plot_point: Optional[str] # The plot point for the current stage
    
    current_challenge_description: Optional[str] # The narrative + challenge text
    expected_solution: Optional[str] # AI's view of a good solution
    current_hint: Optional[str] # New: For dynamically generated hint
    outcome_description: Optional[str] # New: For describing result of user action


# 2. Nodes (Agents/Functions) - Stubs

def stage_manager(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: stage_manager ---")

    story_plot_points = state.get('story_plot_points')
    if not story_plot_points or len(story_plot_points) != len(STORY_STAGES):
        print("Error: story_plot_points not found in state or has incorrect length. Cannot proceed.")
        # This is a critical error, game should probably not continue gracefully
        # For now, returning a state that might halt or indicate error
        return { 
            "system_message": "Critical Error: Missing or invalid story plot points.",
            # Fill other fields to avoid crashing on access, though game is broken
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

    # Handle initial call specifically to set the first stage
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
            "story_plot_points": story_plot_points, # Pass along all plot points
            "current_stage_plot_point": current_plot, # Set specific plot for this stage
            "current_challenge_description": None,
            "expected_solution": None,
            "current_hint": None,
            "outcome_description": None,
            "user_response": None,
        }

    # If current_stage is set, proceed with normal logic
    current_stage_index = STORY_STAGES.index(state['current_stage'])
    system_message = state.get('system_message', "")

    if state.get('challenge_passed', False):
        if state['current_stage'] == "RESOLUTION":
            print("Stage Manager: Game successfully completed!")
            # Update the final plot point for resolution for completeness
            final_plot_point = story_plot_points[current_stage_index]
            return {
                "system_message": "Congratulations! You have completed the story.",
                "current_stage_plot_point": final_plot_point,
                 # Preserve other necessary fields from state for END node or final display
                "story_plot_points": story_plot_points,
                "story_log": state.get("story_log", []),
                "current_stage": state["current_stage"],
                "user_response": state.get("user_response"),
                "attempt_count": state.get("attempt_count"),
                "hint_level": state.get("hint_level"),
                "max_failed_attempts_before_intervention": state.get("max_failed_attempts_before_intervention", 3),
                "challenge_passed": True,
                "failed_responses": state.get("failed_responses", [])
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
                "current_challenge_description": None,
                "expected_solution": None,
                "current_hint": None,
                "outcome_description": None,
                "user_response": None,
                "failed_responses": [],
                "story_plot_points": story_plot_points,
                "current_stage_plot_point": next_plot,
                "story_log": state.get("story_log", []) # Preserve story log
            }
        else:
            print("Stage Manager: Attempted to advance beyond RESOLUTION, but it wasn't passed. This is unexpected.")
            return {"system_message": "Error: Trying to advance beyond the final stage without proper completion.", "story_plot_points": story_plot_points, "current_stage_plot_point": story_plot_points[-1]}
    else:
        # This block now handles re-entry to a stage if a challenge was not passed,
        # OR if stage_manager is somehow entered without challenge_passed being True (e.g. first setup for the stage)
        current_plot = story_plot_points[current_stage_index]
        print(f"Stage Manager: Preparing for stage {state['current_stage']} (challenge not yet passed or first setup for this stage).")
        print(f"Stage Manager: Plot for {state['current_stage']}: {current_plot}")
        return {
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
            "story_plot_points": story_plot_points,
            "current_stage_plot_point": current_plot,
            "current_stage": state['current_stage'], # Ensure current_stage is passed through
            "story_log": state.get("story_log", []), # Preserve story log
            "max_failed_attempts_before_intervention": state.get("max_failed_attempts_before_intervention", 3) # Preserve max attempts
        }


def challenge_generator(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: challenge_generator for stage {state['current_stage']} ---")
    if not generate_model:
        print("Error: generate_model not initialized. Cannot generate challenge.")
        return {
            **state,
            "system_message": "Critical Error: Generation model not available.",
            "current_challenge_description": "Error: LLM offline.",
            "expected_solution": "Error: LLM offline.",
            "current_hint": None, "outcome_description": None
        }

    stage = state['current_stage']
    plot_point = state.get('current_stage_plot_point', "No specific plot point provided for this stage.")
    # [-4:]
    story_log_context = "\n".join(state.get('story_log', [-10])) # Use last 10 entries for richer context
    # print(f"-----  Story Log Context: -------")
    # print(f"{story_log_context}")
    # print(f"----- Story Log Context: end -----")
    if not story_log_context.strip():
        story_log_context = "The story is just beginning. This is the first scene."
    else:
        story_log_context = f"RECENT STORY EVENTS:\n{story_log_context}"

    prompt = get_challenge_generation_prompt(stage, plot_point, story_log_context)
    # (
    #     f"You are a master storyteller and game designer. Your task is to craft the next part of an interactive story. "
    #     f"The current story stage is: {stage}."
    #     f"The overarching plot point or goal for this stage is: '{plot_point}'."
    #     f"{story_log_context}" # This now includes how the previous stage resolved.
    #     f"\nBased on the RECENT STORY EVENTS and the current STAGE ({stage}) with its PLOT POINT ('{plot_point}'), generate the following. Ensure your narrative directly and logically continues from the last event in RECENT STORY EVENTS:"
    #     f"\nNARRATIVE: [Write 1-2 engaging paragraphs of narrative that continue the story, leading into a new challenge. This narrative MUST directly follow from the last event in RECENT STORY EVENTS.]"
    #     f"\nCHALLENGE: [Based on your NARRATIVE, subtly create a situation or question that presents an immediate challenge for the player. This should be a direct consequence of the NARRATIVE you just wrote.]"
    #     f"\nEXPECTED_SOLUTION: [Describe the general idea or key elements of a good player response to *your* CHALLENGE. This will guide AI evaluation.]"
    #     f"\nOutput ONLY these three sections, each clearly labeled on a new line. Keep the language engaging."
    # )

    print("\n--- Generating Challenge Content via LLM ---")

    try:
        response = generate_model.generate_content(prompt)
        raw_llm_response_text = response.text
        
        # print("\n--- Full LLM Response for Challenge Generation (DEBUG) ---")
        # print(raw_llm_response_text)
        # print("--- End LLM Response ---")

        if not response.parts:
            # print("Warning: Received an empty or blocked response from the model for challenge generation.")
            return { 
                **state,
                "system_message": "Error generating challenge content (empty/blocked LLM response).",
                "current_challenge_description": "Error: Could not generate challenge.",
                "expected_solution": "Error: Could not generate solution criteria.",
                "current_hint": None, "outcome_description": None
            }

        narrative = parse_labeled_content(raw_llm_response_text, "NARRATIVE")
        challenge_text = parse_labeled_content(raw_llm_response_text, "CHALLENGE")
        solution_criteria = parse_labeled_content(raw_llm_response_text, "EXPECTED_SOLUTION")

        if not all([narrative, challenge_text, solution_criteria]):
            # print("Error: Failed to parse NARRATIVE, CHALLENGE, or EXPECTED_SOLUTION from LLM response.")
            return { 
                **state,
                "system_message": "Error: Critical failure in parsing LLM response for challenge generation.",
                "current_challenge_description": (narrative or "Narrative failed.") + "\n" + (challenge_text or "Challenge failed."),
                "expected_solution": solution_criteria or "Solution criteria failed.",
                "current_hint": None, "outcome_description": None,
                "challenge_passed": False, "attempt_count": 0, "hint_level": 0, "failed_responses": []
            }
        
        current_challenge_full_text = f"{narrative}\n\nCHALLENGE: {challenge_text}"
        # The story_log from the state already contains the previous stage's resolution.
        # We just add the new challenge narrative here.
        updated_story_log = state.get('story_log', []) + [f"NARRATIVE FOR STAGE {stage}:\n{narrative}\n\nCHALLENGE: {challenge_text}"]

        # print(f"Challenge Generator: Parsed Narrative: '{narrative[:100]}...'")
        # print(f"Challenge Generator: Parsed Challenge: '{challenge_text[:100]}...'")
        # print(f"Challenge Generator: Parsed Solution Criteria: '{solution_criteria[:100]}...'")

        return {
            **state,
            "story_log": updated_story_log,
            "current_challenge_description": current_challenge_full_text,
            "expected_solution": solution_criteria,
            "attempt_count": 0,
            "hint_level": 0,
            "challenge_passed": False,
            "system_message": f"Challenge for {stage} generated and ready.", # Changed message
            "failed_responses": [],
            "user_response": None,
            "current_hint": None, 
            "outcome_description": None
        }

    except Exception as e:
        # print(f"Error during LLM challenge generation: {e}")
        return { 
            **state,
            "system_message": f"Error generating challenge for {stage}: {e}",
            "current_challenge_description": f"Error: Could not generate challenge for {stage} due to exception.",
            "expected_solution": "Error: Criteria generation failed.",
            "current_hint": None, "outcome_description": None
        }

def present_challenge_and_get_response(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: present_challenge_and_get_response ---")
    
    system_message = state.get("system_message")
    outcome_description_exists = bool(state.get("outcome_description"))
    challenge_previously_failed = not state.get("challenge_passed", True) and state.get("attempt_count", 0) > 0

    # Display custom message from evaluator (outcome + hint) if it exists from a previous failed attempt
    if system_message:
        print(f"\nSystem: {system_message}")
    
    # Only display the full story/challenge if it's the first attempt for this specific challenge instance,
    # or if there isn't a specific outcome/hint message to show (e.g. fresh stage setup).
    # The `attempt_count` being > 0 and `challenge_passed` being False indicates a retry on the *same* challenge.
    # `system_message` being present and containing a hint from a failure also signals a retry.

    # If it's a retry (attempt_count > 0, not passed) AND we have a system_message (which should be outcome+hint)
    # then we might have already printed it. The key is not to reprint the full challenge.
    
    # New simplified logic:
    # If there was a system message from the evaluator (outcome + hint), we've already printed it.
    # We only print the full challenge if it's effectively the first presentation of *this instance* of the challenge.
    # `attempt_count` being 0 signals the first time for *this* challenge instance.
    if state.get("attempt_count", 0) == 0:
        print("\n--- Your Story So Far ---")
        if state.get('story_log'):
            # The last entry in story_log should be the one set by challenge_generator
            # which includes the narrative and the challenge text itself.
            print(state['story_log'][-1])
        else:
            print("The story is about to begin...")
    elif not system_message: # Fallback if it's a retry but somehow system_message was cleared
        print("\n--- Continuing Challenge ---")
        if state.get('current_challenge_description'):
             print(state.get('current_challenge_description'))
        else:
            print("Error: No challenge description found for retry.")

    user_input = input("\nWhat do you do? > ")
    print(f"Present Challenge: User responded: '{user_input}'")
    
    # Clear system_message, current_hint, and outcome_description before next evaluation cycle for this node.
    # The evaluator will set new ones if the user fails again.
    return {
        "user_response": user_input,
        "system_message": None,
        "current_hint": None,
        "outcome_description": None
        # Preserve other parts of the state by not listing them or by passing **state if needed
    }

def answer_evaluator(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: answer_evaluator ---")
    if not evaluate_model:
        print("Error: evaluate_model not initialized. Cannot evaluate answer.")
        return { 
            **state,
            "challenge_passed": False, "attempt_count": state.get('attempt_count', 0) + 1,
            "system_message": "Error: Evaluation model not available. Assuming FAIL.",
            "failed_responses": state.get('failed_responses', []) + ([state.get('user_response')] if state.get('user_response') else []),
            "current_hint": "No hint available due to system error.",
            "outcome_description": "The outcome is uncertain due to a system error."
        }

    user_response = state.get('user_response')
    expected_solution_criteria = state.get('expected_solution')
    challenge_desc = state.get('current_challenge_description')
    current_attempt_count = state.get('attempt_count', 0)
    max_attempts = state.get('max_failed_attempts_before_intervention', 3)
    current_hint_level = state.get('hint_level', 0) # To guide hint strength
    story_log_context = "\n".join(state.get('story_log', [])[-3:])

    # --- Intervention Logic (Auto-Pass) ---
    if current_attempt_count >= max_attempts:
        print(f"Max attempts ({max_attempts}) reached. Generating intervention.")
        intervention_prompt = get_intervention_prompt(challenge_desc, story_log_context, current_attempt_count, user_response)
        # (
        #     f"The player is stuck on a challenge: '{challenge_desc}'.\n"
        #     f"The story context: '{story_log_context}'.\n"
        #     f"The player has failed {current_attempt_count} times. Their last attempt was: '{user_response}'.\n"
        #     f"Generate a short narrative (2-3 sentences) where an external event or another character intervenes to resolve the current situation for the player, allowing the story to progress. "
        #     f"This intervention should make it clear that the immediate challenge is overcome. Label it clearly as INTERVENTION_NARRATIVE:"
        #     f"\nINTERVENTION_NARRATIVE: [Your intervention narrative]"
        # )
        try:
            intervention_response = generate_model.generate_content(intervention_prompt)
            intervention_text = parse_labeled_content(intervention_response.text, "INTERVENTION_NARRATIVE")
            
            # print("\n--- Full LLM Response for Intervention (DEBUG) ---")
            # print(intervention_response.text)
            # print("--- End LLM Response ---")

            if not intervention_text:
                intervention_text = "Suddenly, a deus ex machina occurs! The way forward becomes clear."
                # print("Warning: Failed to parse intervention narrative, using fallback.")
            
            # Log the intervention
            updated_story_log = state.get('story_log', []) + [f"INTERVENTION: {intervention_text}"]
            return {
                **state,
                "challenge_passed": True,
                "system_message": f"After struggling for a while... {intervention_text} The path forward is now clear.",
                "story_log": updated_story_log,
                "outcome_description": intervention_text,
                "current_hint": None,
                "attempt_count": current_attempt_count + 1
            }
        except Exception as e:
            print(f"Error during intervention generation: {e}. Proceeding with generic auto-pass.")
            intervention_text = "Suddenly, the challenge resolves itself!"
            # Log the generic intervention
            updated_story_log = state.get('story_log', []) + [f"INTERVENTION: {intervention_text}"]
            return {
                **state,
                "challenge_passed": True,
                "system_message": intervention_text,
                "story_log": updated_story_log,
                "outcome_description": intervention_text,
                "current_hint": None,
                "attempt_count": current_attempt_count + 1
            }
    # --- End Intervention Logic ---

    updated_story_log = list(state.get('story_log', [])) # Get a mutable copy at the start of evaluation

    if not user_response:
        print("Answer Evaluator: No user response provided.")
        # Log this non-action for context if desired, though it might be noisy
        # updated_story_log.append("PLAYER ACTION: No response given.")
        return {
            **state,
            "challenge_passed": False, "attempt_count": current_attempt_count + 1,
            "system_message": "Evaluation: No response provided. That won't work!",
            "failed_responses": state.get('failed_responses', []),
            "current_hint": state.get("current_hint") or "Please try to describe what you do.",
            "outcome_description": "Nothing happens because no action was taken.",
            "story_log": updated_story_log # Pass through story_log
        }

    # Log the player's attempt before evaluation by LLM
    updated_story_log.append(f"PLAYER ATTEMPT (Attempt #{current_attempt_count + 1}): {user_response}")

    if not expected_solution_criteria or not challenge_desc:
        print("Error: Missing expected solution criteria or challenge description in state for answer_evaluator.")
        # Log error state for context
        updated_story_log.append(f"EVALUATION ERROR: Missing context (expected solution or challenge desc). Player response was: '{user_response}'.")
        return {
            **state, "challenge_passed": False, "attempt_count": current_attempt_count + 1,
            "system_message": "Error: Game error - missing context for evaluation. Assuming FAIL.",
            "failed_responses": state.get('failed_responses', []) + [user_response],
            "current_hint": "Cannot provide hint due to game error.",
            "outcome_description": "The evaluation could not proceed due to a game error.",
            "story_log": updated_story_log # Pass through story_log
        }

    # Determine hint directness based on hint_level or attempt_count
    # Hint level 0: subtle, 1: direct, 2: very direct
    hint_directness = "subtle"
    if current_hint_level == 1:
        hint_directness = "direct"
    elif current_hint_level >= 2:
        hint_directness = "very direct"

    prompt = get_answer_evaluation_prompt(
        story_log_context=story_log_context,
        challenge_desc=challenge_desc,
        # expected_solution_criteria=expected_solution_criteria, # Intentionally commented out
        user_response=user_response,
        failed_responses=state.get('failed_responses', []),
        current_hint_level=current_hint_level
    )
    # (
    #     f"You are an AI companion in a narrative game. Evaluate the player's response and provide an outcome description and a new hint."
    #     f"STORY CONTEXT SO FAR:\n{story_log_context}"
    #     f"\nCHALLENGE: '{challenge_desc}'."
    #     # f"\nEXPECTED SOLUTION CRITERIA: '{expected_solution_criteria}'."
    #     f"\nPLAYER RESPONSE: '{user_response}'."
    #     f"\nPLAYER'S PREVIOUS FAILED ATTEMPTS THIS CHALLENGE: {state.get('failed_responses', [])}"
    #     f"\nCURRENT HINT LEVEL (0=subtle, 1=direct, 2=very direct): {current_hint_level}"
    #     f"\nTASKS:"
    #     f"1. EVALUATION: Determine if the PLAYER RESPONSE is either at least a bit reasonable or creative. In that case PASS. Be very undemanding, almost anything goes. Just don't be a jerk. Only fail if it's some nonsensical chars or trolling or clearly stupid."
    #     f"2. OUTCOME: Describe what happens in the game world as a result of the player's failed action (1-2 sentences). Make it engaging."
    #     f"3. only if EVALUATION is FAIL - HINT: Provide a new, {hint_directness} hint. "
    #     f"\nOUTPUT FORMAT: Use the following labels EXACTLY, each on a new line. "
    #     f"EVALUATION: [PASS or FAIL]"
    #     f"OUTCOME: [Your outcome description]"
    #     f"HINT: [Your new hint, only if FAIL]"
    # )

# This hint is crucial. It MUST directly help the player understand how to meet the specific EXPECTED SOLUTION CRITERIA: '{expected_solution_criteria}'. Guide them towards these criteria without giving the exact answer, unless hint_directness is 'very direct'."

    print("\n--- Evaluating Answer and Generating Hint/Outcome via LLM ---")

    passed = False
    outcome_desc_update = state.get("outcome_description")
    current_hint_update = state.get("current_hint")
    evaluation_feedback = "FAIL (Evaluation Error)"
    # updated_story_log was already initialized above, and player attempt logged

    try:
        response = evaluate_model.generate_content(prompt)
        raw_llm_output = response.text
        # print("\n--- Full LLM Response for Answer Evaluation/Hint (DEBUG) ---")
        # print(raw_llm_output)
        # print("--- End LLM Response ---")

        if not response.parts:
            # print("Warning: Received an empty or blocked response from the model.")
            evaluation_feedback = "FAIL (LLM error)"
            outcome_desc_update = "The mists of uncertainty cloud the outcome..."
            current_hint_update = "The path is unclear; try a different approach."
        else:
            parsed_eval = parse_labeled_content(raw_llm_output, "EVALUATION")
            
            if parsed_eval and "PASS" in parsed_eval.upper():
                passed = True
                evaluation_feedback = "PASS"
                llm_pass_outcome = parse_labeled_content(raw_llm_output, "OUTCOME")
                outcome_desc_update = llm_pass_outcome or f"Your action ('{user_response}') was successful!"
                current_hint_update = None 
                updated_story_log.append(f"EVALUATION RESULT: PASS\nGAME OUTCOME: {outcome_desc_update}")
            elif parsed_eval and "FAIL" in parsed_eval.upper():
                passed = False
                evaluation_feedback = "FAIL"
                outcome_desc_update = parse_labeled_content(raw_llm_output, "OUTCOME") or "That didn't achieve the desired result."
                current_hint_update = parse_labeled_content(raw_llm_output, "HINT") or "Try focusing on the main objective mentioned in the challenge."
                updated_story_log.append(f"EVALUATION RESULT: FAIL\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update}")
            else:
                print(f"Warning: LLM evaluation response ('{parsed_eval}') was not a clear 'PASS' or 'FAIL'. Defaulting to FAIL.")
                evaluation_feedback = "FAIL (unclear LLM response)"
                outcome_desc_update = parse_labeled_content(raw_llm_output, "OUTCOME") or "The result of your action is ambiguous."
                current_hint_update = parse_labeled_content(raw_llm_output, "HINT") or "The way forward is hazy. Re-evaluate your approach."
                updated_story_log.append(f"EVALUATION RESULT: {evaluation_feedback} (Unclear LLM output)\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update}")
                
    except Exception as e:
        print(f"Error during LLM answer evaluation/hint generation: {e}")
        evaluation_feedback = f"FAIL (exception during evaluation)"
        outcome_desc_update = "An unexpected event occurred. The outcome is unknown."
        current_hint_update = "System error during hint generation. Please try again."
        updated_story_log.append(f"EVALUATION RESULT: EXCEPTION ({e})\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update}")

    new_attempt_count = current_attempt_count + 1
    updated_failed_responses = list(state.get('failed_responses', []))
    if not passed and user_response:
        updated_failed_responses.append(user_response)
    
    system_msg_for_user = f"{outcome_desc_update or ' '} "
    if not passed and current_hint_update:
        system_msg_for_user += f"\nHint: {current_hint_update}"
    elif passed:
        # For passed, system_message should just be the outcome. The stage_manager will announce progression.
        system_msg_for_user = outcome_desc_update or f"Your action was a success!"

    new_hint_level = current_hint_level if passed else current_hint_level + 1

    return {
        **state,
        "challenge_passed": passed,
        "attempt_count": new_attempt_count,
        "system_message": system_msg_for_user.strip(),
        "failed_responses": updated_failed_responses,
        "current_hint": current_hint_update if not passed else None,
        "outcome_description": outcome_desc_update,
        "hint_level": new_hint_level,
        "story_log": updated_story_log # Ensure the potentially updated story_log is returned
    }


# 3. Graph Definition
workflow = StateGraph(NarrativeGameState)

# Add nodes
workflow.add_node("stage_manager_node", stage_manager)
workflow.add_node("challenge_generator_node", challenge_generator)
workflow.add_node("present_challenge_node", present_challenge_and_get_response)
workflow.add_node("answer_evaluator_node", answer_evaluator)

# Set entry point
workflow.set_entry_point("stage_manager_node")

# Define edges
workflow.add_edge("stage_manager_node", "challenge_generator_node")
workflow.add_edge("challenge_generator_node", "present_challenge_node")
workflow.add_edge("present_challenge_node", "answer_evaluator_node")

# Conditional edges
def should_advance_or_retry(state: NarrativeGameState):
    if state['challenge_passed']:
        if state['current_stage'] == "RESOLUTION":
            print("Conditional: Challenge passed at RESOLUTION. Ending game.")
            return "END"
        print("Conditional: Challenge passed. Advancing to next stage.")
        return "stage_manager_node"
    else:
        # If challenge failed, go back to present the challenge again.
        # answer_evaluator will have set system_message with failure description + hint.
        print("Conditional: Challenge failed. Retrying challenge with new hint/description.")
        return "present_challenge_node"

workflow.add_conditional_edges(
    "answer_evaluator_node",
    should_advance_or_retry,
    {
        "stage_manager_node": "stage_manager_node",
        "present_challenge_node": "present_challenge_node", # Loop back to present_challenge
        "END": END
    }
)

# Compile the graph (with checkpointing)
# memory = SqliteSaver.from_conn_string(":memory:") # In-memory for testing
# app = workflow.compile(checkpointer=memory)
app = workflow.compile()


# Function to run the game loop (simplified)
def run_game():
    print("Starting the Narrative Game...")
    # Initial state. Stage manager will set the first stage.
    initial_state = {
        "current_stage": None, # Stage manager will set to EXPOSITION
        "story_log": [],
        "attempt_count": 0,
        "hint_level": 0,
        "max_failed_attempts_before_intervention": 3, # Give first hint after 1st fail, second after 2nd fail.
        "challenge_passed": False,
        "failed_responses": [],
        "story_plot_points": None,
        "current_stage_plot_point": None,
        "current_challenge_description": None,
        "expected_solution": None,
        "current_hint": None,
        "outcome_description": None
    }
    
    config = {"configurable": {"thread_id": "narrative-game-thread-1"}} # Example thread_id for checkpointing

    # Stream events to see the flow
    # The input to stream is the initial state or subsequent inputs for nodes that require them.
    # For a state machine that manages its own inputs internally (like ours mostly does after init),
    # the initial call is with the starting state.
    # Subsequent calls might not be needed if graph flows automatically, or if a node needs external input
    # that isn't part of the state (which present_challenge_and_get_response handles via input()).

    # The loop here is mostly for manual control/demonstration if the graph doesn't reach END naturally
    # or if we want to inspect state between "human" steps.
    # LangGraph's stream will run until it hits END or needs input it can't get from state.

    print("\\n--- Game Start ---")
    current_state_stream = app.stream(initial_state, config=config)
    for event in current_state_stream:
        # event is a dictionary where keys are node names and values are the output of that node
        # print(f"Event: {event}")
        # We can inspect the full state if needed from the event,
        # for example, the last event before it stops or asks for input.
        # The `present_challenge_node` will pause for `input()`.
        # The loop will continue after `input()` is provided and that node finishes.
        pass # The print statements within nodes will show progress.
    
    final_state = app.get_state(config)
    print("\\n--- Game End ---")
    print("Final Story Log:")
    for entry in final_state.values['story_log']:
        print(entry)
    print(f"Final system message: {final_state.values.get('system_message')}")


# Helper function to parse labeled content from LLM response
def parse_labeled_content(text: str, label: str) -> Optional[str]:
    # Looks for "LABEL: content until next LABEL: or end of string"
    # More robust parsing might be needed depending on LLM output consistency
    pattern = re.compile(rf"^{label.upper()}:\s*(.*?)(?=\n\S+:|$)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def generate_initial_story_plot(story_setting: str) -> tuple[Optional[str], Optional[List[str]]]:
    if not generate_model:
        print("Skipping initial story plot generation as the generate_model was not initialized.")
        return None, None

    stages_for_prompt = "\n".join([f"{stage.upper()}: [Brief plot point or key event for this stage]" for stage in STORY_STAGES])

    prompt = get_initial_story_plot_prompt(story_setting, STORY_STAGES)
    # (
    #     f"You are a master storyteller. Your task is to outline a short story plot based on Freytag's Pyramid. "
    #     f"The story should be set in: '{story_setting}'. "
    #     f"For each of the following story stages, provide a concise plot point (1-2 sentences) that describes the key event or development for that stage. "
    #     f"Ensure each plot point clearly flows into the next, creating a coherent narrative arc. "
    #     f"Format your response *exactly* as follows, with each stage on a new line followed by its plot point:"
    #     f"\nEXPOSITION: [Your plot point for Exposition]"
    #     f"\nINCITING_INCIDENT: [Your plot point for Inciting Incident]"
    #     f"\n(and so on for all stages: {', '.join(STORY_STAGES)})"
    #     f"\nKeep the language engaging but simple. The overall collection of plot points should form a cohesive story outline."
    #     f"\nExample for EXPOSITION only: EXPOSITION: A young scholar in a quiet monastery discovers a hidden map pointing to a legendary library said to contain all lost knowledge."
    #     f"\nNow, generate the plot points for all stages for the setting: '{story_setting}'."
    # )

    print("\n--- Generating Initial Story Plot Points ---")
    # print(f"Prompt for initial plot: {prompt}") # Keep this for debugging if needed

    full_response_text = ""
    plot_points_list = []

    try:
        response = generate_model.generate_content(prompt)
        # print("\n--- Full LLM Response for Initial Plot Points (DEBUG) ---")
        # print(response.text)
        # print("--- End LLM Response ---")

        if response.parts:
            full_response_text = response.text
            current_plot_points = []
            for stage_name in STORY_STAGES:
                point = parse_labeled_content(full_response_text, stage_name)
                if point:
                    current_plot_points.append(point)
                    # print(f"Parsed plot for {stage_name}: {point[:100]}...") # Commenting out parsed component print
                else:
                    # print(f"Warning: Could not parse plot point for stage: {stage_name} from LLM response.") # Commenting out warning
                    current_plot_points.append(f"Plot point for {stage_name} needs to be manually checked or regenerated.")
            
            if len(current_plot_points) == len(STORY_STAGES):
                plot_points_list = current_plot_points
            else:
                # print(f"Error: Parsed {len(current_plot_points)} plot points, but expected {len(STORY_STAGES)}. Check LLM output format.") # Commenting out error
                plot_points_list = [f"Error in parsing for {s}" for s in STORY_STAGES]

            # The 'summary' is now the collection of these plot points, or the raw text if parsing fails to be useful
            return full_response_text, plot_points_list
        else:
            print(f"Warning: Received an empty or blocked response from the model for initial plot generation.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return "Initial plot could not be generated (empty/blocked response).", None

    except Exception as e:
        print(f"Error during initial plot generation: {e}")
        return f"Initial plot generation failed due to an error: {e}", None

if __name__ == "__main__":
    predefined_settings = [
        "a bustling medieval marketplace",
        "a derelict spaceship drifting in the cosmos",
        "a hidden magical school",
        "a sun-scorched desert oasis",
        "a mist-shrouded Victorian London",
        "a vibrant coral reef teeming with life",
        "a quiet village at the foot of a dragon's mountain"
    ]

    chosen_setting = None
    while chosen_setting is None:
        print("\nPlease choose a story setting:")
        for i, setting in enumerate(predefined_settings):
            print(f"{i+1}. {setting}")
        
        user_input = input(
            "Enter a number (1-7) to choose a predefined setting, \n" 
            "type your own custom setting (must be more than 5 characters), \n"
            "or leave blank for a random choice: "
        ).strip()

        if not user_input:  # Empty input
            chosen_setting = random.choice(predefined_settings)
            print(f"No input. Using random setting: {chosen_setting}")
        elif user_input.isdigit():
            try:
                choice_num = int(user_input)
                if 1 <= choice_num <= len(predefined_settings):
                    chosen_setting = predefined_settings[choice_num - 1]
                    print(f"You chose setting {choice_num}: {chosen_setting}")
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(predefined_settings)}.")
            except ValueError:
                # Should not happen if isdigit() is true, but as a safeguard
                print("Invalid input. Please enter a number or type your custom setting.")
        elif len(user_input) > 5:
            chosen_setting = user_input
            print(f"Using your custom setting: {chosen_setting}")
        else: # Input is not a number, not empty, and not > 5 chars
            print("Invalid input. Custom settings must be longer than 5 characters, or choose a valid number.")
            # Loop will continue

    # Generate initial story plot points
    full_plot_summary_text, parsed_plot_points = generate_initial_story_plot(story_setting=chosen_setting)

    # if full_plot_summary_text:
    #     print("\n--- Generated Story Plot Outline (Full Text) ---")
    #     print(full_plot_summary_text)
    
    if parsed_plot_points:
        pass
        # print("\n--- Parsed Individual Plot Points ---")
        # for i, point in enumerate(parsed_plot_points):
        #     print(f"Stage {STORY_STAGES[i]}: {point}")
    else:
        print("Could not generate or parse initial plot points. Game cannot proceed with AI-driven challenges.")
        # Decide if you want to exit or fall back to old non-LLM challenges
        exit() 

    # Initial state for the game, now including plot points
    initial_game_state_params = {
        "current_stage": None, 
        "story_log": [],
        "attempt_count": 0,
        "hint_level": 0,
        "max_failed_attempts_before_intervention": 3,
        "challenge_passed": False,
        "failed_responses": [],
        "story_plot_points": parsed_plot_points, # Add the parsed plot points here
        "current_stage_plot_point": None,
        "current_challenge_description": None,
        "expected_solution": None,
        "current_hint": None,
        "outcome_description": None
    }

    # Then run the game
    print("\n--- Starting Interactive Game ---")
    
    config = {"configurable": {"thread_id": "narrative-game-thread-1"}} # Example thread_id for checkpointing

    # Pass the fully prepared initial state to the stream method
    current_state_stream = app.stream(initial_game_state_params, config=config)
    for event in current_state_stream:
        pass 
    
    final_state = app.get_state(config)
    print("\n--- Game End ---")
    if final_state.values.get('story_log'):
        print("Final Story Log:")
        for entry in final_state.values['story_log']:
            print(entry)
    print(f"Final system message: {final_state.values.get('system_message')}")

