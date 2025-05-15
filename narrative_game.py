from typing import List, TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

import os
from dotenv import load_dotenv
import google.generativeai as genai
import random 
import re 
from langchain_core.runnables import RunnableConfig 

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
    story_log_context = "\n".join(state.get('story_log', [])[-10:]) # Use last 10 entries for richer context
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
        else:
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
        # print(f"Error during LLM challenge generation: {e}")
        return {
            **state,
            "system_message": f"Error generating content for {stage}: {e}",
            "current_challenge_description": f"Error: Could not generate content for {stage} due to exception.",
            "expected_solution": "Error: Criteria generation failed." if stage != "RESOLUTION" else None,
            "challenge_passed": stage == "RESOLUTION" # Auto-pass for resolution if LLM fails here
        }

def present_challenge_and_get_response(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: present_challenge_and_get_response --- Stage: {state.get('current_stage')}")

    # If it's RESOLUTION stage and challenge_passed is True (meaning conclusion was generated)
    if state.get('current_stage') == "RESOLUTION" and state.get('challenge_passed') == True:
        print("\n--- Story Conclusion ---")
        conclusion_text = state.get('current_challenge_description', "The story has ended.")
        print(conclusion_text)
        # No user input needed, just pass the state along. answer_evaluator will then lead to END.
        return {**state} # Ensure all state is passed through

    system_message = state.get("system_message")

    # Display custom message from evaluator (outcome + hint) if it exists from a previous failed attempt
    if system_message:
        print(f"\nSystem: {system_message}")
    
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
        **state, # Preserve all existing state fields
        "user_response": user_input,
        "system_message": None, # Clear for next eval cycle
        "current_hint": None, # Clear for next eval cycle
        "outcome_description": None # Clear for next eval cycle
    }

def answer_evaluator(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: answer_evaluator --- Stage: {state.get('current_stage')}")

    # If it's RESOLUTION and challenge_passed is True, means conclusion was handled.
    # Pass through to allow should_advance_or_retry to send to END.
    if state.get('current_stage') == "RESOLUTION" and state.get('challenge_passed') == True:
        print("Answer Evaluator: RESOLUTION stage with challenge_passed=True. Passing through for game end.")
        return {**state} # Pass the state as is

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
    
    config = RunnableConfig(
        recursion_limit=500, 
        configurable={"thread_id": "narrative-game-thread-1"}
    )

    print("\\n--- Game Start ---")
    current_state_stream = app.stream(initial_state, config=config)
    last_event = None
    for event in current_state_stream:
        last_event = event # Capture the last event
        pass 
    
    final_state_values = None
    try:
        final_state_check = app.get_state(config) # config still holds the thread_id
        if final_state_check:
            final_state_values = final_state_check.values
    except ValueError as e:
        if "No checkpointer set" in str(e):
            # This is expected if no checkpointer is set
            pass
        else:
            print(f"Error calling app.get_state(): {e}") # Other ValueErrors
    except Exception as e:
        # Catch other potential exceptions from app.get_state if any
        print(f"Unexpected error calling app.get_state(): {e}")

    if final_state_values is None and last_event:
        print("Using last event from stream as final state.")
        if isinstance(last_event, dict) and last_event:
            final_state_values = last_event

    if final_state_values:
        print("\\n--- Game End ---")
        print("Final Story Log:")
        for entry in final_state_values['story_log']:
            print(entry)
        print(f"Final system message: {final_state_values.get('system_message')}")
    else:
        print("Final state could not be retrieved. Game ended without completing.")


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

    prompt = get_initial_story_plot_prompt(story_setting, STORY_STAGES)

    print("\n--- Generating Initial Story Plot Points ---")

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
    
    config = RunnableConfig(
        recursion_limit=500, 
        configurable={"thread_id": "narrative-game-thread-1"}
    )

    # Pass the fully prepared initial state to the stream method
    current_state_stream = app.stream(initial_game_state_params, config=config)
    for event in current_state_stream:
        pass 
    
