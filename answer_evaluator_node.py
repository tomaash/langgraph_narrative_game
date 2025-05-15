from narrative_game_types import NarrativeGameState
from models import evaluate_model, generate_model 
from prompts import get_intervention_prompt, get_answer_evaluation_prompt
from utils import parse_labeled_content

def answer_evaluator(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: answer_evaluator --- Stage: {state.get('current_stage')}")

    if state.get('current_stage') == "RESOLUTION" and state.get('challenge_passed') == True:
        print("Answer Evaluator: RESOLUTION stage with challenge_passed=True. Passing through for game end.")
        return {**state}

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
    expected_solution_criteria = state.get('expected_solution') # Though not used in current prompt, keep for context
    challenge_desc = state.get('current_challenge_description')
    current_attempt_count = state.get('attempt_count', 0)
    max_attempts = state.get('max_failed_attempts_before_intervention', 3)
    current_hint_level = state.get('hint_level', 0)
    story_log_context = "\n".join(state.get('story_log', [])[-3:])

    if current_attempt_count >= max_attempts:
        print(f"Max attempts ({max_attempts}) reached. Generating intervention.")
        intervention_prompt_text = get_intervention_prompt(challenge_desc, story_log_context, current_attempt_count, user_response)
        try:
            intervention_response = generate_model.generate_content(intervention_prompt_text)
            intervention_text = parse_labeled_content(intervention_response.text, "INTERVENTION_NARRATIVE")
            if not intervention_text:
                intervention_text = "Suddenly, a deus ex machina occurs! The way forward becomes clear."
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

    updated_story_log = list(state.get('story_log', []))

    if not user_response:
        print("Answer Evaluator: No user response provided.")
        updated_story_log.append(f"PLAYER ACTION (Attempt #{current_attempt_count + 1}): No response given.")
        return {
            **state,
            "challenge_passed": False, "attempt_count": current_attempt_count + 1,
            "system_message": "Evaluation: No response provided. That won't work!",
            "failed_responses": state.get('failed_responses', []),
            "current_hint": state.get("current_hint") or "Please try to describe what you do.",
            "outcome_description": "Nothing happens because no action was taken.",
            "story_log": updated_story_log
        }

    updated_story_log.append(f"PLAYER ATTEMPT (Attempt #{current_attempt_count + 1}): {user_response}")

    if not expected_solution_criteria or not challenge_desc: # expected_solution_criteria might be None for RESOLUTION stage, handled by first check.
        # This check is more for regular challenge stages where these are expected.
        if state.get('current_stage') != "RESOLUTION": # Only error out if not in resolution (where criteria aren't used)
            print("Error: Missing expected solution criteria or challenge description in state for answer_evaluator.")
            updated_story_log.append(f"EVALUATION ERROR: Missing context (expected solution or challenge desc). Player response was: '{user_response}'.")
            return {
                **state, "challenge_passed": False, "attempt_count": current_attempt_count + 1,
                "system_message": "Error: Game error - missing context for evaluation. Assuming FAIL.",
                "failed_responses": state.get('failed_responses', []) + [user_response],
                "current_hint": "Cannot provide hint due to game error.",
                "outcome_description": "The evaluation could not proceed due to a game error.",
                "story_log": updated_story_log
            }

    prompt_text = get_answer_evaluation_prompt(
        story_log_context=story_log_context,
        challenge_desc=challenge_desc,
        user_response=user_response,
        failed_responses=state.get('failed_responses', []),
        current_hint_level=current_hint_level
    )

    print("\n--- Evaluating Answer and Generating Hint/Outcome via LLM ---")
    passed = False
    outcome_desc_update = state.get("outcome_description")
    current_hint_update = state.get("current_hint")
    # updated_story_log was already initialized above, and player attempt logged

    try:
        response = evaluate_model.generate_content(prompt_text)
        raw_llm_output = response.text
        if not response.parts:
            outcome_desc_update = "The mists of uncertainty cloud the outcome..."
            current_hint_update = "The path is unclear; try a different approach."
            updated_story_log.append(f"EVALUATION RESULT: FAIL (LLM error)\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update if current_hint_update else 'N/A'}")
        else:
            parsed_eval = parse_labeled_content(raw_llm_output, "EVALUATION")
            if parsed_eval and "PASS" in parsed_eval.upper():
                passed = True
                llm_pass_outcome = parse_labeled_content(raw_llm_output, "OUTCOME")
                outcome_desc_update = llm_pass_outcome or f"Your action ('{user_response}') was successful!"
                current_hint_update = None 
                updated_story_log.append(f"EVALUATION RESULT: PASS\nGAME OUTCOME: {outcome_desc_update}")
            elif parsed_eval and "FAIL" in parsed_eval.upper():
                outcome_desc_update = parse_labeled_content(raw_llm_output, "OUTCOME") or "That didn't achieve the desired result."
                current_hint_update = parse_labeled_content(raw_llm_output, "HINT") or "Try focusing on the main objective mentioned in the challenge."
                updated_story_log.append(f"EVALUATION RESULT: FAIL\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update}")
            else:
                print(f"Warning: LLM evaluation response ('{parsed_eval}') was not a clear 'PASS' or 'FAIL'. Defaulting to FAIL.")
                outcome_desc_update = parse_labeled_content(raw_llm_output, "OUTCOME") or "The result of your action is ambiguous."
                current_hint_update = parse_labeled_content(raw_llm_output, "HINT") or "The way forward is hazy. Re-evaluate your approach."
                updated_story_log.append(f"EVALUATION RESULT: FAIL (Unclear LLM output)\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update}")
                
    except Exception as e:
        print(f"Error during LLM answer evaluation/hint generation: {e}")
        outcome_desc_update = "An unexpected event occurred. The outcome is unknown."
        current_hint_update = "System error during hint generation. Please try again."
        updated_story_log.append(f"EVALUATION RESULT: EXCEPTION ({e})\nGAME OUTCOME: {outcome_desc_update}\nHINT PROVIDED: {current_hint_update}")

    new_attempt_count = current_attempt_count + 1
    updated_failed_responses = list(state.get('failed_responses', []))
    if not passed and user_response:
        updated_failed_responses.append(user_response)
    
    system_msg_for_user = f"{outcome_desc_update or ' '}".strip()
    if not passed and current_hint_update:
        system_msg_for_user += f"\nHint: {current_hint_update}"
    elif passed and not outcome_desc_update: # Ensure success has a message
        system_msg_for_user = f"Your action ('{user_response}') was successful!"

    new_hint_level = current_hint_level if passed else current_hint_level + 1

    return {
        **state,
        "challenge_passed": passed,
        "attempt_count": new_attempt_count,
        "system_message": system_msg_for_user,
        "failed_responses": updated_failed_responses,
        "current_hint": current_hint_update if not passed else None,
        "outcome_description": outcome_desc_update,
        "hint_level": new_hint_level,
        "story_log": updated_story_log
    } 