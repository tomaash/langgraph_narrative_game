from narrative_game_types import NarrativeGameState

def present_challenge_and_get_response(state: NarrativeGameState) -> NarrativeGameState:
    print(f"--- Entering Node: present_challenge_and_get_response --- Stage: {state.get('current_stage')}")

    if state.get('current_stage') == "RESOLUTION" and state.get('challenge_passed') == True:
        print("\n--- Story Conclusion ---")
        conclusion_text = state.get('current_challenge_description', "The story has ended.")
        print(conclusion_text)
        return {**state}

    system_message = state.get("system_message")
    
    if system_message:
        print(f"\nSystem: {system_message}")
    
    if state.get("attempt_count", 0) == 0:
        print("\n--- Your Story So Far ---")
        if state.get('story_log') and state['story_log']:
            print(state['story_log'][-1])
        else:
            print("The story is about to begin...")
    elif not system_message: 
        print("\n--- Continuing Challenge ---")
        if state.get('current_challenge_description'):
             print(state.get('current_challenge_description'))
        else:
            print("Error: No challenge description found for retry.")

    user_input = input("\nWhat do you do? > ")
    print(f"Present Challenge: User responded: '{user_input}'")
    
    return {
        **state, 
        "user_response": user_input,
        "system_message": None, 
        "current_hint": None, 
        "outcome_description": None 
    } 