from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig 
from narrative_game_types import STORY_STAGES, NarrativeGameState
from stage_manager_node import stage_manager
from present_challenge_node import present_challenge_and_get_response
from answer_evaluator_node import answer_evaluator
from challenge_generator_node import challenge_generator
from story_setup import generate_initial_story_plot
from utils import get_predefined_choice
from prompts import predefined_settings

# Graph
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

# Compile the graph // checkpointer=SqliteSaver.from_conn_string(":memory:")
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

    print("\n--- Game Start ---")
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
        print("\n--- Game End ---")
        print("Final Story Log:")
        for entry in final_state_values['story_log']:
            print(entry)
        print(f"Final system message: {final_state_values.get('system_message')}")
    else:
        print("Final state could not be retrieved. Game ended without completing.")




if __name__ == "__main__":
    chosen_setting = get_predefined_choice(predefined_settings)
    full_plot_summary_text, parsed_plot_points = generate_initial_story_plot(chosen_setting)
    
    if parsed_plot_points:
        pass
        print("\n--- Parsed Individual Plot Points ---")
        for i, point in enumerate(parsed_plot_points):
            print(f"Stage {STORY_STAGES[i]}: {point}")
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
    
