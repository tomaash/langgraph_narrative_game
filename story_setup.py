from typing import Optional, List, Tuple

from models import generate_model
from prompts import get_initial_story_plot_prompt
from narrative_game_types import STORY_STAGES
from utils import parse_labeled_content

def generate_initial_story_plot(story_setting: str) -> Tuple[Optional[str], Optional[List[str]]]:
    if not generate_model:
        print("Skipping initial story plot generation as the generate_model was not initialized.")
        return None, None

    # Ensure STORY_STAGES is a list of strings if it's not already, for the prompt function
    # No, get_initial_story_plot_prompt in prompts.py already handles STORY_STAGES correctly.
    prompt = get_initial_story_plot_prompt(story_setting, STORY_STAGES) 

    print("\n--- Generating Initial Story Plot Points ---")

    full_response_text = ""
    plot_points_list = []

    try:
        response = generate_model.generate_content(prompt)
        if response.parts:
            full_response_text = response.text
            current_plot_points = []
            for stage_name in STORY_STAGES:
                point = parse_labeled_content(full_response_text, stage_name)
                if point:
                    current_plot_points.append(point)
                else:
                    # print(f"Warning: Could not parse plot point for stage: {stage_name} from LLM response.") # Optional: for debugging
                    current_plot_points.append(f"Plot point for {stage_name} needs to be manually checked or regenerated.")
            
            if len(current_plot_points) == len(STORY_STAGES):
                plot_points_list = current_plot_points
            else:
                # print(f"Error: Parsed {len(current_plot_points)} plot points, but expected {len(STORY_STAGES)}. Check LLM output format.") # Optional: for debugging
                plot_points_list = [f"Error in parsing for {s}" for s in STORY_STAGES]
            return full_response_text, plot_points_list
        else:
            print(f"Warning: Received an empty or blocked response from the model for initial plot generation.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return "Initial plot could not be generated (empty/blocked response).", None

    except Exception as e:
        print(f"Error during initial plot generation: {e}")
        return f"Initial plot generation failed due to an error: {e}", None 