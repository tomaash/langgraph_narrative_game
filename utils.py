import re
from typing import Optional
import random

def parse_labeled_content(text: str, label: str) -> Optional[str]:
    # Looks for "LABEL: content until next LABEL: or end of string"
    # More robust parsing might be needed depending on LLM output consistency
    pattern = re.compile(rf"^{label.upper()}:\s*(.*?)(?=\n\S+:|$)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None 

def get_predefined_choice(predefined_settings: list[str]) -> str:
    """
    Prompts the user to choose a story setting from a predefined list,
    enter a custom setting, or opt for a random choice.

    Args:
        predefined_settings: A list of predefined story settings.

    Returns:
        The chosen story setting as a string.
    """
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
    return chosen_setting 