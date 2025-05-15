# LangGraph Narrative Game

This project implements a "Choose Your Own Adventure" style game with a pedagogical structure using LangGraph.
The game guides the user through a story based on Freytag's Pyramid, presenting challenges at each stage.

## Features

- Story progression through defined stages (Exposition, Rising Action, Climax, etc.)
- AI-generated challenges and narrative content
- AI-evaluated user responses
- Hint system for users who are stuck
- Structured approach using LangGraph for managing game state and flow.

## Setup

1.  Clone the repository (or create the project).

2.  Create `.env` file from `.env.sample` with your own Gemini api key

3.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Run the game:
    ```bash
    python main.py
    ```
