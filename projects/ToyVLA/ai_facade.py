"""
AI Facade for integrating with Google Gemini API for vision-language-action tasks.
"""
import os
import json
import logging
from typing import Optional
from PIL import Image
import google.generativeai as genai
from thesaurus import ActionOutput, ActionType, SkillType, FailureType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIFacade:
    """
    Facade for interacting with Google Gemini Vision-Language Model.

    Handles image analysis and structured action prediction based on natural language commands.
    """

    SYSTEM_PROMPT = """You are a vision-language-action (VLA) system controller for a robotic manipulation task.
This is a TOY EXAMPLE for demonstration purposes - pretend that execution is in progress.

Your role is to:
1. Analyze the provided image of the environment (typically a tabletop with objects)
2. Interpret the natural language command
3. Identify relevant objects in the scene
4. Determine the appropriate action to execute

Guidelines:
- Be specific when identifying objects (use color, shape, size as descriptors)
- If you cannot clearly identify the target object, indicate an appropriate failure mode
- Estimate completion percentage based on visual cues (e.g., 0% if just starting, 100% if already complete)
- Choose the most relevant skill currently being employed
- Provide brief reasoning for your decision

Remember: This is a simulation. Output structured predictions as if the robot is executing the command."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the AI Facade.

        Args:
            api_key: Google API key (if None, will try to get from environment)
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be provided or set in environment")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"AIFacade initialized with model: {model_name}")

    def predict_action(
        self,
        image: Image.Image,
        command: str,
        context: Optional[str] = None
    ) -> ActionOutput:
        """
        Predict the action to take based on the image and command.

        Args:
            image: PIL Image of the scene
            command: Natural language command to execute
            context: Optional additional context about the task

        Returns:
            ActionOutput with structured prediction
        """
        try:
            # Build the prompt
            user_prompt = f"""Natural Language Command: "{command}"

Please analyze the image and provide a structured action plan to execute this command.

Output your response as a JSON object with the following structure:
{{
    "action": "one of: move, pick_up, place, count, point_to, push, none",
    "target": "description of the target object",
    "destination": "description of destination (null if not applicable)",
    "completion": <number between 0-100>,
    "current_skill": "one of: object_detection, spatial_reasoning, manipulation, counting, navigation",
    "failure": "one of: object_not_found, ambiguous_reference, unreachable_location, physical_constraint_violation, insufficient_visual_information, none",
    "reasoning": "brief explanation of your decision"
}}"""

            if context:
                user_prompt = f"Additional Context: {context}\n\n{user_prompt}"

            # Generate response with structured output
            response = self.model.generate_content(
                [self.SYSTEM_PROMPT, user_prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for more deterministic outputs
                )
            )

            # Parse the response
            response_text = response.text.strip()
            logger.info(f"Raw Gemini response: {response_text}")

            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            # Parse JSON and validate with Pydantic
            action_dict = json.loads(json_text)
            action_output = ActionOutput(**action_dict)

            logger.info(f"Predicted action: {action_output.action} on {action_output.target}")
            return action_output

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            # Return a failure output
            return ActionOutput(
                action=ActionType.NONE,
                target="unknown",
                destination=None,
                completion=0.0,
                current_skill=SkillType.OBJECT_DETECTION,
                failure=FailureType.INSUFFICIENT_VISUAL_INFORMATION,
                reasoning=f"Failed to parse model response: {str(e)}"
            )

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return ActionOutput(
                action=ActionType.NONE,
                target="unknown",
                destination=None,
                completion=0.0,
                current_skill=SkillType.OBJECT_DETECTION,
                failure=FailureType.INSUFFICIENT_VISUAL_INFORMATION,
                reasoning=f"Error during prediction: {str(e)}"
            )

    def predict_action_with_schema(
        self,
        image: Image.Image,
        command: str,
        context: Optional[str] = None
    ) -> ActionOutput:
        """
        Predict action using Gemini's structured output feature (if available).

        This method uses the response_schema parameter for more reliable structured outputs.

        Args:
            image: PIL Image of the scene
            command: Natural language command to execute
            context: Optional additional context about the task

        Returns:
            ActionOutput with structured prediction
        """
        try:
            # Build the prompt
            user_prompt = f"""Natural Language Command: "{command}"

Analyze the image and provide a structured action plan to execute this command."""

            if context:
                user_prompt = f"Additional Context: {context}\n\n{user_prompt}"

            # Define the schema for structured output
            schema = {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["move", "pick_up", "place", "count", "point_to", "push", "none"]
                    },
                    "target": {"type": "string"},
                    "destination": {"type": ["string", "null"]},
                    "completion": {"type": "number", "minimum": 0, "maximum": 100},
                    "current_skill": {
                        "type": "string",
                        "enum": ["object_detection", "spatial_reasoning", "manipulation", "counting", "navigation"]
                    },
                    "failure": {
                        "type": "string",
                        "enum": [
                            "object_not_found",
                            "ambiguous_reference",
                            "unreachable_location",
                            "physical_constraint_violation",
                            "insufficient_visual_information",
                            "none"
                        ]
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["action", "target", "completion", "current_skill", "failure"]
            }

            # Generate response with structured output schema
            response = self.model.generate_content(
                [self.SYSTEM_PROMPT, user_prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )

            # Parse the response directly as JSON
            response_text = response.text.strip()
            logger.info(f"Structured Gemini response: {response_text}")

            action_dict = json.loads(response_text)
            action_output = ActionOutput(**action_dict)

            logger.info(f"Predicted action: {action_output.action} on {action_output.target}")
            return action_output

        except Exception as e:
            logger.error(f"Error during structured prediction: {e}")
            # Fall back to regular prediction method
            logger.info("Falling back to regular prediction method")
            return self.predict_action(image, command, context)
