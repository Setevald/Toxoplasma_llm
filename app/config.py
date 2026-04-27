"""
Configuration module for Toxoplasma LLM Backend.

All model and generation-related settings are centralized here
to make adjustments easier for future maintainers.
"""

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Generation parameters
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.2
TOP_P = 0.85
