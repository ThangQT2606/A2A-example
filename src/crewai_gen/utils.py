from openai import OpenAI

def initialize_client(token: str = None) -> OpenAI:
    """Initialize the OpenAI client."""
    if not token:
        raise ValueError("Token is required")
    return OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token
    )

