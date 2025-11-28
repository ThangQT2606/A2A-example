import asyncio
import logging
from uuid import uuid4
from typing import Any

import click
import httpx

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, SendStreamingMessageRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:10002"
API_KEY = "my_default_secret_key"

def extract_text_from_chunk(data: dict[str, Any]) -> str:
    """
    Extract the main text answer from a response or streaming chunk.
    Supports both invoke and streaming (artifact-update chunks).
    """
    try:
        # Streaming artifact-update
        artifact = data.get("result", {}).get("artifact")
        if artifact:
            parts = artifact.get("parts", [])
            texts = [p.get("text") for p in parts if p.get("kind") == "text"]
            return "\n".join(texts) if texts else ""

        # Normal invoke response
        artifacts = data.get("result", {}).get("artifacts", [])
        if artifacts:
            parts = artifacts[0].get("parts", [])
            texts = [p.get("text") for p in parts if p.get("kind") == "text"]
            return "\n".join(texts) if texts else ""

        return ""
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        return ""


async def run_client(streaming: bool):
    headers = {"X-API-Key": API_KEY}
    try: 
        async with httpx.AsyncClient(timeout=60, headers=headers) as httpx_client:
            # Fetch agent card with authentication
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
            agent_card = await resolver.get_agent_card()
            logger.info("Successfully fetched agent card.")
            logger.info(agent_card.model_dump_json(indent=2, exclude_none=True))

            # Initialize client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            logger.info("A2AClient initialized.")

            # Prepare user message
            user_message = {
                "role": "user",
                "parts": [{"kind": "text", "text": "What is the weather in California (USA)?"}],
                "message_id": uuid4().hex,
            }

            if streaming:
                logger.info("Running in STREAMING mode...")
                streaming_request = SendStreamingMessageRequest(
                    id=str(uuid4()), params=MessageSendParams(message=user_message)
                )
                stream_response = client.send_message_streaming(streaming_request)

                async for chunk in stream_response:
                    data = chunk.model_dump(mode="json", exclude_none=True)
                    print(data)  # Print full response
                    answer = extract_text_from_chunk(data)
                    if answer:
                        print("\n=== Answer (streaming) ===")
                        print(answer)

                    if data.get("result", {}).get("final", False):
                        logger.info("Task completed!")

            else:
                logger.info("Running in INVOKE mode...")
                request = SendMessageRequest(
                    id=str(uuid4()), params=MessageSendParams(message=user_message)
                )
                response = await client.send_message(request)
                data = response.model_dump(mode="json", exclude_none=True)
                print(data)  # Print full response
                answer = extract_text_from_chunk(data)
                if answer:
                    print("\n=== Answer (invoke) ===")
                    print(answer)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

@click.command()
@click.option("--streaming", is_flag=True, default=False, help="Enable streaming mode")
def cli(streaming: bool):
    asyncio.run(run_client(streaming=streaming))


if __name__ == "__main__":
    cli()
