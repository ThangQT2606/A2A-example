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

USERNAME = "admin"
PASSWORD = "123456"

STREAMING_MODE = True  # True: streaming, False: invoke


def extract_text_from_chunk(data: dict[str, Any]) -> str:
    """Extract main text answer from a response or streaming chunk."""
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


async def get_jwt_token() -> str:
    """Login to server and get JWT token."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/auth/login",
            json={"username": USERNAME, "password": PASSWORD},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError("Failed to get JWT token")
        logger.info("Login successful, token obtained")
        return token


async def send_message(agent_card, client: httpx.AsyncClient, streaming: bool):
    """Send a test message to agent."""
    # Prepare user message
    user_message = {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is the weather in Kim Giang - Thanh Xuan - Ha Noi (Vietnam)?"}],
        "message_id": uuid4().hex,
    }

    a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)

    if streaming:
        logger.info("Sending message in STREAMING mode...")
        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(message=user_message)
        )
        stream_response = a2a_client.send_message_streaming(streaming_request)

        async for chunk in stream_response:
            data = chunk.model_dump(mode="json", exclude_none=True)
            answer = extract_text_from_chunk(data)
            if answer:
                print("\n=== Answer (streaming) ===")
                print(answer)

            if data.get("result", {}).get("final", False):
                logger.info("Task completed!")

    else:
        logger.info("Sending message in INVOKE mode...")
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(message=user_message)
        )
        response = await a2a_client.send_message(request)
        data = response.model_dump(mode="json", exclude_none=True)
        answer = extract_text_from_chunk(data)
        if answer:
            print("\n=== Answer (invoke) ===")
            print(answer)


async def run_client(streaming: bool):
    token = await get_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Dùng 1 client duy nhất
    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        # Lấy agent card
        resolver = A2ACardResolver(httpx_client=client, base_url=BASE_URL)
        agent_card = await resolver.get_agent_card()
        logger.info("Agent card fetched successfully")
        logger.info(agent_card.model_dump_json(indent=2, exclude_none=True))

        # Gửi message
        await send_message(agent_card, client, streaming=streaming)


@click.command()
@click.option("--streaming", is_flag=True, default=False, help="Enable streaming mode")
def cli(streaming: bool):
    asyncio.run(run_client(streaming=streaming))


if __name__ == "__main__":
    cli()
