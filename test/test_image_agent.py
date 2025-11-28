"""Test client for Image Generation Agent.

This test client is specifically designed to test the image generation agent.
It handles both text responses and image artifacts returned by the agent.
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from uuid import uuid4
from typing import Any

import click
import httpx
from PIL import Image
from io import BytesIO

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, SendStreamingMessageRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:10001"


def extract_artifacts_from_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract artifacts (images, files) from a response or streaming chunk.
    Supports both invoke and streaming (artifact-update chunks).
    """
    artifacts = []
    try:
        result = data.get("result", {})
        
        # Streaming artifact-update
        artifact = result.get("artifact")
        if artifact:
            artifacts.append(artifact)

        # Normal invoke response - artifacts at top level
        response_artifacts = result.get("artifacts", [])
        if response_artifacts:
            artifacts.extend(response_artifacts)

        # Check for artifacts in task object
        task = result.get("task")
        if task:
            task_artifacts = task.get("artifacts", [])
            if task_artifacts:
                artifacts.extend(task_artifacts)

        return artifacts
    except Exception as e:
        logger.error(f"Failed to extract artifacts: {e}")
        return []


def extract_text_from_response(data: dict[str, Any]) -> str:
    """
    Extract text messages from a response or streaming chunk.
    """
    try:
        result = data.get("result", {})
        
        # Check for text in task status message
        task = result.get("task")
        if task:
            status = task.get("status", {})
            message = status.get("message")
            if message:
                parts = message.get("parts", [])
                for part in parts:
                    if part.get("kind") == "text" and part.get("text"):
                        return part.get("text", "")
                    # Check for root.text (TextPart)
                    root = part.get("root", {})
                    if root.get("text"):
                        return root.get("text", "")
            
            # Check for text in task messages
            messages = task.get("messages", [])
            for msg in messages:
                if msg.get("role") == "agent":
                    # Check parts directly in message
                    parts = msg.get("parts", [])
                    for part in parts:
                        if part.get("kind") == "text" and part.get("text"):
                            return part.get("text", "")
                    # Check content.parts
                    content = msg.get("content", {})
                    if content:
                        parts = content.get("parts", [])
                        for part in parts:
                            if part.get("kind") == "text" and part.get("text"):
                                return part.get("text", "")

        # Check for text in artifacts
        artifacts = extract_artifacts_from_response(data)
        for artifact in artifacts:
            parts = artifact.get("parts", [])
            for part in parts:
                if part.get("kind") == "text" and part.get("text"):
                    return part.get("text", "")
                # Check for root.text (TextPart)
                root = part.get("root", {})
                if root.get("text"):
                    return root.get("text", "")

        return ""
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        return ""


def save_and_display_image(artifact: dict[str, Any], output_dir: Path = None) -> str | None:
    """Save and display image from artifact. Returns path to saved image."""
    parts = artifact.get("parts", [])
    for part in parts:
        if part.get("kind") == "file":
            file_data = part.get("file", {})
            file_id = file_data.get("id", "unknown")
            mime_type = file_data.get("mime_type", "image/png")
            name = file_data.get("name", "unknown")
            bytes_data = file_data.get("bytes")
            
            if not bytes_data:
                logger.warning("No image data found in artifact")
                return None
            
            try:
                # Decode base64 image data
                image_bytes = base64.b64decode(bytes_data)
                
                # Determine file extension from MIME type
                ext_map = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                }
                ext = ext_map.get(mime_type, ".png")
                
                # Create output directory if not provided
                if output_dir is None:
                    output_dir = Path("test_output")
                output_dir.mkdir(exist_ok=True)
                
                # Generate filename
                filename = f"{name}{ext}" if name != "unknown" else f"image_{file_id}{ext}"
                filepath = output_dir / filename
                
                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                # Display image info
                size_kb = len(image_bytes) / 1024
                print(f"\n{'='*60}")
                print(f"Image Artifact Information:")
                print(f"  ID: {file_id}")
                print(f"  Name: {name}")
                print(f"  MIME Type: {mime_type}")
                print(f"  Size: {size_kb:.2f} KB ({len(image_bytes)} bytes)")
                print(f"  Saved to: {filepath.absolute()}")
                print(f"{'='*60}")
                
                # Try to open image with PIL to verify it's valid
                try:
                    img = Image.open(BytesIO(image_bytes))
                    print(f"  Image dimensions: {img.size[0]}x{img.size[1]} pixels")
                    print(f"  Image mode: {img.mode}")
                    
                    # Try to open image in default viewer (platform dependent)
                    try:
                        if os.name == 'nt':  # Windows
                            os.startfile(filepath)
                            print(f"  ✓ Image opened in default viewer")
                        elif os.name == 'posix':  # macOS and Linux
                            import platform
                            if platform.system() == 'Darwin':  # macOS
                                os.system(f'open "{filepath}"')
                            else:  # Linux
                                os.system(f'xdg-open "{filepath}"')
                            print(f"  ✓ Image opened in default viewer")
                        else:
                            print(f"  Please open the image manually: {filepath.absolute()}")
                    except Exception as e:
                        logger.info(f"  Note: Could not open image automatically: {e}")
                        print(f"  Please open the image manually: {filepath.absolute()}")
                except Exception as e:
                    logger.warning(f"Could not verify image: {e}")
                
                return str(filepath.absolute())
                
            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                return None
    
    return None


async def run_image_test(streaming: bool, prompt: str):
    """Run a test for image generation."""
    async with httpx.AsyncClient(timeout=120.0) as httpx_client:
        try:
            # Fetch agent card
            logger.info(f"Connecting to agent at {BASE_URL}...")
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
            agent_card = await resolver.get_agent_card()
            logger.info("Successfully fetched agent card.")
            logger.info(f"Agent: {agent_card.name}")
            logger.info(f"Description: {agent_card.description}")

            # Initialize client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            logger.info("A2AClient initialized.")

            # Prepare user message
            user_message = {
                "role": "user",
                "parts": [{"kind": "text", "text": prompt}],
                "message_id": uuid4().hex,
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"Test Prompt: {prompt}")
            logger.info(f"Mode: {'STREAMING' if streaming else 'INVOKE'}")
            logger.info(f"{'='*60}\n")

            if streaming:
                logger.info("Running in STREAMING mode...")
                streaming_request = SendStreamingMessageRequest(
                    id=str(uuid4()), params=MessageSendParams(message=user_message)
                )
                stream_response = client.send_message_streaming(streaming_request)

                artifacts_found = []
                async for chunk in stream_response:
                    data = chunk.model_dump(mode="json", exclude_none=True)
                    
                    # Extract and print text messages
                    text = extract_text_from_response(data)
                    if text:
                        print(f"\n[Agent Message]: {text}")

                    # Extract and process artifacts
                    artifacts = extract_artifacts_from_response(data)
                    for artifact in artifacts:
                        if artifact not in artifacts_found:
                            artifacts_found.append(artifact)
                            save_and_display_image(artifact)

                    # Check if task is completed
                    if data.get("result", {}).get("final", False):
                        logger.info("\n✓ Task completed!")
                        break

                if not artifacts_found:
                    logger.warning("⚠ No image artifacts found in streaming response!")
                    # Debug: Check last chunk structure
                    logger.info("Checking last chunk structure...")
                    import json
                    # Note: We can't access last chunk here, but we can log

            else:
                logger.info("Running in INVOKE mode...")
                request = SendMessageRequest(
                    id=str(uuid4()), params=MessageSendParams(message=user_message)
                )
                response = await client.send_message(request)
                data = response.model_dump(mode="json", exclude_none=True)
                
                # Extract and print text messages
                text = extract_text_from_response(data)
                if text:
                    print(f"\n[Agent Message]: {text}")

                # Extract and process artifacts
                artifacts = extract_artifacts_from_response(data)
                if artifacts:
                    logger.info(f"✓ Found {len(artifacts)} artifact(s)")
                    saved_images = []
                    for artifact in artifacts:
                        image_path = save_and_display_image(artifact)
                        if image_path:
                            saved_images.append(image_path)
                    
                    if saved_images:
                        print(f"\n{'='*60}")
                        print(f"✓ Successfully saved {len(saved_images)} image(s):")
                        for img_path in saved_images:
                            print(f"  - {img_path}")
                        print(f"{'='*60}\n")
                else:
                    logger.warning("⚠ No image artifacts found in response!")
                    # Debug: Check response structure
                    result = data.get("result", {})
                    task = result.get("task")
                    if task:
                        logger.info(f"Task ID: {task.get('id')}")
                        logger.info(f"Task artifacts: {task.get('artifacts', [])}")
                        logger.info(f"Task status: {task.get('status', {}).get('state')}")
                    logger.info("\nFull response structure:")
                    import json
                    print(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error during test: {e}")
            import traceback
            traceback.print_exc()


@click.command()
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Enable streaming mode"
)
@click.option(
    "--prompt",
    default="Generate a photorealistic image of a cat sitting on a windowsill.",
    help="Image generation prompt"
)
def cli(streaming: bool, prompt: str):
    """Test client for Image Generation Agent."""
    asyncio.run(run_image_test(streaming=streaming, prompt=prompt))


if __name__ == "__main__":
    cli()