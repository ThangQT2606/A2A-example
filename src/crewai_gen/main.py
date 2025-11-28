"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import logging
import os

import click
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_content import ContentGenerationAgent
from agent_executor import ContentGenerationAgentExecutor
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    """Entry point for the A2A + CrewAI Content generation sample."""
    try:
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variables not set.'
            )

        capabilities = AgentCapabilities(streaming=False, push_notifications=True)
        skill = AgentSkill(
            id='content_generator',
            name='Content Generator',
            description=(
                'Generate content based on the user\'s text prompt.'
            ),
            tags=['generate content'],
            examples=['Generate a content for a blog post about the benefits of using AI.'],
        )

        agent_host_url = (
            f'http://{host}:{port}/'
        )
        agent_card = AgentCard(
            name='Content Generator Agent',
            description=(
                'Generate content based on the user\'s text prompt.'
            ),
            url=agent_host_url,
            version='1.0.0',
            default_input_modes=ContentGenerationAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=ContentGenerationAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                        config_store=push_config_store)
        request_handler = DefaultRequestHandler(
            agent_executor=ContentGenerationAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        import uvicorn

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()