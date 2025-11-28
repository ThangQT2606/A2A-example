import os
import sys
import click
import httpx
import uvicorn
import asyncio
from typing import Any
from contextlib import asynccontextmanager

from langchain.tools import tool
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


from a2a.server.apps import A2AStarletteApplication, A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from weather_executor import WeatherExecutor
from weather_agent import WeatherAgent

load_dotenv(find_dotenv(), override=True)

app_context: dict[str, Any] = {}

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 10002
DEFAULT_LOG_LEVEL = 'info'

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = None
weather_agent = None
executor = None

@asynccontextmanager
async def app_lifespan(context: dict[str, Any]):
    """Manages the lifecycle of LLM and Weather Agent"""
    print('Lifespan: Initializing Weather Agent...')

    try:
        if GOOGLE_API_KEY is None:
            raise ValueError('GOOGLE_API_KEY is not set')
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                             google_api_key=GOOGLE_API_KEY, 
                             temperature=0.5, 
                             top_p=1)
        weather_agent = WeatherAgent(llm=llm)
        executor = WeatherExecutor(agent=weather_agent)
        context['executor'] = executor
        context['weather_agent'] = weather_agent
        context['llm'] = llm
        yield  # Application runs here
    except Exception as e:
        print(f'Lifespan: Error during initialization: {e}', file=sys.stderr)
        # If an exception occurs, mcp_client_instance might exist and need cleanup.
        # The finally block below will handle this.
        raise
    finally:
        # Clear the application context
        print('Lifespan: Clearing application context.')
        context.clear()

def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    """Command Line Interface to start the Weather Agent server."""
    async def run_server_async():
        async with app_lifespan(app_context):
            if app_context.get('executor') is None:
                raise ValueError('Executor is not set')
            
            httpx_client = httpx.AsyncClient()
            push_config_store = InMemoryPushNotificationConfigStore()
            push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                        config_store=push_config_store)
            request_handler = DefaultRequestHandler(
                agent_executor=app_context.get('executor'),
                task_store=InMemoryTaskStore(),
                push_config_store=push_config_store,
                push_sender=push_sender,
            )
            agent_host_url = (
            os.getenv('HOST_OVERRIDE')
            if os.getenv('HOST_OVERRIDE')
            else f'http://{host}:{port}/'
        )

            # Create the A2AServer instance
            # a2a_server = A2AStarletteApplication(
            #     agent_card=get_agent_card(agent_host_url),
            #     http_handler=request_handler,
            # )
            a2a_server = A2AFastAPIApplication(
                agent_card=get_agent_card(agent_host_url),
                http_handler=request_handler,
            )

            # Get the ASGI app from the A2AServer instance
            asgi_app = a2a_server.build()

            config = uvicorn.Config(
                app=asgi_app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                lifespan='auto',
            )

            uvicorn_server = uvicorn.Server(config)

            print(
                f'Starting Uvicorn server at http://{host}:{port} with log-level {log_level}...'
            )
            try:
                await uvicorn_server.serve()
            except KeyboardInterrupt:
                print('Server shutdown requested (KeyboardInterrupt).')
            finally:
                print('Uvicorn server has stopped.')
                # The app_lifespan's finally block handles mcp_client shutdown

    try:
        asyncio.run(run_server_async())
    except RuntimeError as e:
        if 'cannot be called from a running event loop' in str(e):
            print(
                'Critical Error: Attempted to nest asyncio.run(). This should have been prevented.',
                file=sys.stderr,
            )
        else:
            print(f'RuntimeError in main: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred in main: {e}', file=sys.stderr)
        sys.exit(1)


def get_agent_card(app_url: str):
    """Returns the Agent Card for the Currency Agent."""
    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    skill = AgentSkill(
        id='weather_search',
        name='Search weather information',
        description='Helps with weather information search',
        tags=['weather information'],
        examples=[
            'Please find the weather in Tokyo'
        ],
    )

    return AgentCard(
        name='Weather Agent',
        description='Helps with searching weather information',
        url=app_url,
        version='1.0.0',
        default_input_modes=WeatherExecutor.SUPPORTED_CONTENT_TYPES,
        default_output_modes=WeatherExecutor.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
        supports_authenticated_extended_card=True,
    )

    
@click.command()
@click.option(
    '--host',
    'host',
    default=DEFAULT_HOST,
    help='Hostname to bind the server to.',
)
@click.option(
    '--port',
    'port',
    default=DEFAULT_PORT,
    type=int,
    help='Port to bind the server to.',
)
@click.option(
    '--log-level',
    'log_level',
    default=DEFAULT_LOG_LEVEL,
    help='Uvicorn log level.',
)
def cli(host: str, port: int, log_level: str):
    main(host, port, log_level)


if __name__ == '__main__':
    main()