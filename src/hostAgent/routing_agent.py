# pylint: disable=logging-fstring-interpolation
import asyncio
import json
import os
import uuid

from typing import Any

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)
from dotenv import load_dotenv, find_dotenv
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)


load_dotenv(find_dotenv(), override=True)


def convert_part(part: Part, tool_context: ToolContext):
    """Convert a part to text. Only text parts are supported."""
    if part.type == 'text':
        return part.text

    return f'Unknown type: {part.type}'


def convert_parts(parts: list[Part], tool_context: ToolContext):
    """Convert parts to text."""
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def extract_text_from_task(task: Task) -> str:
    """Extract text from Task artifacts.
    
    Args:
        task: The Task object returned from remote agent.
        
    Returns:
        Extracted text from task artifacts, or empty string if no text found.
    """
    if not task or not task.artifacts:
        return ""
    
    texts = []
    for artifact in task.artifacts:
        if artifact.parts:
            for part in artifact.parts:
                # Check if part has root with text attribute (TextPart)
                if hasattr(part, 'root'):
                    root = part.root
                    # Handle TextPart with text attribute
                    if isinstance(root, TextPart) or (hasattr(root, 'text') and root.text):
                        if hasattr(root, 'text') and root.text:
                            texts.append(root.text)
                # Fallback: check if part has text directly
                elif hasattr(part, 'text') and part.text:
                    texts.append(part.text)
    
    return "\n".join(texts) if texts else ""


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'kind': 'text', 'text': text}],
            'message_id': uuid.uuid4().hex,
        },
    }

    if task_id:
        payload['message']['taskId'] = task_id

    if context_id:
        payload['message']['contextId'] = context_id
    return payload


class RoutingAgent:
    """The Routing agent.

    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''

    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        """Asynchronous part of initialization."""
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        async with httpx.AsyncClient(timeout=120) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(
                    client, address
                )  # Constructor is sync
                try:
                    card = (
                        await card_resolver.get_agent_card()
                    )  # get_agent_card is async

                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(
                        f'ERROR: Failed to get agent card from {address}: {e}'
                    )
                except Exception as e:  # Catch other potential errors
                    print(
                        f'ERROR: Failed to initialize connection for {address}: {e}'
                    )

        # Populate self.agents using the logic from original __init__ (via list_remote_agents)
        agent_info = []
        for agent_detail_dict in self.list_remote_agents():
            agent_info.append(json.dumps(agent_detail_dict))
        self.agents = '\n'.join(agent_info)

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ) -> 'RoutingAgent':
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        """Create an instance of the RoutingAgent."""
        return Agent(
            # model='gemini-2.5-flash-lite',
            model=LiteLlm(model='openai/gpt-4.1-mini'),
            name='Routing_agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                'This Routing agent orchestrates the decomposition of the user asking for weather forecast or generate content'
            ),
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate the root instruction for the RoutingAgent."""
        current_agent = self.check_active_agent(context)
        return f"""
        **Role:** You are an expert Routing Delegator. Your primary function is to accurately delegate user inquiries regarding weather or content generation to the appropriate specialized remote agents.

        **Core Directives:**

        * **Task Delegation:** Utilize the `send_message` function to assign actionable tasks to remote agents.
        * **Contextual Awareness for Remote Agents:** If a remote agent repeatedly requests user confirmation, assume it lacks access to the full conversation history. In such cases, enrich the task description with all necessary contextual information relevant to that specific agent.
        * **Autonomous Agent Engagement:** Never seek user permission before engaging with remote agents. If multiple agents are required to fulfill a request, connect with them directly without requesting user preference or confirmation.
        * **Transparent Communication:** Always present the complete and detailed response from the remote agent to the user.
        * **User Confirmation Relay:** If a remote agent asks for confirmation, and the user has not already provided it, relay this confirmation request to the user.
        * **Focused Information Sharing:** Provide remote agents with only relevant contextual information. Avoid extraneous details.
        * **No Redundant Confirmations:** Do not ask remote agents for confirmation of information or actions.
        * **Tool Reliance:** Strictly rely on available tools to address user requests. Do not generate responses based on assumptions. If information is insufficient, request clarification from the user.
        * **Prioritize Recent Interaction:** Focus primarily on the most recent parts of the conversation when processing requests.
        * **Active Agent Prioritization:** If an active agent is already engaged, route subsequent related requests to that agent using the appropriate task update tool.

        **Agent Roster:**

        * Available Agents: `{self.agents}`
        * Currently Active Seller Agent: `{current_agent['active_agent']}`
                """

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            'session_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'active_agent' in state
        ):
            return {'active_agent': f'{state["active_agent"]}'}
        return {'active_agent': 'None'}

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ):
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            if 'session_id' not in state:
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f'Found agent card: {card.model_dump(exclude_none=True)}')
            print('=' * 100)
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_message(
        self, agent_name: str, task: str, tool_context: ToolContext
    ):
        """Sends a task to remote seller agent.

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.

        Yields:
            A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f'Agent {agent_name} not found')
        state = tool_context.state
        state['active_agent'] = agent_name
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        
        # Only use existing task_id if it's already in state (for continuing tasks)
        # For new tasks, don't send taskId - let the server create it
        task_id = state.get('task_id')

        if 'context_id' in state:
            context_id = state['context_id']
        else:
            context_id = str(uuid.uuid4())
            state['context_id'] = context_id

        message_id = ''
        metadata = {}
        if 'input_message_metadata' in state:
            metadata.update(**state['input_message_metadata'])
            if 'message_id' in state['input_message_metadata']:
                message_id = state['input_message_metadata']['message_id']
        if not message_id:
            message_id = str(uuid.uuid4())

        # Create message payload matching test_client.py format
        user_message = {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': task}
            ],
            'message_id': message_id,
        }

        # Only add taskId if it's an existing task (for task updates)
        if task_id:
            user_message['taskId'] = task_id

        # Add contextId to maintain conversation context
        if context_id:
            user_message['contextId'] = context_id

        payload = {
            'message': user_message,
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(
            message_request=message_request
        )
        print(
            'send_response',
            send_response.model_dump_json(exclude_none=True, indent=2),
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print('received non-success response. Aborting get task ')
            # Check if error message contains task_id (task already in terminal state)
            error_message = None
            if hasattr(send_response.root, 'error'):
                error = send_response.root.error
                if hasattr(error, 'message'):
                    error_message = error.message
                elif isinstance(error, dict):
                    error_message = error.get('message', '')
            
            # If task is already in terminal state and we have task_id, try to get task from previous response
            if error_message and 'already in a terminal state' in error_message and task_id:
                print(f'Task {task_id} is already in terminal state. Attempting to get task artifacts from previous response.')
                # Try to get task from state (saved from previous successful response)
                task_key = f'task_{task_id}'
                if task_key in state:
                    saved_task = state[task_key]
                    print(f'Found saved task in state with {len(saved_task.get("artifacts", []))} artifact(s)')
                    return {'result': saved_task}
                else:
                    print(f'No saved task found in state for task_id: {task_id}')
                    # Return task_id in response so main.py can handle it
                    return {'result': {'id': task_id, 'status': {'state': 'completed'}, 'artifacts': []}}
            
            return None

        if not isinstance(send_response.root.result, Task):
            print('received non-task response. Aborting get task ')
            return None

        task_result = send_response.root.result
        
        # Save task_id to state for future messages in the same task
        if hasattr(task_result, 'id') and task_result.id:
            state['task_id'] = task_result.id
            print(f'Saved task_id to state: {task_result.id}')
        
        # Save task result in state for later retrieval when task is in terminal state
        try:
            if hasattr(task_result, 'model_dump'):
                # Pydantic model - convert to dict
                task_dict = task_result.model_dump(exclude_none=True)
                state[f'task_{task_result.id}'] = task_dict
            elif hasattr(task_result, '__dict__'):
                # Regular object - convert to dict
                task_dict = {
                    'id': getattr(task_result, 'id', None),
                    'artifacts': getattr(task_result, 'artifacts', []),
                    'status': getattr(task_result, 'status', None),
                }
                if task_dict['id']:
                    state[f'task_{task_dict["id"]}'] = task_dict
        except Exception as e:
            print(f'Error saving task to state: {e}')

        # Return Task object (or dict representation) so artifacts can be extracted
        # Convert Task to dict if needed for serialization
        try:
            if hasattr(task_result, 'model_dump'):
                # Pydantic model - convert to dict
                task_dict = task_result.model_dump(exclude_none=True)
                return {'result': task_dict}
            elif hasattr(task_result, '__dict__'):
                # Regular object - convert to dict
                task_dict = {
                    'id': getattr(task_result, 'id', None),
                    'artifacts': getattr(task_result, 'artifacts', []),
                    'status': getattr(task_result, 'status', None),
                }
                return {'result': task_dict}
            else:
                # Already a dict or other type
                return {'result': task_result}
        except Exception as e:
            print(f'Error converting task to dict: {e}')
            # Fallback: return text as before
            extracted_text = extract_text_from_task(task_result)
            if extracted_text:
                return extracted_text
            else:
                status_info = f"Task {task_result.id} status: {task_result.status.state if hasattr(task_result.status, 'state') else 'unknown'}"
                return status_info


def _get_initialized_routing_agent_sync() -> Agent:
    """Synchronously creates and initializes the RoutingAgent."""

    async def _async_main() -> Agent:
        routing_agent_instance = await RoutingAgent.create(
            remote_agent_addresses=[
                os.getenv('WEA_AGENT_URL', 'http://localhost:10002'),
                os.getenv('Content_AGENT_URL', 'http://localhost:10001'),
            ]
        )
        return routing_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if 'asyncio.run() cannot be called from a running event loop' in str(e):
            print(
                f'Warning: Could not initialize RoutingAgent with asyncio.run(): {e}. '
                'This can happen if an event loop is already running (e.g., in Jupyter). '
                'Consider initializing RoutingAgent within an async function in your application.'
            )
        raise


root_agent = _get_initialized_routing_agent_sync()