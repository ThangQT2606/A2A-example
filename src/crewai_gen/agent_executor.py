import re

from typing_extensions import override
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    InvalidParamsError,
    Part,
    Task,
    TextPart,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_task,
    new_agent_text_message,
    completed_task
)
from a2a.utils.errors import ServerError
from agent_content import ContentGenerationAgent, Contentdata


class ContentGenerationAgentExecutor(AgentExecutor):
    """Content Generation AgentExecutor Example."""

    def __init__(self) -> None:
        self.agent = ContentGenerationAgent()

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            result = self.agent.invoke(query)
            print(f'Final Result ===> {result}')
            # CrewOutput is an object, not a dict. Use .raw attribute or convert to string
            result_text = result.raw if hasattr(result, 'raw') else str(result)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(result_text, task.context_id, task.id),
            )
        except Exception as e:
            print('Error invoking agent: %s', e)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f'Error invoking agent: {e}', task.context_id, task.id),
            )
        await updater.complete()

    @override
    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        return False