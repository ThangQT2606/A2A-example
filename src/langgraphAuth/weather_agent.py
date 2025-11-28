import os
from dotenv import load_dotenv
from typing import Any, Literal, Annotated
from pydantic import BaseModel, Field
from collections.abc import AsyncIterable

from langchain.tools import tool, Tool
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)

@tool
def search_web(query: str) -> str:
    """Search weather information from the web"""
    return search.results(query)

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    
class WeatherAgent:
    """WeatherAgent - a specialized assistant for weather information."""

    SYSTEM_INSTRUCTION = SystemMessage(
        content="""You are a helpful assistant that can search the web for weather information.
        "You are a helpful assistant that can search the web for weather information."
        You can use the following tools to get weather information:
        search_web: Search the web for weather information
        """
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if you need more information from the user. '
        'Set response status to error if something went wrong. '
        'Set response status to completed if the request was fully answered.'
    )

    def __init__(self, llm: BaseChatModel):
        """Initialize WeatherAgent with chosen LLM and weather search tool."""
        self.tools = [search_web]
        self.graph = create_react_agent(
            model=llm,
            tools=self.tools,
            checkpointer=InMemorySaver(),
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )
        
    def invoke(self, query: str, context_id: str) -> str:
        """Invoke the weather agent with a query and context id."""
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}
        self.graph.invoke(inputs, config)        
        return self.get_agent_response(config)

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        """Stream intermediate and final responses from the weather agent."""
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]

            if isinstance(message, AIMessage) and message.tool_calls:
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Fetching the latest weather information...',
                }

            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing weather data...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: dict[str, Any]) -> dict[str, Any]:
        """Retrieve the structured final response from the LangGraph agent."""
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')

        if structured_response and isinstance(structured_response, ResponseFormat):
            status = structured_response.status
            message = structured_response.message

            if status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': message,
                }

            if status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': f'Error occurred: {message}',
                }

            if status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'Unable to process your request at the moment. Please try again.',
        }


if __name__ == "__main__":
    import asyncio
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                             google_api_key=GOOGLE_API_KEY, 
                             temperature=0.5, 
                             top_p=1)
    agent = WeatherAgent(llm=llm)
    async def main():
        res = agent.stream(query="What is the weather in Tokyo?", context_id="123")
        async for item in res:
            print(item)
    asyncio.run(main())