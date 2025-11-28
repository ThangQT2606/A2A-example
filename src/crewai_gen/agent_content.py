"""Crew AI based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

import logging
import os
import re

from collections.abc import AsyncIterable
from typing import Any

from crewai import LLM, Agent, Crew, Task
from crewai.process import Process
from crewai.tools import tool
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

from utils import initialize_client


load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)


class Contentdata(BaseModel):
    """Represents image data.

    Attributes:
      id: Unique identifier for the image.
      name: Name of the image.
      mime_type: MIME type of the image.
      bytes: Base64 encoded image data.
      error: Error message if there was an issue with the image.
    """

    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    bytes: str | None = None
    error: str | None = None

client = initialize_client(os.getenv('GITHUB_TOKEN'))

@tool('ContentGenerationTool')
def generate_content_tool(
    prompt: str
) -> str:
    """Content generation tool that generates content based on a prompt."""
    if not prompt:
        raise ValueError('Prompt cannot be empty')

    try:
        response = client.chat.completions.create(
            model='openai/gpt-4.1',
            messages=[
                {'role': 'system', 'content': 'You are a content creation expert. You specialize in taking textual descriptions and transforming them into content using a powerful content generation tool.'},
                {'role': 'user', 'content': prompt},
            ],
        )
        # Check if response has choices and at least one choice
        if not response.choices or len(response.choices) == 0:
            error_msg = 'No choices returned from API response'
            logger.error(error_msg)
            return f'Error: {error_msg}'
        
        # Check if message and content exist
        choice = response.choices[0]
        if not hasattr(choice, 'message') or not choice.message:
            error_msg = 'No message in API response choice'
            logger.error(error_msg)
            return f'Error: {error_msg}'
        
        content = choice.message.content
        if not content:
            error_msg = 'No content in API response message'
            logger.error(error_msg)
            return f'Error: {error_msg}'
        
        return content
    except Exception as e:
        error_msg = f'Error getting content data: {e}'
        logger.error(error_msg, exc_info=True)
        return f'Error: {error_msg}'

class ContentGenerationAgent:
    """Agent that generates content based on user prompts."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        if os.getenv('GOOGLE_API_KEY'):
            self.model = LLM(
                model='gemini/gemini-2.0-flash',
                api_key=os.getenv('GOOGLE_API_KEY'),
            )
        else:
            raise ValueError('GOOGLE_API_KEY is not set')

        self.content_creator_agent = Agent(
            role='Content Creation Expert',
            goal=(
                "Generate content based on the user's text prompt.If the prompt is"
                ' vague, ask clarifying questions (though the tool currently'
                " doesn't support back-and-forth within one run). Focus on"
                " interpreting the user's request and using the Content Generation"
                ' tool effectively.'
            ),
            backstory=(
                'You are a content creation expert powered by AI. You specialize in taking'
                ' textual descriptions and transforming them into content using a powerful content generation tool. You aim'
                ' for accuracy and creativity based on the prompt provided.'
            ),
            verbose=False,
            allow_delegation=False,
            tools=[generate_content_tool],
            llm=self.model,
        )

        self.content_creator_task = Task(
            description=(
                """Receive a user prompt: {query}.\nAnalyze the prompt and"
                identify if you need to create a new content or edit an existing
                one. Use the 'Content Generation' tool to for your content
                creation or modification. The tool will expect a prompt which is
                the query."""
            ),
            expected_output='The content of the generated content',
            agent=self.content_creator_agent,
        )

        self.content_creator_crew = Crew(
            agents=[self.content_creator_agent],
            tasks=[self.content_creator_task],
            process=Process.sequential,
            verbose=False,
        )

    def invoke(self, query) -> str:
        """Kickoff CrewAI and return the response."""
        response = self.content_creator_crew.kickoff({'query': query})
        return response

    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not supported by CrewAI."""
        raise NotImplementedError('Streaming is not supported by CrewAI.')
    
if __name__ == '__main__':
    agent = ContentGenerationAgent()
    response = agent.invoke('Generate a content for a blog post about the benefits of using AI.')
    print(response)