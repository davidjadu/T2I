from langchain_openai import AzureChatOpenAI
from typing import Optional, Dict, Any
import os

class OpenaiLLM:
    def __init__(
        self,
        model: str = "gpt-35-turbo",
        temperature: float = 0.7
    ):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("OPENAI_API_VERSION")

        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=model,
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        response = self.llm.invoke(prompt)
        return response.content
