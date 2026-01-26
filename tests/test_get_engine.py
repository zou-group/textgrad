import unittest
from unittest.mock import patch, MagicMock
from textgrad.engine import get_engine

class TestGetEngine(unittest.TestCase):
    @patch('textgrad.engine.openai.ChatOpenAI')
    def test_get_engine_gpt4(self, mock_chat_openai):
        get_engine("gpt-4")
        mock_chat_openai.assert_called_once()
    
    @patch('textgrad.engine.openai.ChatOpenAI')
    def test_get_engine_gpt35(self, mock_chat_openai):
        get_engine("gpt-3.5-turbo")
        mock_chat_openai.assert_called_once()

    @patch('textgrad.engine.anthropic.ChatAnthropic')
    def test_get_engine_claude(self, mock_chat_anthropic):
        get_engine("claude-3-opus-20240229")
        mock_chat_anthropic.assert_called_once()

    @patch('textgrad.engine.gemini.ChatGemini')
    def test_get_engine_gemini(self, mock_chat_gemini):
        get_engine("gemini-pro")
        mock_chat_gemini.assert_called_once()

    @patch('textgrad.engine.cohere.ChatCohere')
    def test_get_engine_cohere(self, mock_chat_cohere):
        get_engine("command-r")
        mock_chat_cohere.assert_called_once()
    
    @patch('textgrad.engine.LiteLLMEngine')
    def test_get_engine_experimental(self, mock_litellm):
        get_engine("experimental:some-model")
        mock_litellm.assert_called_once()
        
    def test_get_engine_invalid(self):
        with self.assertRaises(ValueError):
            get_engine("invalid-engine-name")

if __name__ == '__main__':
    unittest.main()
