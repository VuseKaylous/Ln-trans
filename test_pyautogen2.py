from autogen import AssistantAgent, UserProxyAgent

config_list = [
  {
    "model": "llama3",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "price": ["prompt_price_per_1k", "completion_token_price_per_1k"]
  }
]

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "translation", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Translate the following sentence from English to Vietnamese: The moon is beautiful today.")
