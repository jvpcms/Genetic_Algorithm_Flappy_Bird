import os
import dotenv

class EnvConfig:
  env_file: str

  def __init__(self, env_file: str):
    self.env_file = env_file
    self.load_envs()

  def load_envs(self):
    dotenv.load_dotenv(self.env_file)

  @property
  def logging_level(self) -> int:
    self.load_envs()
    return int(os.getenv("LOGGING_LEVEL"))


def get_env_config() -> EnvConfig:
  return EnvConfig(".env")