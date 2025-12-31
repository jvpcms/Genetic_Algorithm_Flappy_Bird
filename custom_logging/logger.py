from typing import Optional
from enum import Enum
import datetime

class LogLevel(Enum):
  DEBUG = 5
  INFO = 4
  WARNING = 3
  ERROR = 2
  CRITICAL = 1

  def __ge__(self, other: 'LogLevel') -> bool:
    return self.value >= other.value

  @staticmethod
  def from_int(log_level: int) -> 'LogLevel':
    if log_level == 5:
      return LogLevel.DEBUG
    elif log_level == 4:
      return LogLevel.INFO
    elif log_level == 3:
      return LogLevel.WARNING
    elif log_level == 2:
      return LogLevel.ERROR
    elif log_level == 1:
      return LogLevel.CRITICAL

class LogColor(Enum):
  DEBUG = "\033[94m"
  INFO = "\033[92m"
  WARNING = "\033[93m"
  ERROR = "\033[91m"
  CRITICAL = "\033[95m"
  RESET = "\033[0m"

class Logger:
  log_file: Optional[str]
  log_level: LogLevel
  log_name: str

  def __init__(self, log_level: int = LogLevel.INFO, log_name: str = "logger", log_file: Optional[str] = None):
    """
    Initialize the logger.
    """

    self.log_level = LogLevel.from_int(log_level)
    self.log_file = log_file
    self.log_name = log_name

  def log(self, message: str, log_color: LogColor = LogColor.INFO):
    """
    Log a message.
    """
    if self.log_file is not None:
      with open(self.log_file, 'a') as f:
        f.write(f"{self.log_name} -- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- {message}\n")
    else:
      print(
        f"{log_color.value}{log_color.name}: {self.log_name} -- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- {LogColor.RESET.value} ", 
        end=""
      )
      print(message)

  def debug(self, message: str):
    """
    Log a debug message.
    """
    if self.log_level >= LogLevel.DEBUG:
      self.log(f"{message}", LogColor.DEBUG)

  def info(self, message: str):
    """
    Log an info message.
    """
    if self.log_level >= LogLevel.INFO:
      self.log(f"{message}", LogColor.INFO)

  def warning(self, message: str):
    """
    Log a warning message.
    """
    if self.log_level >= LogLevel.WARNING:
      self.log(f"{message}", LogColor.WARNING)

  def error(self, message: str):
    """
    Log an error message.
    """
    if self.log_level >= LogLevel.ERROR:
      self.log(f"{message}", LogColor.ERROR)

  def critical(self, message: str):
    """
    Log a critical message.
    """
    if self.log_level >= LogLevel.CRITICAL:
      self.log(f"{message}", LogColor.CRITICAL)


def get_logger(log_level: LogLevel = LogLevel.INFO, log_name: str = "logger", log_file: Optional[str] = None) -> Logger:
  """
  Get a logger.
  """
  return Logger(log_level, log_name, log_file)