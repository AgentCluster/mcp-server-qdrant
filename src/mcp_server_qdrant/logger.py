import logging
import os
import sys
from typing import Optional
from mcp.server.fastmcp import Context

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class ContextLogger:
    def __init__(self, name: str, ctx: Optional[Context] = None):
        self.logger = logging.getLogger(name)
        self.ctx = ctx
        
        if not self.logger.handlers:
            # Console handler - priority for Docker
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
            
            # File handler
            log_file = os.path.join(LOG_DIR, f"{name.replace('.', '_')}.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            
            # Set logger level to DEBUG for more verbose output
            self.logger.setLevel(logging.INFO)
            
            # Prevent propagation to avoid duplicate console messages
            self.logger.propagate = False

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    # Async versions for compatibility
    async def adebug(self, message: str):
        if self.ctx:
            await self.ctx.debug(message)
        self.debug(message)

    async def ainfo(self, message: str):
        if self.ctx:
            await self.ctx.debug(message)
        self.info(message)

    async def awarning(self, message: str):
        if self.ctx:
            await self.ctx.debug(message)
        self.warning(message)

    async def aerror(self, message: str):
        if self.ctx:
            await self.ctx.debug(message)
        self.error(message)

def get_logger(name: str, ctx: Optional[Context] = None) -> ContextLogger:
    return ContextLogger(name, ctx)