import time
from datetime import datetime

class Colors:
    """ANSI color codes for enhanced console output. This class is now complete."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Standard colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors (FIX: All bright variants are now included)
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'

class DebugLogger:
    """Enhanced debug logger with visual formatting"""
    def __init__(self):
        self.start_time = time.time()
        self.session_id = f"TAG_{int(time.time())}"
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    def header(self, title: str, char: str = "=", width: int = 80):
        border = char * width; padding = (width - len(title) - 2) // 2; centered_title = f"{' ' * padding}{title}{' ' * padding}"; print(f"\n{Colors.BRIGHT_CYAN}{border}\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{centered_title.ljust(width)}\n{Colors.BRIGHT_CYAN}{border}{Colors.RESET}\n")
    def section(self, title: str):
        print(f"\n{Colors.BRIGHT_CYAN}{'─' * 20} {Colors.BOLD}{title} {'─' * 20}{Colors.RESET}")
    def info(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_CYAN}[{self._get_timestamp()}]{Colors.RESET} {Colors.WHITE}ℹ️  {message}{Colors.RESET}")
    def success(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_GREEN}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_GREEN}✅ {message}{Colors.RESET}")
    def warning(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_YELLOW}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")
    def error(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_RED}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_RED}❌ {message}{Colors.RESET}")
    def debug(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.DIM}[{self._get_timestamp()}]{Colors.RESET} {Colors.DIM}🔍 {message}{Colors.RESET}")
    def process_start(self, process_name: str):
        print(f"{Colors.BRIGHT_MAGENTA}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_MAGENTA}🚀 Starting: {Colors.BOLD}{process_name}{Colors.RESET}")
    def process_end(self, process_name: str, duration: float, status: str = "completed"):
        status_emoji = "✅" if status == "completed" else "❌"; print(f"{Colors.BRIGHT_MAGENTA}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_MAGENTA}{status_emoji} Finished: {Colors.BOLD}{process_name}{Colors.RESET} ({duration:.2f}s)")
    def metric(self, name: str, value: str, unit: str = "", indent: int = 1):
        print(f"{'  ' * indent}{Colors.BRIGHT_CYAN}📊 {Colors.BOLD}{name}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value}{unit}{Colors.RESET}")

# Initialize global logger
logger = DebugLogger()