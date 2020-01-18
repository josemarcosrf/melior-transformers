import logging

from melior_transformers.utils.common import configure_colored_logging

logger = logging.getLogger(__name__)

name = "melior_transformers"

# disable import loggins
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# configure colored loging for the library
configure_colored_logging()
