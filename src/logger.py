import logging

# Set up logging configuration
logging.basicConfig(
    filename='log.log',  # Log file
    filemode='a',                # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of logs
    level=logging.INFO           # Log level
)

logger = logging.getLogger(__name__)  # Get logger object
