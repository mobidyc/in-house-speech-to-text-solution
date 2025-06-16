#!/bin/env python

import asyncio

from app.resources.Config import config
from app.resources.CustomLogger import CustomLogger
from app.resources.RedisFileProcessor import RedisFileProcessor

logger = CustomLogger("TranscriptAPI", log_level="DEBUG").get_logger()


RedisObject = RedisFileProcessor()


async def main():
    while True:
        try:
            # Process the file upload queue
            logger.info("Processing file upload queue...")
            await RedisObject.process_next_file()
        except Exception as e:
            logger.error(f"Error processing queue: {str(e)}", exc_info=True, stack_info=True)

        # Sleep for a while before checking the queue again
        await asyncio.sleep(5)  # Adjust the sleep time as needed


if __name__ == "__main__":
    asyncio.run(main())
