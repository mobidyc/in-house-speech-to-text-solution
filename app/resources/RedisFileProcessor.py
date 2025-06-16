import json
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Awaitable, Dict, Optional, Union

import redis.asyncio as redis

from app.resources.Config import config
from app.resources.CustomLogger import CustomLogger
from app.resources.TransferFile import TransferFile, TransferFileRequest
from app.TranscriptAPI import TranscriptionConfig, WhisperXTranscriber

logger = CustomLogger("TranscriptAPI", log_level="DEBUG").get_logger()


class RedisFileProcessor:
    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: Optional[int] = None,
    ):
        print(f"config.redis_host: {config.redis_host}")
        print(f"config.redis_port: {config.redis_port}")
        print(f"config.redis_db: {config.redis_db}")
        self.redis_host = redis_host or config.redis_host
        self.redis_port = redis_port or config.redis_port
        self.redis_db = redis_db or config.redis_db
        if not self.redis_host or not self.redis_port or self.redis_db is None:
            raise ValueError("Redis configuration is incomplete. Please check your settings.")

        self.redis_client = None
        self.upload_queue = "file_upload_queue"

    async def _get_redis_client(self):
        """Get or create Redis client connection"""
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=self.redis_host, port=self.redis_port, db=self.redis_db, decode_responses=True
            )
        return self.redis_client

    async def upload_file_with_notif(
        self,
        file_object: Union[str, BytesIO],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload a file and create a lock in Redis queue
        Accepts either a file path (str) or BytesIO object
        Returns the lock key
        """

        # Create unique key with timestamp
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        path_date = datetime.now().strftime("%Y/%m/%d")
        notif_key = f"{path_date}/{os.path.basename(filename)}:{timestamp}"

        Transfer_client = TransferFile()

        # Handle both file paths and BytesIO objects
        if isinstance(file_object, str):
            # Original file path functionality
            if not os.path.exists(file_object):
                raise FileNotFoundError(f"File not found: {file_object}")

            Transfer_client.upload_file(
                TransferFileRequest(
                    filename=os.path.basename(filename),
                    source_path=file_object,
                    destination_key=notif_key,
                )
            )
        elif isinstance(file_object, BytesIO):
            # New BytesIO functionality
            Transfer_client.upload_file(
                TransferFileRequest(
                    filename=os.path.basename(filename),
                    file_object=file_object,
                    destination_key=notif_key,
                )
            )
        else:
            raise ValueError("file_object must be either a file path (str) or BytesIO object")

        # Add to queue (FIFO)
        redis_client = await self._get_redis_client()
        async with redis_client.pipeline() as pipe:
            pipe.lpush(self.upload_queue, notif_key)  # Add to left (newest)
            await pipe.execute()

        print(f"File uploaded: {notif_key}")
        return notif_key

    async def process_next_file(self) -> Optional[str]:
        """
        Get the next file to process from the queue (FIFO)
        Returns the lock key if found, None if queue is empty
        """
        redis_client = await self._get_redis_client()

        # Get from right side (oldest first - FIFO) and removes the redis object
        if not await redis_client.exists(self.upload_queue):
            print("No files in the upload queue.")
            return None

        # Pop the oldest file from the queue
        key = await redis_client.rpop(self.upload_queue)  # type: ignore

        if not key:
            return None

        ### PROCESS FILE ###
        file_path = f"/tmp/{os.path.basename(key)}"  # type: ignore
        s3_file = str(key)

        print(f"Processing file: {key} from S3: {s3_file}")

        try:
            Transfer_client = TransferFile()
            Transfer_client.download_file(
                TransferFileRequest(
                    filename=str(key),
                    source_path=s3_file,
                    destination_key=file_path,
                )
            )
        except Exception as e:
            logger.error(f"Failed to transfer file {key} from S3: {str(e)}", exc_info=True, stack_info=True)
            # Reintroducing the file in queue for retry
            async with redis_client.pipeline() as pipe:
                pipe.lpush(self.upload_queue, str(key))  # Add to left (newest)
                await pipe.execute()

        transcriber_config = TranscriptionConfig(hf_token=config.hf_token)
        transcriber = WhisperXTranscriber(config=transcriber_config)
        try:
            if not config.hf_token:
                result = transcriber.transcribe(file_path)
            else:
                result = transcriber.transcribe_with_diarization(file_path)
        except Exception as e:
            logger.error(f"Transcription failed for {file_path}: {str(e)}", exc_info=True, stack_info=True)
            # Reintroducing the file in queue for retry
            async with redis_client.pipeline() as pipe:
                pipe.lpush(self.upload_queue, str(key))
            return None

        full_text, formatted_segments = transcriber.format_transcription(
            result, include_speakers=config.hf_token is not None
        )

        result_file = f"{file_path}.txt"

        transcriber.save_results(result, result_file)

        Transfer_client.upload_file(
            TransferFileRequest(
                filename=os.path.basename(result_file),
                source_path=result_file,
                destination_key=f"{key}.txt",
            )
        )

        return str(key)

    async def get_queue_status(self) -> dict[str, Awaitable[int] | int]:
        """
        Get current queue status
        """
        redis_client = await self._get_redis_client()
        queue_length = await redis_client.llen(self.upload_queue)  # type: ignore
        print(f"Current queue length: {queue_length}")

        return {
            "pending": queue_length,
        }

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
