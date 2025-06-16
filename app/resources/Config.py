from typing import Optional

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Hugginface token
    hf_token: Optional[str] = None

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ssl: bool = False

    # S3 configuration
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket_name: str = "audio_files"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = Config()
