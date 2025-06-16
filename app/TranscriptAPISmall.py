import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import whisper
from pydantic import BaseModel, Field, field_validator

try:
    from app.resources.CustomLogger import CustomLogger
except ImportError:
    from resources.CustomLogger import CustomLogger

logger = CustomLogger("TranscriptAPI", log_level="DEBUG").get_logger()


class WhisperModelSize(str, Enum):
    """Available Whisper model sizes"""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TranscriptionConfig(BaseModel):
    """Configuration for transcription parameters"""

    model_size: WhisperModelSize = Field(default=WhisperModelSize.BASE, description="Whisper model size to use")
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en', 'fr', 'es')")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    verbose: bool = Field(default=False, description="Enable verbose output")

    @field_validator("language")
    def validate_language(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("Language code must be 2 characters (e.g., 'en', 'fr')")
        return v


class TranscriptionResult(BaseModel):
    """Result of audio transcription"""

    text: str = Field(description="Transcribed text")
    language: str = Field(description="Detected/specified language")
    segments: list = Field(default_factory=list, description="Detailed segments with timestamps")
    model_used: str = Field(description="Whisper model used for transcription")


class WhisperTranscriber(BaseModel):
    """Main transcriber class using OpenAI Whisper"""

    config: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    _model: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **data):
        super().__init__(**data)
        self._load_model()

    def _load_model(self):
        """Load the Whisper model"""
        try:
            self._model = whisper.load_model(self.config.model_size.value)
            print(f"Loaded Whisper model: {self.config.model_size.value}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    def validate_audio_file(self, file_path: str) -> Path:
        """Validate that the audio file exists and is readable"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check file extension (Whisper supports many formats)
        supported_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".mp4", ".mkv", ".avi"}
        if path.suffix.lower() not in supported_extensions:
            print(f"Warning: File extension {path.suffix} might not be supported")

        return path

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file

        Args:
            file_path: Path to the audio file

        Returns:
            TranscriptionResult with transcription details
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        # Validate file
        audio_path = self.validate_audio_file(file_path)

        print(f"Transcribing: {audio_path}")

        try:
            # Prepare transcription options
            options = {"temperature": self.config.temperature, "verbose": self.config.verbose}

            if self.config.language:
                options["language"] = self.config.language

            # Perform transcription
            try:
                result = self._model.transcribe(str(audio_path), **options)
            except Exception as e:
                raise RuntimeError(f"Failed to transcribe audio file: {e}") from e

            # Extract segments with timestamps if available
            segments = []
            if "segments" in result:
                segments = [
                    {"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "").strip()}
                    for seg in result["segments"]
                ]

            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language", self.config.language or "unknown"),
                segments=segments,
                model_used=self.config.model_size.value,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

    def transcribe_to_file(
        self,
        result: TranscriptionResult,
        audio_path: str,
        output_path: str = "transcription_output.txt",
        include_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio and save result to file

        Args:
            audio_path: Path to audio file
            output_path: Path to output text file
            include_timestamps: Whether to include timestamps in output

        Returns:
            TranscriptionResult
        """
        result = self.transcribe_file(audio_path)

        with open(output_path, "w", encoding="utf-8") as f:
            if include_timestamps and result.segments:
                f.write(f"Transcription of: {Path(audio_path).name}\n")
                f.write(f"Language: {result.language}\n")
                f.write(f"Model: {result.model_used}\n")
                f.write("-" * 50 + "\n\n")

                for segment in result.segments:
                    start_time = f"{segment['start']:.2f}s"
                    end_time = f"{segment['end']:.2f}s"
                    f.write(f"[{start_time} - {end_time}] {segment['text']}\n")
            else:
                f.write(result.text)

        print(f"Transcription saved to: {output_path}")
        return result


# Example usage
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Openai-Whisper Transcriber")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        logger.error(f"Error: Audio file not found: {args.audio_path}")
        return
    audio_file = args.audio_path

    # Configure transcription
    config = TranscriptionConfig(
        model_size=args.model,
        language="en",  # Set to None for auto-detection
        temperature=0.0,
        verbose=True,
    )

    # Create transcriber
    transcriber = WhisperTranscriber(config=config)

    try:
        # Transcribe
        result = transcriber.transcribe_file(audio_file)

        print("\nTranscription Result:")
        print(f"Language: {result.language}")
        print(f"Model: {result.model_used}")
        print(f"\nText:\n{result.text}")

        # Save to file with timestamps
        transcriber.transcribe_to_file(result, audio_file, args.output, include_timestamps=True)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
