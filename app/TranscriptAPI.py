#!/usr/bin/env python
"""
WhisperX Audio Transcriber Class with Language Detection and Diarization

This module provides a WhisperXTranscriber class that can:
1. Detect the language of an audio file
2. Transcribe the entire audio file
3. Perform speaker diarization
4. Align transcription with speaker segments

Requirements:
- pip install whisperx
- torch (PyTorch)
- ffmpeg (for audio processing)
- pyannote.audio (for diarization)

Usage:
    from whisperx_transcriber import WhisperXTranscriber

    transcriber = WhisperXTranscriber()
    result = transcriber.transcribe_with_diarization("audio.wav")
"""

import json
import os
import sys
from enum import Enum
from pathlib import Path
from pprint import pprint as pp
from typing import Any, Dict, Optional, Tuple

import torch
import whisperx
from pydantic import BaseModel, Field

try:
    from app.resources.CustomLogger import CustomLogger
except ImportError:
    from resources.CustomLogger import CustomLogger

logger = CustomLogger("TranscriptAPI", log_level="DEBUG").get_logger()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class WhisperXModelSize(str, Enum):
    """Available Whisper model sizes"""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGEV2 = "large-v2"
    LARGEV3 = "large-v3"


class TranscriptionConfig(BaseModel):
    """Configuration for transcription parameters"""

    model_size: WhisperXModelSize = Field(default=WhisperXModelSize.BASE, description="Whisper model size to use")
    device: Optional[str] = Field(default=None, description="Device to use ('cpu', 'cuda', 'mps')")
    compute_type: str = Field(default="float16", description="Compute type for inference")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token for diarization model access")


class TranscriptionResult(BaseModel):
    """Result of audio transcription"""

    language: str = Field(description="Detected/specified language")
    language_probability: float = Field(default=0.0, description="Confidence score for detected language")
    segments: list = Field(default_factory=list, description="Detailed segments with timestamps")
    diarization: Optional[Any] = Field(default=None, description="Speaker diarization results")
    audio_path: str = Field(description="Path to the original audio file")


class WhisperXTranscriber(BaseModel):
    """
    A class for audio transcription with language detection and speaker diarization using WhisperX
    """

    config: TranscriptionConfig = Field(default_factory=TranscriptionConfig)

    # Non-pydantic fields (initialized post-init)
    whisper_model: Any = Field(default=None, exclude=True)
    align_model: Any = Field(default=None, exclude=True)
    align_metadata: Any = Field(default=None, exclude=True)
    diarize_model: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)

        # Auto-detect device if not specified
        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.config.device = "mps"
            else:
                self.config.device = "cpu"

        if self.config.device == "cpu":
            self.config.compute_type = "float32"  # Use float32 for CPU

        # Initialize models as None - will be loaded on demand
        object.__setattr__(self, "whisper_model", None)
        object.__setattr__(self, "align_model", None)
        object.__setattr__(self, "align_metadata", None)
        object.__setattr__(self, "diarize_model", None)

        logger.info("WhisperX Transcriber initialized:")
        logger.info(f"  Model: {self.config.model_size.value}")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Compute type: {self.config.compute_type}")

    def _load_whisper_model(self) -> None:
        """Load the Whisper model if not already loaded"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.config.model_size.value}")
            self.whisper_model = whisperx.load_model(
                self.config.model_size.value, self.config.device, compute_type=self.config.compute_type
            )

    def _load_align_model(self, language_code: str) -> bool:
        """Load the alignment model for a specific language"""
        try:
            logger.info(f"Loading alignment model for language: {language_code}")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code, device=self.config.device
            )
            return True
        except Exception as e:
            logger.info(f"Warning: Could not load alignment model for {language_code}: {e}")
            return False

    def _load_diarization_model(self):
        """Load the diarization model"""
        if self.diarize_model is None:
            if self.config.hf_token is None:
                raise ValueError(
                    "HuggingFace token is required for diarization. Get one at https://huggingface.co/settings/tokens"
                )

            logger.info("Loading diarization model...")

            self.diarize_model = whisperx.diarize.DiarizationPipeline(  # type: ignore
                use_auth_token=self.config.hf_token, device=self.config.device
            )  # type: ignore
            return

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect the language of an audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            Tuple[str, float]: (language_code, confidence_score)
        """
        self._load_whisper_model()

        logger.info(f"Loading audio: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        logger.info("Detecting language...")
        # Ensure whisper_model is loaded
        if self.whisper_model is None:
            raise RuntimeError("Whisper model failed to load.")

        # Transcribe a small portion for language detection
        result = self.whisper_model.transcribe(audio[: 30 * 16000], batch_size=16)  # First 30 seconds

        language = result.get("language", "unknown")
        confidence = result.get("language_probability", 0.0)

        logger.info(f"Detected language: {language} (confidence: {confidence:.2%})")
        return language, confidence

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file with language detection

        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (if known). Will auto-detect if None

        Returns:
            Dict: Transcription results with language information
        """
        self._load_whisper_model()

        if self.whisper_model is None:
            raise RuntimeError("Whisper model failed to load.")

        logger.debug(f"Loading audio: {audio_path}")
        try:
            audio = whisperx.load_audio(audio_path)
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}", exc_info=True, stack_info=True)
            raise RuntimeError(f"Could not load audio file: {audio_path}") from e

        logger.info("Transcribing audio...")
        try:
            if language:
                result = self.whisper_model.transcribe(audio, batch_size=16, language=language)
            else:
                result = self.whisper_model.transcribe(audio, batch_size=16)
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}", exc_info=True, stack_info=True)
            raise RuntimeError(f"Transcription failed for {audio_path}") from e

        detected_language = result.get("language", "unknown")
        language_probability = result.get("language_probability", 0.0)

        logger.debug(f"Language: {detected_language} (confidence: {language_probability:.2%})")

        # Load alignment model and align timestamps
        if self._load_align_model(detected_language):
            logger.info("Aligning timestamps...")
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False,
            )
        else:
            logger.warning(f"Alignment model for {detected_language} not loaded. Skipping alignment.")

        return TranscriptionResult(
            language=detected_language,
            language_probability=language_probability,
            segments=result.get("segments", []),
            audio_path=audio_path,
        )

    def diarize(
        self, audio_path: str, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path (str): Path to the audio file
            min_speakers (int): Minimum number of speakers (optional)
            max_speakers (int): Maximum number of speakers (optional)

        Returns:
            Dict: Diarization results
        """
        self._load_diarization_model()

        if self.diarize_model is None:
            raise RuntimeError("Diarization model failed to load.")

        logger.info(f"Performing diarization on: {audio_path}")
        try:
            diarize_segments = self.diarize_model(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True, stack_info=True)
            raise RuntimeError(f"Diarization failed for {audio_path}") from e

        return diarize_segments

    def transcribe_with_diarization(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """
        Perform complete transcription with speaker diarization

        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (if known). Will auto-detect if None
            min_speakers (int): Minimum number of speakers
            max_speakers (int): Maximum number of speakers

        Returns:
            Dict: Complete results with transcription and diarization
        """
        logger.info("Starting transcription with diarization...")

        # Step 1: Transcribe and Align
        transcription_result = self.transcribe(audio_path, language)

        # Step 2: Diarize
        diarization_result = self.diarize(audio_path, min_speakers, max_speakers)

        # Step 3: Assign speakers to transcription segments
        logger.info("Assigning speakers to segments...")
        final_result = whisperx.assign_word_speakers(diarization_result, transcription_result.model_dump())

        return TranscriptionResult(
            language=transcription_result.language,
            language_probability=transcription_result.language_probability,
            segments=final_result["segments"],
            diarization=diarization_result,
            audio_path=audio_path,
        )

    def format_transcription(self, result: TranscriptionResult, include_speakers: bool = True) -> Tuple[str, str]:
        """
        Format transcription results for display

        Args:
            result (TranscriptionResult): Transcription result
            include_speakers (bool): Whether to include speaker information

        Returns:
            Tuple[str, str]: (full_text, formatted_segments)
        """
        if not result or not result.segments:
            return "No transcription available", ""

        segments = result.segments
        full_text = ""
        formatted_segments = ""

        current_speaker = None

        for segment in segments:
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            speaker = segment.get("speaker", "Unknown") if include_speakers else None

            full_text += text + " "

            if include_speakers and speaker:
                if speaker != current_speaker:
                    formatted_segments += f"\n[{speaker}]\n"
                    current_speaker = speaker
                formatted_segments += f"[{start:.2f}s - {end:.2f}s] {text}\n"
            else:
                formatted_segments += f"[{start:.2f}s - {end:.2f}s] {text}\n"

        return full_text.strip(), formatted_segments

    def save_results(self, result: TranscriptionResult, output_path: Optional[str] = None, format: str = "txt"):
        """
        Save transcription results to file

        Args:
            result TranscriptionResult: Transcription result
            output_path (str): Output file path (optional)
            format (str): Output format ("txt", "json", "srt")
        """
        if not result:
            logger.info("No results to save")
            return

        if output_path is None:
            audio_path = Path(result.audio_path)
            output_path = str(audio_path.with_suffix(f".{format}"))

        if format == "txt":
            self._save_txt(result, output_path)
        elif format == "json":
            self._save_json(result, output_path)
        elif format == "srt":
            self._save_srt(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'txt', 'json', or 'srt'")

        logger.info(f"Results saved to: {output_path}")

    def _save_txt(self, result: TranscriptionResult, output_path: str):
        """Save results as formatted text"""
        full_text, formatted_segments = self.format_transcription(result)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Audio File: {result.audio_path}\n")
            f.write(f"Detected Language: {result.language}\n")
            f.write(f"Language Confidence: {result.language_probability:.2%}\n")
            f.write("=" * 60 + "\n\n")
            f.write("FULL TRANSCRIPTION:\n")
            f.write(full_text + "\n\n")
            f.write("DETAILED SEGMENTS:\n")
            f.write(formatted_segments)

    def _save_json(self, result: TranscriptionResult, output_path: str):
        """Save results as JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

    def _save_srt(self, result: TranscriptionResult, output_path: str):
        """Save results as SRT subtitle file"""
        segments = result.segments

        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker")

                # Format timestamps for SRT
                start_time = self._seconds_to_srt_time(start)
                end_time = self._seconds_to_srt_time(end)

                # Add speaker if available
                if speaker:
                    text = f"[{speaker}] {text}"

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WhisperX Transcriber with Diarization")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--hf-token", help="HuggingFace token for diarization")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", choices=["txt", "json", "srt"], default="txt", help="Output format")
    parser.add_argument("--no-diarization", action="store_true", help="Skip diarization")

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        logger.error(f"Error: Audio file not found: {args.audio_path}")
        return

    config = TranscriptionConfig(
        model_size=args.model,
        device=args.device,
        hf_token=args.hf_token,
    )

    # Initialize transcriber
    transcriber = WhisperXTranscriber(config=config)

    try:
        if args.no_diarization or args.hf_token is None:
            if args.hf_token is None:
                logger.info("Note: Diarization disabled (no HuggingFace token provided)")
            result = transcriber.transcribe(args.audio_path)
        else:
            result = transcriber.transcribe_with_diarization(
                args.audio_path, min_speakers=args.min_speakers, max_speakers=args.max_speakers
            )

        # Display results
        logger.debug("\n" + "=" * 60)
        logger.debug("TRANSCRIPTION RESULTS")
        logger.debug("=" * 60)

        full_text, formatted_segments = transcriber.format_transcription(
            result, include_speakers=not args.no_diarization and args.hf_token is not None
        )

        logger.debug("\nFull Transcription:")
        logger.debug("-" * 40)
        logger.debug(full_text)

        logger.debug("\nDetailed Segments:")
        logger.debug("-" * 40)
        logger.debug(formatted_segments)

        # Save results
        transcriber.save_results(result, args.output, args.format)

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
