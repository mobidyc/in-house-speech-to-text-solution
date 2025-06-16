import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from app.TranscriptAPI import TranscriptionConfig, TranscriptionResult, WhisperXModelSize, WhisperXTranscriber

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TestWhisperXTranscriber:
    @pytest.fixture
    def mock_transcriber(self):
        """Create a transcriber with mocked models"""
        config = TranscriptionConfig(model_size=WhisperXModelSize.BASE, device="cpu", hf_token="mock_token")

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            transcriber = WhisperXTranscriber(config=config)

        return transcriber

    @pytest.fixture
    def sample_transcription_result(self):
        """Sample transcription result for testing"""
        return TranscriptionResult(
            language="en",
            language_probability=0.95,
            segments=[
                {"start": 0.0, "end": 2.5, "text": "Hello world", "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 5.0, "text": "How are you today", "speaker": "SPEAKER_01"},
            ],
            audio_path="test_audio.wav",
        )

    def test_transcription_config_defaults(self):
        """Test default configuration values"""
        config = TranscriptionConfig()
        assert config.model_size == WhisperXModelSize.BASE
        assert config.device is None
        assert config.compute_type == "float16"
        assert config.hf_token is None

    def test_whisperx_transcriber_initialization(self):
        """Test transcriber initialization with device detection"""
        with patch("torch.cuda.is_available", return_value=True):
            transcriber = WhisperXTranscriber()
            assert transcriber.config.device == "cuda"
            assert transcriber.config.compute_type == "float16"

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            transcriber = WhisperXTranscriber()
            assert transcriber.config.device == "mps"

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            transcriber = WhisperXTranscriber()
            assert transcriber.config.device == "cpu"
            assert transcriber.config.compute_type == "float32"

    @patch("whisperx.load_model")
    def test_load_whisper_model(self, mock_load_model, mock_transcriber):
        """Test loading Whisper model"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        mock_transcriber._load_whisper_model()

        mock_load_model.assert_called_once_with("base", "cpu", compute_type="float32")
        assert mock_transcriber.whisper_model == mock_model

    @patch("whisperx.load_align_model")
    def test_load_align_model_success(self, mock_load_align, mock_transcriber):
        """Test successful alignment model loading"""
        mock_model = Mock()
        mock_metadata = Mock()
        mock_load_align.return_value = (mock_model, mock_metadata)

        result = mock_transcriber._load_align_model("en")

        assert result is True
        mock_load_align.assert_called_once_with(language_code="en", device="cpu")
        assert mock_transcriber.align_model == mock_model
        assert mock_transcriber.align_metadata == mock_metadata

    @patch("whisperx.load_align_model")
    def test_load_align_model_failure(self, mock_load_align, mock_transcriber):
        """Test alignment model loading failure"""
        mock_load_align.side_effect = Exception("Model not found")

        result = mock_transcriber._load_align_model("unknown")

        assert result is False

    @patch("whisperx.diarize.DiarizationPipeline")
    def test_load_diarization_model(self, mock_diarize_pipeline, mock_transcriber):
        """Test loading diarization model"""
        mock_pipeline = Mock()
        mock_diarize_pipeline.return_value = mock_pipeline

        mock_transcriber._load_diarization_model()

        mock_diarize_pipeline.assert_called_once_with(use_auth_token="mock_token", device="cpu")
        assert mock_transcriber.diarize_model == mock_pipeline

    def test_load_diarization_model_no_token(self, mock_transcriber):
        """Test diarization model loading without token"""
        mock_transcriber.config.hf_token = None

        with pytest.raises(ValueError, match="HuggingFace token is required"):
            mock_transcriber._load_diarization_model()

    @patch("whisperx.load_audio")
    @patch("whisperx.load_model")
    def test_detect_language(self, mock_load_model, mock_load_audio, mock_transcriber):
        """Test language detection"""
        mock_audio = [0.1] * 16000 * 30  # 30 seconds of mock audio
        mock_load_audio.return_value = mock_audio

        mock_model = Mock()
        mock_model.transcribe.return_value = {"language": "en", "language_probability": 0.95}
        mock_load_model.return_value = mock_model

        language, confidence = mock_transcriber.detect_language("test.wav")

        assert language == "en"
        assert confidence == 0.95
        mock_load_audio.assert_called_once_with("test.wav")

    @patch("whisperx.load_audio")
    @patch("whisperx.load_model")
    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_transcribe(self, mock_align, mock_load_align_model, mock_load_model, mock_load_audio, mock_transcriber):
        """Test audio transcription"""
        mock_audio = [0.1] * 16000 * 10
        mock_load_audio.return_value = mock_audio

        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "language_probability": 0.95,
            "segments": [{"start": 0, "end": 2, "text": "test"}],
        }
        mock_load_model.return_value = mock_model

        mock_align_model = Mock()
        mock_align_metadata = Mock()
        mock_load_align_model.return_value = (mock_align_model, mock_align_metadata)

        mock_align.return_value = {"segments": [{"start": 0, "end": 2, "text": "test aligned"}]}

        result = mock_transcriber.transcribe("test.wav")

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "test aligned"

    def test_format_transcription(self, mock_transcriber, sample_transcription_result):
        """Test transcription formatting"""
        full_text, formatted_segments = mock_transcriber.format_transcription(
            sample_transcription_result, include_speakers=True
        )

        assert "Hello world" in full_text
        assert "How are you today" in full_text
        assert "[SPEAKER_00]" in formatted_segments
        assert "[SPEAKER_01]" in formatted_segments
        assert "[0.00s - 2.50s]" in formatted_segments

    def test_format_transcription_no_speakers(self, mock_transcriber, sample_transcription_result):
        """Test transcription formatting without speakers"""
        full_text, formatted_segments = mock_transcriber.format_transcription(
            sample_transcription_result, include_speakers=False
        )

        assert "Hello world" in full_text
        assert "SPEAKER_00" not in formatted_segments
        assert "[0.00s - 2.50s]" in formatted_segments

    def test_save_results_txt(self, mock_transcriber, sample_transcription_result):
        """Test saving results as TXT"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            mock_transcriber.save_results(sample_transcription_result, output_path, "txt")

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "Audio File: test_audio.wav" in content
                assert "Detected Language: en" in content
                assert "Hello world" in content
        finally:
            os.unlink(output_path)

    def test_save_results_json(self, mock_transcriber, sample_transcription_result):
        """Test saving results as JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            mock_transcriber.save_results(sample_transcription_result, output_path, "json")

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["language"] == "en"
                assert data["audio_path"] == "test_audio.wav"
        finally:
            os.unlink(output_path)

    def test_save_results_srt(self, mock_transcriber, sample_transcription_result):
        """Test saving results as SRT"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            output_path = f.name

        try:
            mock_transcriber.save_results(sample_transcription_result, output_path, "srt")

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "00:00:00,000 --> 00:00:02,500" in content
                assert "[SPEAKER_00] Hello world" in content
        finally:
            os.unlink(output_path)

    def test_seconds_to_srt_time(self, mock_transcriber):
        """Test SRT time conversion"""
        assert mock_transcriber._seconds_to_srt_time(0) == "00:00:00,000"
        assert mock_transcriber._seconds_to_srt_time(65.5) == "00:01:05,500"
        assert mock_transcriber._seconds_to_srt_time(3661.123) == "01:01:01,123"

    def test_unsupported_save_format(self, mock_transcriber, sample_transcription_result):
        """Test error handling for unsupported save format"""
        with pytest.raises(ValueError, match="Unsupported format"):
            mock_transcriber.save_results(sample_transcription_result, "test.xyz", "xyz")


@pytest.mark.functional
class TestWhisperXTranscriberFunctional:
    """Functional tests using actual audio file"""

    @pytest.fixture
    def harvard_audio_path(self):
        """Path to the harvard.wav test file"""
        test_audio_path = Path("tests/harvard.wav")
        if not test_audio_path.exists():
            pytest.skip("harvard.wav test file not found in tests/ directory")
        return str(test_audio_path)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_transcription_with_harvard_audio(self, harvard_audio_path):
        """Functional test with actual audio file (requires CUDA)"""
        config = TranscriptionConfig(
            model_size=WhisperXModelSize.TINY,  # Use smallest model for speed
            device="cuda",
        )

        transcriber = WhisperXTranscriber(config=config)

        # Test basic transcription
        result = transcriber.transcribe(harvard_audio_path)

        assert isinstance(result, TranscriptionResult)
        assert result.language is not None
        assert len(result.segments) > 0
        assert result.audio_path == harvard_audio_path

        # Check that some common words from Harvard sentences appear
        full_text, _ = transcriber.format_transcription(result)
        assert len(full_text) > 0
        # Test that the transcription contains expected text from Harvard sentences
        assert "The stale smell of old beer lingers" in full_text

    def test_cpu_transcription_with_harvard_audio(self, harvard_audio_path):
        """Functional test with CPU (slower but more compatible)"""
        config = TranscriptionConfig(model_size=WhisperXModelSize.TINY, device="cpu")

        transcriber = WhisperXTranscriber(config=config)

        # Test language detection
        language, confidence = transcriber.detect_language(harvard_audio_path)
        assert language is not None
        assert 0 <= confidence <= 1

        # Test transcription
        result = transcriber.transcribe(harvard_audio_path)
        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) > 0
        assert len(result.segments) > 0
        assert len(result.segments) > 0
        assert len(result.segments) > 0
        assert len(result.segments) > 0
        assert len(result.segments) > 0
