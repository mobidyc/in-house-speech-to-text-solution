import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import whisper

from app.TranscriptAPISmall import TranscriptionConfig, TranscriptionResult, WhisperModelSize, WhisperTranscriber


class TestTranscriptionConfig:
    def test_default_values(self):
        """Test default configuration values"""
        config = TranscriptionConfig()
        assert config.model_size == WhisperModelSize.BASE
        assert config.language is None
        assert config.temperature == 0.0
        assert config.verbose is False

    def test_custom_values(self):
        """Test custom configuration values"""
        config = TranscriptionConfig(model_size=WhisperModelSize.LARGE, language="fr", temperature=0.5, verbose=True)
        assert config.model_size == WhisperModelSize.LARGE
        assert config.language == "fr"
        assert config.temperature == 0.5
        assert config.verbose is True

    def test_language_validation_valid(self):
        """Test valid language codes"""
        config = TranscriptionConfig(language="en")
        assert config.language == "en"

        config = TranscriptionConfig(language="fr")
        assert config.language == "fr"

    def test_language_validation_invalid(self):
        """Test invalid language codes"""
        with pytest.raises(ValueError, match="Language code must be 2 characters"):
            TranscriptionConfig(language="eng")

        with pytest.raises(ValueError, match="Language code must be 2 characters"):
            TranscriptionConfig(language="f")

    def test_temperature_validation(self):
        """Test temperature validation"""
        # Valid temperatures
        config = TranscriptionConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = TranscriptionConfig(temperature=1.0)
        assert config.temperature == 1.0

        config = TranscriptionConfig(temperature=0.5)
        assert config.temperature == 0.5


class TestTranscriptionResult:
    def test_transcription_result_creation(self):
        """Test creating a transcription result"""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            segments=[{"start": 0.0, "end": 2.0, "text": "Hello world"}],
            model_used="base",
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.model_used == "base"

    def test_transcription_result_defaults(self):
        """Test default values for optional fields"""
        result = TranscriptionResult(text="Hello", language="en", model_used="base")

        assert result.segments == []


class TestWhisperTranscriber:
    @pytest.fixture
    def mock_transcriber(self):
        """Create a transcriber with mocked Whisper model"""
        config = TranscriptionConfig(model_size=WhisperModelSize.BASE)

        with patch("whisper.load_model") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            transcriber = WhisperTranscriber(config=config)
            transcriber._model = mock_model

        return transcriber

    @patch("whisper.load_model")
    def test_initialization_loads_model(self, mock_load_model):
        """Test that model is loaded during initialization"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        config = TranscriptionConfig(model_size=WhisperModelSize.TINY)
        transcriber = WhisperTranscriber(config=config)

        mock_load_model.assert_called_once_with("tiny")
        assert transcriber._model == mock_model

    @patch("whisper.load_model")
    def test_initialization_model_load_failure(self, mock_load_model):
        """Test handling of model loading failure"""
        mock_load_model.side_effect = Exception("Model loading failed")

        with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
            WhisperTranscriber()

    def test_validate_audio_file_success(self, mock_transcriber, tmp_path):
        """Test successful audio file validation"""
        # Create a temporary audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("dummy audio content")

        result = mock_transcriber.validate_audio_file(str(audio_file))
        assert result == audio_file

    def test_validate_audio_file_not_found(self, mock_transcriber):
        """Test validation with non-existent file"""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            mock_transcriber.validate_audio_file("nonexistent.wav")

    def test_validate_audio_file_not_file(self, mock_transcriber, tmp_path):
        """Test validation with directory instead of file"""
        directory = tmp_path / "test_dir"
        directory.mkdir()

        with pytest.raises(ValueError, match="Path is not a file"):
            mock_transcriber.validate_audio_file(str(directory))

    def test_validate_audio_file_supported_extensions(self, mock_transcriber, tmp_path):
        """Test validation with supported file extensions"""
        supported_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".mp4"]

        for ext in supported_extensions:
            audio_file = tmp_path / f"test{ext}"
            audio_file.write_text("dummy content")

            result = mock_transcriber.validate_audio_file(str(audio_file))
            assert result == audio_file

    def test_transcribe_file_success(self, mock_transcriber, tmp_path):
        """Test successful transcription"""
        # Create test audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("dummy audio")

        # Mock transcription result
        mock_result = {
            "text": "Hello world",
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello world"}],
        }
        mock_transcriber._model.transcribe.return_value = mock_result

        result = mock_transcriber.transcribe_file(str(audio_file))

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.model_used == "base"

    def test_transcribe_file_no_model(self, tmp_path):
        """Test transcription without loaded model"""
        config = TranscriptionConfig()
        with patch("whisper.load_model") as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            try:
                transcriber = WhisperTranscriber(config=config)
            except RuntimeError:
                # Create a properly initialized transcriber with None model
                with patch("whisper.load_model") as mock_load_success:
                    mock_load_success.return_value = Mock()
                    transcriber = WhisperTranscriber(config=config)
                    transcriber._model = None

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("dummy audio")

        with pytest.raises(RuntimeError, match="Whisper model not loaded"):
            transcriber.transcribe_file(str(audio_file))

    def test_transcribe_file_with_language_config(self, mock_transcriber, tmp_path):
        """Test transcription with specific language configured"""
        mock_transcriber.config.language = "fr"

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("dummy audio")

        mock_result = {"text": "Bonjour monde", "language": "fr"}
        mock_transcriber._model.transcribe.return_value = mock_result

        result = mock_transcriber.transcribe_file(str(audio_file))

        # Verify that language parameter was passed to transcribe call
        call_args = mock_transcriber._model.transcribe.call_args
        assert call_args[1]["language"] == "fr"
        assert result.language == "fr"

    def test_transcribe_file_transcription_failure(self, mock_transcriber, tmp_path):
        """Test handling of transcription failure"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("dummy audio")

        mock_transcriber._model.transcribe.side_effect = Exception("Transcription failed")

        with pytest.raises(RuntimeError, match="Transcription failed"):
            mock_transcriber.transcribe_file(str(audio_file))

    # def test_transcribe_to_file_without_timestamps(self, mock_transcriber, tmp_path):
    #     """Test saving transcription to file without timestamps"""
    #     # Create test audio file
    #     audio_file = tmp_path / "test.wav"
    #     audio_file.write_text("dummy audio")

    #     # Create mock result
    #     result = TranscriptionResult(text="Hello world", language="en", model_used="base")

    #     # Mock the transcribe_file method
    #     mock_transcriber.transcribe_file = Mock(return_value=result)

    #     output_file = tmp_path / "output.txt"
    #     returned_result = mock_transcriber.transcribe_to_file(
    #         result, str(audio_file), str(output_file), include_timestamps=False
    #     )

    #     # Check that file was written
    #     assert output_file.exists()
    #     content = output_file.read_text(encoding="utf-8")
    #     assert content == "Hello world"
    #     assert returned_result == result

    # def test_transcribe_to_file_with_timestamps(self, mock_transcriber: WhisperTranscriber, tmp_path):
    #     """Test saving transcription to file with timestamps"""
    #     # Create test audio file
    #     audio_file = tmp_path / "test.wav"
    #     audio_file.write_text("dummy audio")

    #     # Create mock result with segments
    #     result = TranscriptionResult(
    #         text="Hello world",
    #         language="en",
    #         model_used="base",
    #         segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}, {"start": 1.0, "end": 2.0, "text": "world"}],
    #     )

    #     # Mock the transcribe_file method to return our result
    #     mock_transcriber.transcribe_file = Mock(return_value=result)

    #     output_file = tmp_path / "output.txt"
    #     mock_transcriber.transcribe_to_file(result, str(audio_file), str(output_file), include_timestamps=True)

    #     # Check file content
    #     assert output_file.exists()
    #     content = output_file.read_text(encoding="utf-8")

    #     assert "test.wav" in content
    #     assert "Language: en" in content
    #     assert "Model: base" in content
    #     assert "[0.00s - 1.00s] Hello" in content
    #     assert "[1.00s - 2.00s] world" in content


@pytest.mark.functional
class TestWhisperTranscriberFunctional:
    """Functional tests using actual audio file"""

    @pytest.fixture
    def harvard_audio_path(self):
        """Path to the harvard.wav test file"""
        test_audio_path = Path("tests/harvard.wav")
        if not test_audio_path.exists():
            pytest.skip("harvard.wav test file not found in tests/ directory")
        return str(test_audio_path)

    def test_transcribe_harvard_audio(self, harvard_audio_path):
        """Functional test with actual harvard.wav audio file"""
        config = TranscriptionConfig(
            model_size=WhisperModelSize.TINY,  # Use smallest model for speed
            language="en",
            verbose=False,
        )

        transcriber = WhisperTranscriber(config=config)
        result = transcriber.transcribe_file(harvard_audio_path)

        # Verify result structure
        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert len(result.text) > 0
        assert result.model_used == "tiny"

        # Check that transcription contains expected content from Harvard sentences
        text_lower = result.text.lower()
        assert "the stale smell of old beer lingers" in text_lower

    def test_transcribe_to_file_harvard_audio(self, harvard_audio_path, tmp_path):
        """Test transcribing harvard.wav and saving to file"""
        config = TranscriptionConfig(model_size=WhisperModelSize.TINY, language="en")

        transcriber = WhisperTranscriber(config=config)

        # First get the transcription result
        result = transcriber.transcribe_file(harvard_audio_path)

        # Save with timestamps
        output_file = tmp_path / "harvard_transcription.txt"
        transcriber.transcribe_to_file(result, harvard_audio_path, str(output_file), include_timestamps=True)

        # Verify file was created and contains expected content
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")

        assert "harvard.wav" in content
        assert "Language: en" in content
        assert "Model: tiny" in content
        assert "the stale smell of old beer lingers" in content.lower()

        # Check timestamp format
        assert "[" in content and "]" in content  # Should contain timestamp markers

    def test_auto_language_detection(self, harvard_audio_path):
        """Test automatic language detection (without specifying language)"""
        config = TranscriptionConfig(
            model_size=WhisperModelSize.TINY,
            language=None,  # Auto-detect
        )

        transcriber = WhisperTranscriber(config=config)
        result = transcriber.transcribe_file(harvard_audio_path)

        # Should detect English
        assert result.language == "en"
        assert len(result.text) > 0
        assert len(result.text) > 0
        assert len(result.text) > 0
        assert len(result.text) > 0
        assert len(result.text) > 0
        assert len(result.text) > 0
        assert len(result.text) > 0
        assert len(result.text) > 0
