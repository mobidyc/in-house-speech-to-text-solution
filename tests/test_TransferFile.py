import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from app.resources.TransferFile import TransferFile, TransferFileRequest


class TestTransferFileRequest:
    def test_transfer_file_request_with_filename_only(self):
        """Test creating TransferFileRequest with only filename"""
        request = TransferFileRequest(filename="test.txt")
        assert request.filename == "test.txt"
        assert request.file_object is None
        assert request.source_path is None
        assert request.destination_key is None

    def test_transfer_file_request_with_all_fields(self):
        """Test creating TransferFileRequest with all fields"""
        file_obj = BytesIO(b"test content")
        request = TransferFileRequest(
            filename="test.txt", file_object=file_obj, source_path="/path/to/source.txt", destination_key="dest/key.txt"
        )
        assert request.filename == "test.txt"
        assert request.file_object == file_obj
        assert request.source_path == "/path/to/source.txt"
        assert request.destination_key == "dest/key.txt"


class TestTransferFile:
    @pytest.fixture
    def mock_config(self):
        """Mock config with S3 settings"""
        with patch("app.resources.TransferFile.config") as mock_config:
            mock_config.s3_endpoint = "https://s3.amazonaws.com"
            mock_config.s3_access_key = "test_access_key"
            mock_config.s3_secret_key = "test_secret_key"
            mock_config.s3_bucket_name = "test-bucket"
            yield mock_config

    @pytest.fixture
    def mock_s3_client(self):
        """Mock boto3 S3 client"""
        with patch("app.resources.TransferFile.boto3.client") as mock_client:
            mock_s3_client = Mock()
            mock_client.return_value = mock_s3_client
            yield mock_s3_client

    def test_transfer_file_initialization_with_config(self, mock_config, mock_s3_client):
        """Test TransferFile initialization using config values"""
        transfer_file = TransferFile()

        assert transfer_file.s3_endpoint == "https://s3.amazonaws.com"
        assert transfer_file.s3_access_key == "test_access_key"
        assert transfer_file.s3_secret_key == "test_secret_key"
        assert transfer_file.s3_bucket_name == "test-bucket"

    def test_transfer_file_initialization_with_parameters(self, mock_s3_client):
        """Test TransferFile initialization with explicit parameters"""
        transfer_file = TransferFile(
            s3_endpoint="https://custom-s3.com",
            s3_access_key="custom_access",
            s3_secret_key="custom_secret",
            s3_bucket_name="custom-bucket",
        )

        assert transfer_file.s3_endpoint == "https://custom-s3.com"
        assert transfer_file.s3_access_key == "custom_access"
        assert transfer_file.s3_secret_key == "custom_secret"
        assert transfer_file.s3_bucket_name == "custom-bucket"

    def test_transfer_file_initialization_missing_s3_config(self):
        """Test TransferFile initialization with missing S3 configuration"""
        with patch("app.resources.TransferFile.config") as mock_config:
            mock_config.s3_endpoint = None
            mock_config.s3_access_key = None
            mock_config.s3_secret_key = None
            mock_config.s3_bucket_name = "test-bucket"

            with pytest.raises(ValueError, match="S3 configuration is incomplete"):
                TransferFile()

    def test_transfer_file_initialization_missing_bucket_name(self):
        """Test TransferFile initialization with missing bucket name"""
        with patch("app.resources.TransferFile.config") as mock_config:
            mock_config.s3_endpoint = "https://s3.amazonaws.com"
            mock_config.s3_access_key = "test_access_key"
            mock_config.s3_secret_key = "test_secret_key"
            mock_config.s3_bucket_name = None

            with pytest.raises(ValueError, match="Bucket name must be provided"):
                TransferFile()

    def test_upload_file_with_file_object(self, mock_config, mock_s3_client):
        """Test uploading file using BytesIO object"""
        transfer_file = TransferFile()

        file_content = b"test file content"
        file_obj = BytesIO(file_content)
        request = TransferFileRequest(filename="test.txt", file_object=file_obj)

        result = transfer_file.upload_file(request)

        assert result is True
        mock_s3_client.upload_fileobj.assert_called_once_with(file_obj, "test-bucket", "test.txt")

    def test_upload_file_with_source_path(self, mock_config, mock_s3_client):
        """Test uploading file using source path"""
        transfer_file = TransferFile()

        request = TransferFileRequest(filename="test.txt", source_path="/path/to/test.txt")

        result = transfer_file.upload_file(request)

        assert result is True
        mock_s3_client.upload_file.assert_called_once_with("/path/to/test.txt", "test-bucket", "test.txt")

    def test_upload_file_with_custom_destination_key(self, mock_config, mock_s3_client):
        """Test uploading file with custom destination key"""
        transfer_file = TransferFile()

        file_obj = BytesIO(b"test content")
        request = TransferFileRequest(filename="test.txt", file_object=file_obj, destination_key="custom/path/test.txt")

        result = transfer_file.upload_file(request)

        assert result is True
        mock_s3_client.upload_fileobj.assert_called_once_with(file_obj, "test-bucket", "custom/path/test.txt")

    def test_upload_file_no_source_provided(self, mock_config, mock_s3_client):
        """Test upload file error when neither file_object nor source_path provided"""
        transfer_file = TransferFile()

        request = TransferFileRequest(filename="test.txt")

        with pytest.raises(Exception, match="Either file_object or source_path must be provided"):
            transfer_file.upload_file(request)

    def test_upload_file_s3_error(self, mock_config, mock_s3_client):
        """Test upload file error handling"""
        transfer_file = TransferFile()
        mock_s3_client.upload_fileobj.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"}}, "upload_fileobj"
        )

        file_obj = BytesIO(b"test content")
        request = TransferFileRequest(filename="test.txt", file_object=file_obj)

        with pytest.raises(Exception, match="Failed to upload file test.txt"):
            transfer_file.upload_file(request)

    def test_download_file_default_paths(self, mock_config, mock_s3_client):
        """Test downloading file with default paths"""
        transfer_file = TransferFile()

        request = TransferFileRequest(filename="test.txt")

        result = transfer_file.download_file(request)

        assert result is True
        mock_s3_client.download_file.assert_called_once_with(Bucket="test-bucket", Key="test.txt", Filename="test.txt")

    def test_download_file_with_custom_paths(self, mock_config, mock_s3_client):
        """Test downloading file with custom source and destination paths"""
        transfer_file = TransferFile()

        request = TransferFileRequest(
            filename="test.txt", source_path="remote/path/test.txt", destination_key="/local/path/test.txt"
        )

        result = transfer_file.download_file(request)

        assert result is True
        mock_s3_client.download_file.assert_called_once_with(
            Bucket="test-bucket", Key="remote/path/test.txt", Filename="/local/path/test.txt"
        )

    def test_download_file_s3_error(self, mock_config, mock_s3_client):
        """Test download file error handling"""
        transfer_file = TransferFile()
        mock_s3_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist"}}, "download_file"
        )

        request = TransferFileRequest(filename="test.txt")

        with pytest.raises(Exception, match="Failed to download file test.txt"):
            transfer_file.download_file(request)

    def test_file_object_seek_behavior(self, mock_config, mock_s3_client):
        """Test that file object is properly seeked to beginning before upload"""
        transfer_file = TransferFile()

        file_obj = BytesIO(b"test content")
        file_obj.read()  # Move to end of file
        assert file_obj.tell() != 0  # Verify we're not at the beginning

        request = TransferFileRequest(filename="test.txt", file_object=file_obj)

        transfer_file.upload_file(request)

        # Verify file object was seeked to beginning
        assert file_obj.tell() == 0
        mock_s3_client.upload_fileobj.assert_called_once()

    @patch("builtins.print")
    def test_download_file_prints_debug_info(self, mock_print, mock_config, mock_s3_client):
        """Test that download_file prints debug information"""
        transfer_file = TransferFile()

        request = TransferFileRequest(
            filename="test.txt", source_path="remote/test.txt", destination_key="/local/test.txt"
        )

        transfer_file.download_file(request)

        mock_print.assert_called_once_with("Downloading file from S3: test-bucket remote/test.txt to /local/test.txt")


@pytest.mark.functional
class TestTransferFileFunctional:
    """Functional tests for TransferFile (requires actual S3 setup)"""

    def test_real_file_upload_download_cycle(self):
        """Functional test for complete upload/download cycle"""
        # Skip if no real S3 credentials available
        pytest.skip("Functional test requires actual S3 credentials - implement when needed")

        # This would be implemented when actual S3 testing is needed
        # with tempfile.NamedTemporaryFile() as temp_file:
        #     temp_file.write(b"test content for functional test")
        #     temp_file.flush()
        #
        #     transfer_file = TransferFile()
        #
        #     # Upload
        #     upload_request = TransferFileRequest(
        #         filename="functional_test.txt",
        #         source_path=temp_file.name
        #     )
        #     assert transfer_file.upload_file(upload_request)
        #
        #     # Download
        #     with tempfile.NamedTemporaryFile() as download_file:
        #         download_request = TransferFileRequest(
        #             filename="functional_test.txt",
        #             destination_key=download_file.name
        #         )
        #         assert transfer_file.download_file(download_request)
        #
        #         # Verify content
        #         download_file.seek(0)
        #         assert download_file.read() == b"test content for functional test"
