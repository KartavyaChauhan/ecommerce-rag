"""Service for uploading, validating, and chunking PDF documents."""

from typing import List
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import logger


class DocumentService:
    """Handles PDF ingestion and text chunking using recursive character splitting."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    async def process_file(self, file: UploadFile) -> List[str]:
        """Validate, extract text from, and chunk a PDF file.

        Args:
            file: The uploaded PDF file.

        Returns:
            A list of text chunks ready for embedding.

        Raises:
            HTTPException: If the file type is invalid or text extraction fails.
        """
        logger.info(f"Processing file: {file.filename}")

        if file.content_type != "application/pdf":
            logger.error(f"Rejected file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        try:
            pdf_reader = PdfReader(file.file)
            text_content = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"

            if not text_content.strip():
                raise HTTPException(
                    status_code=422,
                    detail="PDF contains no extractable text (may be scanned/image-based).",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

        chunks = self.text_splitter.split_text(text_content)
        logger.info(f"Processed '{file.filename}' into {len(chunks)} chunks.")
        return chunks


document_service = DocumentService()