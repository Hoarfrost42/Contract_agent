from __future__ import annotations

import io
from pathlib import Path
from typing import IO

import docx
import pdfplumber


def _read_bytes(uploaded_file: IO[bytes]) -> bytes:
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return data


def _load_docx(data: bytes) -> str:
    document = docx.Document(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def _load_pdf(data: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            content = page.extract_text() or ""
            if content.strip():
                text_parts.append(content.strip())
    return "\n".join(text_parts)


def _load_txt(data: bytes) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("decode", data, 0, len(data), "Unable to decode text file")


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract text from an uploaded file (.docx, .pdf, .txt).
    """

    suffix = Path(uploaded_file.name).suffix.lower()
    data = _read_bytes(uploaded_file)

    try:
        if suffix == ".docx":
            return _load_docx(data)
        if suffix == ".pdf":
            return _load_pdf(data)
        if suffix == ".txt":
            return _load_txt(data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法解析文件：{exc}") from exc

    raise ValueError("不支持的文件类型，请上传 .txt/.pdf/.docx 文件。")
