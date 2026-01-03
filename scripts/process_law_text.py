import re
import os
from pathlib import Path

def process_law_file(input_path: str, output_dir: str):
    """
    Process raw law text file into the standard format:
    《LawName》ArticleNumber规定，Content
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File not found: {input_path}")
        return

    law_name = input_file.stem # e.g., "中华人民共和国劳动法"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_articles = []
    current_article_header = ""
    current_article_content = []
    
    # Regex to match "第X条" at the start of a line (allowing for whitespace)
    # Using Chinese numerals
    article_pattern = re.compile(r"^\s*(第[零一二三四五六七八九十百]+条)")

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip Chapter headers (e.g., "第一章 总 则")
        if re.match(r"^\s*第[零一二三四五六七八九十百]+章", line):
            continue

        match = article_pattern.match(line)
        if match:
            # If we have a previous article accumulating, save it
            if current_article_header:
                full_content = "".join(current_article_content)
                # Remove any internal newlines or excessive spaces if they exist in content
                full_content = re.sub(r"\s+", "", full_content) 
                # Note: The raw text often has spaces inside sentences, but the processed example 
                # shows clean text. Let's just strip whitespace.
                # Actually, looking at the raw file: "第一条　为了..."
                # The content part starts after the article number.
                
                # Let's refine: The content currently includes the header for the first line.
                # We need to separate header and content for the first line?
                # No, the logic below handles "header" as just the "第X条" string.
                
                # Wait, my loop logic needs to be careful.
                # When we hit a NEW article, we process the OLD one.
                
                # Format: 《LawName》ArticleNumber规定，Content
                # The "Content" should not include the ArticleNumber again if we prefix it?
                # Looking at processed example: 
                # 《中华人民共和国劳动合同法》第一条规定，为了...
                # Raw: 第一条　为了...
                # So we replace "第一条" with "《...》第一条规定，"
                
                # Let's reconstruct the content properly.
                pass

            # Start new article
            current_article_header = match.group(1) # "第一条"
            
            # content of this line after the header
            # Remove the header from the line
            content_part = line[len(current_article_header):].strip()
            
            # If there is a full-width space or space after "第一条", strip it
            # Raw: "第一条　为了..." -> content_part: "为了..."
            
            current_article_content = [content_part]
            
            # If we have a previous article to save
            if len(processed_articles) > 0 or (current_article_header and len(current_article_content) > 1):
                 # Wait, the "save previous" logic should be at the TOP of the loop or handled via a buffer
                 pass
        else:
            # Continuation of previous article
            if current_article_header:
                current_article_content.append(line)

    # Refined Loop Logic
    articles = []
    current_header = None
    current_text_parts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip Chapter headers
        if re.match(r"^\s*第[零一二三四五六七八九十百]+章", line):
            continue

        match = article_pattern.match(line)
        if match:
            # Save previous article if exists
            if current_header:
                full_text = "".join(current_text_parts)
                # Clean up text: remove spaces? The processed example looks like standard Chinese text.
                # Raw text has spaces for indentation but we stripped line.
                # Inner spaces? "总　　则" -> "总则". 
                # Let's just join.
                articles.append((current_header, full_text))

            # Start new
            current_header = match.group(1)
            # Remove header from line to get initial text
            # Be careful with "第十一条" vs "第一条"
            # The regex matched the start.
            
            # Find where the header ends in the line
            # usually "第X条" followed by whitespace
            # line is stripped.
            
            # Let's use the match span
            start, end = match.span()
            remaining = line[end:].strip()
            current_text_parts = [remaining]
        else:
            if current_header:
                current_text_parts.append(line)
    
    # Save last article
    if current_header:
        full_text = "".join(current_text_parts)
        articles.append((current_header, full_text))

    # Write output
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_path / f"{law_name}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for header, content in articles:
            # Format: 《LawName》ArticleNumber规定，Content
            # Note: The processed example has "规定，" added.
            # Raw: "第一条　为了..." -> "《...》第一条规定，为了..."
            # So we construct the line.
            
            # Check if content starts with punctuation? usually not.
            line_str = f"《{law_name}》{header}规定，{content}\n"
            f.write(line_str)
            
    print(f"Processed {len(articles)} articles.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    # Configuration
    RAW_DIR = r"F:\Agent_back\11.29\Evaldata\raw_laws"
    PROCESSED_DIR = r"F:\Agent_back\11.29\Evaldata\processed_laws"
    
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print(f"Error: Raw directory not found: {RAW_DIR}")
        exit(1)

    print(f"Processing law files from: {RAW_DIR}")
    
    count = 0
    for file_path in raw_path.glob("*.txt"):
        print(f"Processing: {file_path.name}")
        process_law_file(str(file_path), PROCESSED_DIR)
        count += 1
        
    print(f"Done. Processed {count} files.")
