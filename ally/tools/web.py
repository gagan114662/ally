"""
Web tools for Ally - fetch, search, and table extraction
"""

import time
import requests
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from readability import Document
import hashlib

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.web import WebFetchIn, WebSearchIn, WebReadTablesIn, WebPage, SearchResult, TableData
from ..utils.hashing import content_hash
from ..utils.io import ensure_dir, safe_write_text, safe_read_text

# Cache directory for web content
CACHE_DIR = Path("data/cache/web")


@register("web.fetch")
def web_fetch(**kwargs) -> ToolResult:
    """
    Fetch and clean web content
    
    Supports URLs and local file:// paths, with content cleaning and caching
    """
    try:
        inputs = WebFetchIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    start_time = time.time()
    warnings = []
    
    try:
        # Handle file:// URLs
        if inputs.url.startswith("file://"):
            file_path = inputs.url.replace("file://", "")
            if not Path(file_path).exists():
                return ToolResult.error([f"File not found: {file_path}"])
                
            content = safe_read_text(file_path)
            content_type = "text/html"
            
            if inputs.as_pdf:
                warnings.append("PDF processing not supported for local files")
                
        else:
            # HTTP(S) request
            headers = {
                "User-Agent": inputs.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
            
            response = requests.get(
                inputs.url,
                headers=headers,
                timeout=inputs.timeout_s,
                allow_redirects=inputs.follow_redirects
            )
            response.raise_for_status()
            
            # Check content size
            content_length = len(response.content)
            max_bytes = int(inputs.max_size_mb * 1024 * 1024)
            if content_length > max_bytes:
                return ToolResult.error([f"Content too large: {content_length} bytes > {max_bytes} bytes"])
                
            content = response.text
            content_type = response.headers.get("content-type", "text/html")
            
        # Cache content
        content_hash_value = content_hash(content.encode())
        cache_file = CACHE_DIR / f"{content_hash_value}.html"
        ensure_dir(CACHE_DIR)
        safe_write_text(content, cache_file)
        
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else None
        
        # Clean content with readability
        try:
            doc = Document(content)
            clean_text = BeautifulSoup(doc.summary(), 'html.parser').get_text()
            clean_text = ' '.join(clean_text.split())  # Normalize whitespace
        except Exception:
            # Fallback to basic text extraction
            clean_text = soup.get_text()
            clean_text = ' '.join(clean_text.split())
            warnings.append("Readability extraction failed, using basic text extraction")
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:50]:  # Limit to first 50 links
            href = link.get('href')
            if href:
                # Resolve relative URLs
                if not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                    href = urllib.parse.urljoin(inputs.url, href)
                    
                links.append({
                    'text': link.get_text().strip()[:100],  # Limit text length
                    'url': href
                })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True)[:20]:  # Limit to first 20 images
            src = img.get('src')
            if src:
                if not src.startswith(('http://', 'https://')):
                    src = urllib.parse.urljoin(inputs.url, src)
                    
                images.append({
                    'src': src,
                    'alt': img.get('alt', '').strip()[:100],
                    'title': img.get('title', '').strip()[:100]
                })
        
        # Create page object
        page = WebPage(
            url=inputs.url,
            title=title_text,
            content=content,
            clean_text=clean_text[:10000],  # Limit clean text to 10k chars
            links=links,
            images=images,
            metadata={
                'content_type': content_type,
                'content_hash': content_hash_value,
                'cache_file': str(cache_file)
            },
            content_type=content_type,
            size_bytes=len(content.encode()),
            fetch_time=time.time() - start_time
        )
        
        return ToolResult.success(
            data={
                'page': page.model_dump(),
                'cache_file': str(cache_file),
                'summary': {
                    'title': title_text,
                    'text_length': len(clean_text),
                    'links_count': len(links),
                    'images_count': len(images)
                }
            },
            warnings=warnings
        )
        
    except requests.exceptions.Timeout:
        return ToolResult.error([f"Request timeout after {inputs.timeout_s}s"])
    except requests.exceptions.RequestException as e:
        return ToolResult.error([f"Request failed: {e}"])
    except Exception as e:
        return ToolResult.error([f"Unexpected error: {e}"])


@register("web.search")
def web_search(**kwargs) -> ToolResult:
    """
    Search the web using DuckDuckGo (privacy-focused)
    
    Returns structured search results with ranking and metadata
    """
    try:
        inputs = WebSearchIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    warnings = []
    
    # Mock implementation for demonstration
    # In production, integrate with DuckDuckGo API or similar
    mock_results = [
        SearchResult(
            title=f"Sample result {i+1} for: {inputs.query}",
            url=f"https://example.com/result-{i+1}",
            snippet=f"This is a sample search result snippet for {inputs.query}. "
                   f"It contains relevant information about the query topic.",
            rank=i+1,
            domain="example.com"
        )
        for i in range(min(inputs.limit, 3))
    ]
    
    warnings.append("Using mock search results - integrate with real search API in production")
    
    return ToolResult.success(
        data={
            'query': inputs.query,
            'results': [result.model_dump() for result in mock_results],
            'total_results': len(mock_results),
            'search_time': 0.1
        },
        warnings=warnings
    )


@register("web.read_tables") 
def web_read_tables(**kwargs) -> ToolResult:
    """
    Extract tables from HTML or PDF content
    
    Supports both web URLs and local files
    """
    try:
        inputs = WebReadTablesIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    warnings = []
    tables = []
    
    try:
        # Get content
        if inputs.url.startswith("file://"):
            file_path = inputs.url.replace("file://", "")
            if not Path(file_path).exists():
                return ToolResult.error([f"File not found: {file_path}"])
            content = safe_read_text(file_path, encoding=inputs.encoding)
        else:
            # Fetch from web
            response = requests.get(inputs.url, timeout=30)
            response.raise_for_status()
            content = response.text
        
        if inputs.format.lower() == "pdf":
            warnings.append("PDF table extraction not implemented - returning mock data")
            # Mock table for PDF
            tables = [
                TableData(
                    headers=["Date", "Price", "Volume"],
                    rows=[
                        ["2024-01-01", "100.0", "1000"],
                        ["2024-01-02", "101.5", "1200"],
                        ["2024-01-03", "99.8", "950"]
                    ],
                    table_index=0,
                    shape=(3, 3),
                    metadata={"source": "pdf", "page": 1}
                )
            ]
        else:
            # Extract HTML tables
            soup = BeautifulSoup(content, 'html.parser')
            html_tables = soup.find_all('table')
            
            if not html_tables:
                return ToolResult.success(
                    data={'tables': [], 'total_tables': 0},
                    warnings=["No tables found in content"]
                )
            
            for idx, table in enumerate(html_tables):
                if inputs.table_index is not None and idx != inputs.table_index:
                    continue
                    
                # Extract headers
                headers = []
                header_row = table.find('tr')
                if header_row:
                    for th in header_row.find_all(['th', 'td']):
                        headers.append(th.get_text().strip())
                
                # Extract rows
                rows = []
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = []
                    for td in row.find_all(['td', 'th']):
                        cells.append(td.get_text().strip())
                    if cells:
                        rows.append(cells)
                
                if rows:
                    table_data = TableData(
                        headers=headers,
                        rows=rows,
                        table_index=idx,
                        shape=(len(rows), len(headers) if headers else len(rows[0]) if rows else 0),
                        metadata={"source": "html", "url": inputs.url}
                    )
                    tables.append(table_data)
        
        return ToolResult.success(
            data={
                'tables': [table.model_dump() for table in tables],
                'total_tables': len(tables)
            },
            warnings=warnings
        )
        
    except requests.exceptions.RequestException as e:
        return ToolResult.error([f"Failed to fetch content: {e}"])
    except Exception as e:
        return ToolResult.error([f"Table extraction failed: {e}"])


# Helper function to create sample fixtures
def create_sample_fixtures():
    """Create sample HTML and PDF fixtures for testing"""
    fixtures_dir = ensure_dir("data/fixtures/web")
    
    # Sample HTML with tables
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Financial Data</title>
    </head>
    <body>
        <h1>Market Data</h1>
        <p>This is a sample page with financial data tables.</p>
        
        <h2>Stock Prices</h2>
        <table border="1">
            <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>Change</th>
                <th>Volume</th>
            </tr>
            <tr>
                <td>AAPL</td>
                <td>$150.00</td>
                <td>+2.5%</td>
                <td>1,000,000</td>
            </tr>
            <tr>
                <td>GOOGL</td>
                <td>$2800.00</td>
                <td>-1.2%</td>
                <td>500,000</td>
            </tr>
            <tr>
                <td>TSLA</td>
                <td>$800.00</td>
                <td>+5.1%</td>
                <td>2,000,000</td>
            </tr>
        </table>
        
        <h2>Economic Indicators</h2>
        <table border="1">
            <tr>
                <th>Indicator</th>
                <th>Value</th>
                <th>Previous</th>
            </tr>
            <tr>
                <td>GDP Growth</td>
                <td>2.3%</td>
                <td>2.1%</td>
            </tr>
            <tr>
                <td>Unemployment</td>
                <td>3.8%</td>
                <td>3.9%</td>
            </tr>
        </table>
        
        <p>Data as of 2024.</p>
    </body>
    </html>
    """
    
    safe_write_text(sample_html, fixtures_dir / "sample.html")
    
    # Create a simple "PDF" (text file for now)
    sample_pdf_content = """
    Sample PDF Content
    
    This would contain tabular data extracted from a real PDF.
    In production, this would use libraries like camelot-py or tabula-py.
    """
    
    safe_write_text(sample_pdf_content, fixtures_dir / "sample.pdf")
    
    return fixtures_dir


if __name__ == "__main__":
    # Create sample fixtures for testing
    fixtures_dir = create_sample_fixtures()
    print(f"Created sample fixtures in: {fixtures_dir}")
    
    # Test web.fetch with local file
    sample_url = f"file://{fixtures_dir / 'sample.html'}"
    result = web_fetch(url=sample_url)
    print(f"Fetch test result: {result.ok}")
    
    # Test web.read_tables
    table_result = web_read_tables(url=sample_url, format="html")
    print(f"Table extraction test result: {table_result.ok}")
    if table_result.ok:
        print(f"Tables found: {table_result.data['total_tables']}")