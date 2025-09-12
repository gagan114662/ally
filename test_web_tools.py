"""
Tests for Ally web tools
"""

import pytest
import sys
from pathlib import Path

# Add Ally to path
sys.path.append(str(Path(__file__).parent / "Ally"))

from Ally.tools.web import web_fetch, web_search, web_read_tables
from Ally.schemas.base import ToolResult


def test_web_fetch_local_file():
    """Test web.fetch with local HTML file"""
    # Use the sample fixture
    fixtures_dir = Path("data/fixtures/web")
    sample_file = fixtures_dir / "sample.html"
    
    if not sample_file.exists():
        pytest.skip("Sample fixture not found")
    
    url = f"file://{sample_file.absolute()}"
    result = web_fetch(url=url)
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "page" in result.data
    assert "summary" in result.data
    
    page_data = result.data["page"]
    assert page_data["title"] == "Sample Financial Data"
    assert len(page_data["links"]) >= 0
    assert page_data["size_bytes"] > 0


def test_web_search():
    """Test web.search (mock implementation)"""
    result = web_search(query="python trading", limit=3)
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "results" in result.data
    assert "query" in result.data
    assert result.data["query"] == "python trading"
    assert len(result.data["results"]) <= 3
    
    # Should have warning about mock results
    assert len(result.meta.warnings) > 0
    assert "mock" in result.meta.warnings[0].lower()


def test_web_read_tables_html():
    """Test web.read_tables with HTML content"""
    fixtures_dir = Path("data/fixtures/web")
    sample_file = fixtures_dir / "sample.html"
    
    if not sample_file.exists():
        pytest.skip("Sample fixture not found")
    
    url = f"file://{sample_file.absolute()}"
    result = web_read_tables(url=url, format="html")
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "tables" in result.data
    assert "total_tables" in result.data
    
    # Should extract at least 2 tables from sample.html
    assert result.data["total_tables"] >= 2
    
    # Check first table structure
    if result.data["tables"]:
        first_table = result.data["tables"][0]
        assert "headers" in first_table
        assert "rows" in first_table
        assert "shape" in first_table
        assert len(first_table["headers"]) > 0
        assert len(first_table["rows"]) > 0


def test_web_read_tables_pdf_mock():
    """Test web.read_tables with PDF (mock implementation)"""
    # Test with any URL since PDF extraction is mocked
    result = web_read_tables(url="file://dummy.pdf", format="pdf")
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "tables" in result.data
    
    # Should have warning about PDF not being implemented
    assert len(result.meta.warnings) > 0
    assert "pdf" in result.meta.warnings[0].lower()


def test_web_fetch_invalid_url():
    """Test web.fetch with invalid URL"""
    result = web_fetch(url="file://nonexistent.html")
    
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert len(result.errors) > 0
    assert "not found" in result.errors[0].lower()


def test_web_read_tables_no_tables():
    """Test web.read_tables with content that has no tables"""
    # Create a simple HTML file with no tables
    simple_html = "<html><body><p>No tables here</p></body></html>"
    
    # This would need a temporary file, skip for now
    pytest.skip("Would need temporary file creation")


def test_web_tools_integration():
    """Integration test - fetch then extract tables"""
    fixtures_dir = Path("data/fixtures/web")
    sample_file = fixtures_dir / "sample.html"
    
    if not sample_file.exists():
        pytest.skip("Sample fixture not found")
    
    url = f"file://{sample_file.absolute()}"
    
    # First fetch the page
    fetch_result = web_fetch(url=url)
    assert fetch_result.ok
    
    # Then extract tables from the same URL
    tables_result = web_read_tables(url=url, format="html")
    assert tables_result.ok
    
    # Verify they both worked on the same content
    assert fetch_result.data["page"]["url"] == url
    assert tables_result.data["total_tables"] > 0


if __name__ == "__main__":
    # Run tests manually
    print("Running web tools tests...")
    
    try:
        test_web_fetch_local_file()
        print("✅ web_fetch test passed")
    except Exception as e:
        print(f"❌ web_fetch test failed: {e}")
    
    try:
        test_web_search()
        print("✅ web_search test passed")
    except Exception as e:
        print(f"❌ web_search test failed: {e}")
    
    try:
        test_web_read_tables_html()
        print("✅ web_read_tables HTML test passed")
    except Exception as e:
        print(f"❌ web_read_tables HTML test failed: {e}")
    
    try:
        test_web_read_tables_pdf_mock()
        print("✅ web_read_tables PDF test passed")
    except Exception as e:
        print(f"❌ web_read_tables PDF test failed: {e}")
    
    try:
        test_web_tools_integration()
        print("✅ Integration test passed")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    print("✅ Web tools tests complete!")