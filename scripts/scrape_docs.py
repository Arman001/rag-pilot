# scripts/scrape_docs.py
import time
from pathlib import Path
from config.settings import settings
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
from markdownify import markdownify
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_page(url: str) -> str | None:
    """Fetch HTML with retries and rate limiting."""
    try:
        time.sleep(settings.REQUEST_DELAY)  # Rate limiting
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")
    
    # Try these common content containers (one will likely work)
    for class_name in [
        "main-content",  # New LangChain container
        "docMainContainer",  # Old container
        "theme-doc-markdown",  # Another possible class
        "markdown"  # Fallback option
    ]:
        container = soup.find("div", class_=class_name)
        if container:
            # Clean up unwanted elements
            for element in container.find_all(["nav", "header", "footer", "div.admonition"]):
                element.decompose()
            return str(container)
    
    return None

def html_to_markdown(html):
    """Convert HTML to Markdown and clean it."""
    md = markdownify(html, heading_style="ATX")
    # Remove excessive newlines
    return "\n".join(line for line in md.splitlines() if line.strip())

def save_content(content, filename):
    """Save content to a file."""
    path = os.path.join(settings.OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")

def scrape_langchain_docs():
    """Main scraping workflow using centralized config."""
    logger.info(f"Scraping from {settings.BASE_URL}")
    
    html = fetch_page(settings.BASE_URL)
    if not html:
        return

    main_content = extract_main_content(html)
    if not main_content:
        logger.warning(f"No main content found at {settings.BASE_URL}")
        return

    markdown = html_to_markdown(main_content)
    save_path = Path(settings.OUTPUT_DIR) / "introduction.md"
    save_path.write_text(markdown, encoding="utf-8")
    logger.info(f"Saved to {save_path}")

if __name__ == "__main__":
    scrape_langchain_docs()