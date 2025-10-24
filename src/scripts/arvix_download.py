#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🌙 Moon Dev's arXiv Paper Downloader 🚀
A tool to search and download papers from arXiv.org
"""

# Standard library imports
import argparse
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Standard library from imports
from pathlib import Path

# ============= CONFIGURATION SETTINGS (EDIT THESE) =============
# Search topic - change this to whatever you want to search for
SEARCH_TOPIC = "market making"  # Try: "mean reversion", "momentum", "market making", etc.

# Search category - q-fin is for Quantitative Finance
SEARCH_CATEGORY = "q-fin"  # Options: "q-fin", "cs", "econ", etc.

# Number of papers to download
MAX_PAPERS_TO_DOWNLOAD = 50  # Change this to download more or fewer papers

# Where to save the papers (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOWNLOAD_DIRECTORY = os.path.join(PROJECT_ROOT, "src", "data", "Arxiv")

# Sort order for results
SORT_BY = "relevance"  # Options: "relevance", "lastUpdatedDate", "submittedDate"
SORT_ORDER = "descending"  # Options: "ascending", "descending"

# Track downloaded papers to avoid duplicates
DOWNLOADED_PAPERS_FILE = os.path.join(DOWNLOAD_DIRECTORY, "downloaded_papers.json")
# ===============================================================

# Add required packages to requirements.txt if not already there
try:
    from tqdm import tqdm
except ImportError:
    print("⚠️ tqdm not found. Please add it to requirements.txt and install it.")
    print("pip install tqdm")

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Easter eggs and fun messages
MOON_DEV_QUOTES = [
    "🌙 Moon Dev says: To the moon with knowledge! 🚀",
    "🌙 Thanks for using Moon Dev's arXiv downloader! 💫",
    "🌙 Moon Dev approves this download! 👍",
    "🌙 Moon Dev reminds you: Knowledge is power! 📚",
    "🌙 Moon Dev says: Learning never stops! 🧠",
    "🌙 Moon Dev's secret to success: Read more papers! 📝",
    "🌙 Moon Dev is proud of you for doing research! 🔍",
]

def print_banner():
    """Print a fancy banner for the tool"""
    banner = f"""
    {Colors.CYAN}╔═══════════════════════════════════════════════════════════╗
    ║ {Colors.YELLOW}🌙  MOON DEV's arXiv PAPER DOWNLOADER  🌙{Colors.CYAN}                ║
    ╚═══════════════════════════════════════════════════════════╝{Colors.ENDC}
    """
    print(banner)
    print(f"{Colors.GREEN}{random.choice(MOON_DEV_QUOTES)}{Colors.ENDC}\n")
    print(f"{Colors.BOLD}Current search topic:{Colors.ENDC} {Colors.YELLOW}{SEARCH_TOPIC}{Colors.ENDC}")
    print(f"{Colors.BOLD}Category:{Colors.ENDC} {Colors.YELLOW}{SEARCH_CATEGORY}{Colors.ENDC}")
    print(f"{Colors.BOLD}Max papers:{Colors.ENDC} {Colors.YELLOW}{MAX_PAPERS_TO_DOWNLOAD}{Colors.ENDC}")
    print(f"{Colors.BOLD}Download directory:{Colors.ENDC} {Colors.YELLOW}{DOWNLOAD_DIRECTORY}{Colors.ENDC}")
    topic_dir = os.path.join(DOWNLOAD_DIRECTORY, f"{SEARCH_CATEGORY}_{SEARCH_TOPIC.replace(' ', '_')}")
    print(f"{Colors.BOLD}Papers will be saved to:{Colors.ENDC} {Colors.YELLOW}{topic_dir}{Colors.ENDC}\n")

def create_download_dir(download_dir):
    """Create download directory if it doesn't exist"""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"📁 Created download directory: {download_dir}")
    return download_dir

def load_downloaded_papers():
    """Load the list of previously downloaded papers"""
    if not os.path.exists(DOWNLOADED_PAPERS_FILE):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(DOWNLOADED_PAPERS_FILE), exist_ok=True)
        return {}
    
    try:
        with open(DOWNLOADED_PAPERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_downloaded_paper(paper_id, paper_info):
    """Save a paper to the downloaded papers list"""
    downloaded_papers = load_downloaded_papers()
    
    # Add the paper to the list
    downloaded_papers[paper_id] = {
        'title': paper_info.get('title', ''),
        'authors': paper_info.get('authors', []),
        'categories': paper_info.get('categories', []),
        'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'filepath': paper_info.get('filepath', '')
    }
    
    # Save the updated list
    with open(DOWNLOADED_PAPERS_FILE, 'w') as f:
        json.dump(downloaded_papers, f, indent=2)

def is_paper_downloaded(paper_id):
    """Check if a paper has already been downloaded"""
    downloaded_papers = load_downloaded_papers()
    return paper_id in downloaded_papers

def search_arxiv(query, category=None, start=0, max_results=10, sort_by='relevance', sort_order='descending'):
    """
    Search arXiv API for papers matching the query
    
    Parameters:
    - query: Search query string
    - category: Category to search in (e.g., 'q-fin', 'cs', etc.)
    - start: Starting index for results
    - max_results: Maximum number of results to return
    - sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'
    - sort_order: 'ascending' or 'descending'
    
    Returns:
    - List of paper dictionaries with metadata
    """
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Construct search query with category if provided
    if category:
        # Try different query formats
        if "." in category:  # If it's a subcategory like q-fin.GN
            search_query = f"cat:{category} AND all:{query}"
        else:
            # For main categories, we'll search in all subcategories
            search_query = f"cat:{category}* AND all:{query}"
    else:
        search_query = f"all:{query}"
    
    # Construct query parameters
    params = {
        'search_query': search_query,
        'start': start,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order
    }
    
    # Encode parameters and create full URL
    query_string = urllib.parse.urlencode(params)
    url = base_url + query_string
    
    print(f"🔍 Searching arXiv for: {Colors.YELLOW}{query}{Colors.ENDC} in category {Colors.YELLOW}{category if category else 'all'}{Colors.ENDC}")
    print(f"🌐 API URL: {url}")
    print(f"🔎 Search query: {Colors.CYAN}{search_query}{Colors.ENDC}")
    
    # Make the request
    with urllib.request.urlopen(url) as response:
        response_data = response.read().decode('utf-8')
    
    # Parse XML response
    root = ET.fromstring(response_data)
    
    # Define XML namespaces
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
    }
    
    # Extract total results count
    total_results = root.find('.//opensearch:totalResults', namespaces)
    if total_results is not None:
        print(f"📊 Found {Colors.GREEN}{total_results.text}{Colors.ENDC} total results")
    
    # Extract papers
    papers = []
    entries = root.findall('.//atom:entry', namespaces)
    
    for entry in entries:
        paper = {}
        
        # Extract basic metadata
        paper['id'] = entry.find('./atom:id', namespaces).text
        paper['arxiv_id'] = paper['id'].split('/')[-1]
        paper['title'] = entry.find('./atom:title', namespaces).text.strip()
        paper['summary'] = entry.find('./atom:summary', namespaces).text.strip()
        paper['published'] = entry.find('./atom:published', namespaces).text
        
        # Extract authors
        authors = entry.findall('./atom:author/atom:name', namespaces)
        paper['authors'] = [author.text for author in authors]
        
        # Extract PDF link
        links = entry.findall('./atom:link', namespaces)
        for link in links:
            if link.get('title') == 'pdf':
                paper['pdf_url'] = link.get('href')
                break
        
        # Extract categories
        categories = entry.findall('./atom:category', namespaces)
        paper['categories'] = [category.get('term') for category in categories]
        
        # Extract comments and journal references if available
        comment = entry.find('./arxiv:comment', namespaces)
        if comment is not None:
            paper['comment'] = comment.text
        
        journal_ref = entry.find('./arxiv:journal_ref', namespaces)
        if journal_ref is not None:
            paper['journal_ref'] = journal_ref.text
        
        papers.append(paper)
    
    return papers

def display_papers(papers):
    """Display paper information in a readable format"""
    if not papers:
        print(f"{Colors.RED}No papers found matching your query.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.CYAN}Found {len(papers)} papers:{Colors.ENDC}\n")
    
    for i, paper in enumerate(papers):
        # Check if this paper is already downloaded
        already_downloaded = is_paper_downloaded(paper['arxiv_id'])
        
        # Display title with indicator if already downloaded
        if already_downloaded:
            print(f"{Colors.BOLD}{i+1}. {Colors.RED}[ALREADY DOWNLOADED] {Colors.YELLOW}{paper['title']}{Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}{i+1}. {Colors.YELLOW}{paper['title']}{Colors.ENDC}")
            
        print(f"   {Colors.BLUE}Authors:{Colors.ENDC} {', '.join(paper['authors'])}")
        print(f"   {Colors.BLUE}arXiv ID:{Colors.ENDC} {paper['arxiv_id']}")
        print(f"   {Colors.BLUE}Categories:{Colors.ENDC} {', '.join(paper['categories'])}")
        print(f"   {Colors.BLUE}Published:{Colors.ENDC} {paper['published'][:10]}")
        
        if 'journal_ref' in paper:
            print(f"   {Colors.BLUE}Journal Ref:{Colors.ENDC} {paper['journal_ref']}")
        
        # Print truncated summary
        summary = paper['summary'].replace('\n', ' ').strip()
        if len(summary) > 200:
            summary = summary[:200] + "..."
        print(f"   {Colors.BLUE}Summary:{Colors.ENDC} {summary}")
        print()

def download_paper(paper, download_dir, show_progress=True):
    """Download a paper PDF to the specified directory"""
    if 'pdf_url' not in paper:
        print(f"{Colors.RED}Error: No PDF URL found for paper {paper['arxiv_id']}{Colors.ENDC}")
        return False
    
    # Check if paper has already been downloaded (by ID)
    if is_paper_downloaded(paper['arxiv_id']):
        print(f"{Colors.RED}{Colors.BOLD}⚠️ DUPLICATE DETECTED: {paper['arxiv_id']} - {paper['title']}{Colors.ENDC}")
        print(f"{Colors.RED}⏩ Skipping download - Paper already in tracking database{Colors.ENDC}")
        return True
    
    # Clean the title to create a valid filename
    clean_title = re.sub(r'[^\w\s-]', '', paper['title'])
    clean_title = re.sub(r'[\s]+', '_', clean_title)
    filename = f"{paper['arxiv_id']}_{clean_title[:50]}.pdf"
    filepath = os.path.join(download_dir, filename)
    
    # Check if file already exists on disk but isn't in our tracking database
    if os.path.exists(filepath):
        print(f"{Colors.YELLOW}⚠️ File exists on disk but not in tracking database: {filepath}{Colors.ENDC}")
        # Add it to our tracking system
        paper['filepath'] = filepath
        save_downloaded_paper(paper['arxiv_id'], paper)
        print(f"{Colors.GREEN}✅ Added to tracking database - No download needed{Colors.ENDC}")
        return True
    
    print(f"📥 Downloading: {Colors.GREEN}{paper['title']}{Colors.ENDC}")
    print(f"   To: {filepath}")
    
    try:
        if show_progress:
            # Open the URL
            response = urllib.request.urlopen(paper['pdf_url'])
            file_size = int(response.info().get('Content-Length', 0))
            
            # Create a progress bar
            progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, 
                               desc=f"🌙 Moon Dev's Download", ncols=100)
            
            # Download with progress
            with open(filepath, 'wb') as f:
                while True:
                    buffer = response.read(8192)
                    if not buffer:
                        break
                    f.write(buffer)
                    progress_bar.update(len(buffer))
            
            progress_bar.close()
        else:
            # Simple download without progress bar
            urllib.request.urlretrieve(paper['pdf_url'], filepath)
        
        print(f"{Colors.GREEN}✅ Successfully downloaded paper to {filepath}{Colors.ENDC}")
        print(f"{Colors.CYAN}{random.choice(MOON_DEV_QUOTES)}{Colors.ENDC}")
        
        # Record the downloaded paper
        paper['filepath'] = filepath
        save_downloaded_paper(paper['arxiv_id'], paper)
        
        return True
    
    except Exception as e:
        print(f"{Colors.RED}❌ Error downloading paper: {str(e)}{Colors.ENDC}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def batch_download(papers, download_dir, max_papers=5, delay=1.0):
    """Download multiple papers with a delay between requests"""
    success_count = 0
    skipped_count = 0
    papers_to_download = papers[:max_papers]
    
    print(f"\n{Colors.BOLD}Processing {len(papers_to_download)} of {len(papers)} papers{Colors.ENDC}")
    
    for i, paper in enumerate(papers_to_download):
        print(f"\n{Colors.BOLD}Paper {i+1} of {len(papers_to_download)}: {paper['title']}{Colors.ENDC}")
        
        # Check if already downloaded before attempting download
        if is_paper_downloaded(paper['arxiv_id']):
            print(f"{Colors.RED}{Colors.BOLD}⚠️ DUPLICATE DETECTED: {paper['arxiv_id']}{Colors.ENDC}")
            print(f"{Colors.RED}⏩ Skipping download - Paper already in tracking database{Colors.ENDC}")
            skipped_count += 1
        else:
            # Not in database, try to download
            if download_paper(paper, download_dir):
                success_count += 1
        
        # Add delay between downloads to be nice to arXiv servers
        if i < len(papers_to_download) - 1:
            time.sleep(delay)
    
    # Print summary statistics
    print(f"\n{Colors.CYAN}=== Download Summary ==={Colors.ENDC}")
    print(f"{Colors.GREEN}✅ Successfully downloaded: {success_count} papers{Colors.ENDC}")
    print(f"{Colors.YELLOW}⏩ Skipped (already downloaded): {skipped_count} papers{Colors.ENDC}")
    print(f"{Colors.BLUE}📚 Total processed: {success_count + skipped_count} papers{Colors.ENDC}")
    
    return success_count

def auto_download_by_topic(topic, category=None, max_papers=5, download_dir=None, 
                          sort_by='submittedDate', sort_order='descending'):
    """Automatically search and download papers based on a topic"""
    if download_dir is None:
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads", "arxiv_papers")
    
    # Create topic-specific directory
    topic_dir = os.path.join(download_dir, f"{category or 'all'}_{topic.replace(' ', '_')}")
    topic_dir = create_download_dir(topic_dir)
    
    # Search for papers with the specified category
    print(f"{Colors.BOLD}🔍 Searching for '{topic}' in {category or 'all'} category...{Colors.ENDC}")
    papers = search_arxiv(topic, category=category, max_results=max_papers, sort_by=sort_by, sort_order=sort_order)
    
    # If no results, try without category constraint
    if not papers and category:
        print(f"{Colors.YELLOW}⚠️ No results found in {category}. Trying without category constraint...{Colors.ENDC}")
        papers = search_arxiv(topic, category=None, max_results=max_papers, sort_by=sort_by, sort_order=sort_order)
    
    # If still no results, try with a more specific search
    if not papers:
        print(f"{Colors.YELLOW}⚠️ No results found. Trying with a more specific search...{Colors.ENDC}")
        # Try with quotes around the topic
        papers = search_arxiv(f'"{topic}"', category=category, max_results=max_papers, sort_by=sort_by, sort_order=sort_order)
    
    # If still no results, try with a broader search
    if not papers:
        print(f"{Colors.YELLOW}⚠️ No results found. Trying with a broader search...{Colors.ENDC}")
        # Try with OR operator for related terms
        related_terms = {
            "strategy": "trading OR investment OR portfolio OR algorithmic",
            "mean reversion": "mean-reversion OR stationary OR cointegration",
            "momentum": "trend-following OR price-momentum OR cross-sectional",
            "market making": "liquidity OR bid-ask OR spread OR dealer",
            # Add more mappings as needed
        }
        
        broader_query = related_terms.get(topic.lower(), topic)
        papers = search_arxiv(broader_query, category=category, max_results=max_papers, sort_by=sort_by, sort_order=sort_order)
    
    if not papers:
        print(f"{Colors.RED}No papers found for topic: {topic} after multiple search attempts{Colors.ENDC}")
        return 0
    
    # Display found papers
    display_papers(papers)
    
    # Download papers
    return batch_download(papers, topic_dir, max_papers=max_papers)

def display_downloaded_papers():
    """Display a list of previously downloaded papers"""
    downloaded_papers = load_downloaded_papers()
    
    if not downloaded_papers:
        print(f"{Colors.YELLOW}No papers have been downloaded yet.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.CYAN}Previously Downloaded Papers ({len(downloaded_papers)} total):{Colors.ENDC}\n")
    
    for i, (paper_id, paper_info) in enumerate(downloaded_papers.items()):
        print(f"{Colors.BOLD}{i+1}. {Colors.YELLOW}{paper_info['title']}{Colors.ENDC}")
        print(f"   {Colors.BLUE}arXiv ID:{Colors.ENDC} {paper_id}")
        print(f"   {Colors.BLUE}Authors:{Colors.ENDC} {', '.join(paper_info['authors'])}")
        print(f"   {Colors.BLUE}Categories:{Colors.ENDC} {', '.join(paper_info['categories'])}")
        print(f"   {Colors.BLUE}Downloaded:{Colors.ENDC} {paper_info['download_date']}")
        print(f"   {Colors.BLUE}File:{Colors.ENDC} {paper_info['filepath']}")
        print()

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's arXiv Paper Downloader")
    
    # Add arguments
    parser.add_argument('-t', '--topic', type=str, default=SEARCH_TOPIC,
                        help=f'Search topic (default: {SEARCH_TOPIC})')
    parser.add_argument('-c', '--category', type=str, default=SEARCH_CATEGORY,
                        help=f'arXiv category (default: {SEARCH_CATEGORY})')
    parser.add_argument('-n', '--num-papers', type=int, default=MAX_PAPERS_TO_DOWNLOAD, 
                        help=f'Maximum number of papers to download (default: {MAX_PAPERS_TO_DOWNLOAD})')
    parser.add_argument('-d', '--download-dir', type=str, default=DOWNLOAD_DIRECTORY,
                        help=f'Directory to save downloaded papers (default: {DOWNLOAD_DIRECTORY})')
    parser.add_argument('-s', '--sort-by', type=str, default=SORT_BY,
                        choices=['relevance', 'lastUpdatedDate', 'submittedDate'],
                        help=f'Sort results by (default: {SORT_BY})')
    parser.add_argument('-o', '--sort-order', type=str, default=SORT_ORDER,
                        choices=['ascending', 'descending'],
                        help=f'Sort order (default: {SORT_ORDER})')
    parser.add_argument('-i', '--interactive', action='store_true', 
                        help='Run in interactive mode (legacy)')
    parser.add_argument('--id', type=str, help='Download paper by arXiv ID')
    parser.add_argument('--list', action='store_true', 
                        help='List all previously downloaded papers')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle different modes
    if args.list:
        # Display previously downloaded papers
        display_downloaded_papers()
    elif args.id:
        # Download single paper by ID
        download_dir = create_download_dir(args.download_dir)
        paper = {'arxiv_id': args.id, 'title': args.id, 
                'pdf_url': f'http://arxiv.org/pdf/{args.id}'}
        download_paper(paper, download_dir)
    else:
        # Auto download by topic
        auto_download_by_topic(
            topic=args.topic,
            category=args.category,
            max_papers=args.num_papers,
            download_dir=args.download_dir,
            sort_by=args.sort_by,
            sort_order=args.sort_order
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}🛑 Operation cancelled by user{Colors.ENDC}")
        print(f"{Colors.CYAN}👋 Thanks for using Moon Dev's arXiv Downloader! Goodbye!{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ An error occurred: {str(e)}{Colors.ENDC}")
