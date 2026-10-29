import json
import requests
import os
import time
import re

def sanitize_filename(title):
    """Make title filesystem-safe and truncate long names."""
    return re.sub(r'[\\/*?:"<>|]', "_", title)[:150]

def download_pdf(paper_id, title, output_dir):
    """
    Download PDF using OpenReview paper_id and save as title.pdf.
    Returns: (success_flag, pdf_url, local_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_title = sanitize_filename(title)
    pdf_url = f"https://openreview.net/pdf?id={paper_id}"
    local_path = os.path.join(output_dir, f"{safe_title}.pdf")

    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(pdf_url, headers=headers, stream=True)
        response.raise_for_status()

        if 'application/pdf' not in response.headers.get('content-type', ''):
            print(f"⚠️ {paper_id}: Not a PDF!")
            return False, pdf_url, local_path

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        print(f"✅ Downloaded: {safe_title}")
        return True, pdf_url, local_path

    except Exception as e:
        print(f"❌ Error downloading {paper_id}: {e}")
        return False, pdf_url, local_path

def main():
    json_path = "iclr2025.json"
    output_dir = "iclr_2025_pdfs"
    downloaded, failed = [], []

    with open(json_path, 'r') as f:
        papers = json.load(f)

    for paper in papers:
        paper_id = paper['id']
        title = paper.get('title', f"paper_{paper_id}")

        if paper['status'] not in ['Spotlight', 'Poster', 'Oral']:
            continue
        
        success, link, local_path = download_pdf(paper_id, title, output_dir)
        paper_info = {
            "id": paper_id,
            "title": title,
            "link": link,
            "local_pdf_path": local_path if success else None
        }

        if success:
            downloaded.append(paper_info)
        else:
            failed.append(paper_info)

        time.sleep(1)  # be nice to the server

    # Save logs
    with open("downloaded_pdfs.json", "w") as f:
        json.dump(downloaded, f, indent=2)

    with open("failed_pdfs.json", "w") as f:
        json.dump(failed, f, indent=2)

    print(f"\n✅ Downloaded: {len(downloaded)}")
    print(f"❌ Failed: {len(failed)}")

if __name__ == "__main__":
    main()
