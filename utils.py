import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm


def scrape_speeches(base_url: str, dataset_dir: str, usr_agent):
    """
    currently it supports scraping important speeches only
    :param base_url:
    :param dataset_dir:
    :param usr_agent: use navigator.userAgent in the console to get the usr_agent
    :return:
    """
    url = f"{base_url}top100speechesall.html"
    headers = {
        "User-Agent": usr_agent}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    pdf_urls = [base_url + tag.parent.parent.attrs["href"] for tag in soup.find_all("u", text="PDF")]

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    print("Starting to download")
    for i, pdf_url in enumerate(tqdm(pdf_urls)):
        response = requests.get(pdf_url, headers=headers, allow_redirects=True)
        pdf = open(f"{dataset_dir}/{i}.pdf", 'wb')
        pdf.write(response.content)
        pdf.close()
    print("Download finished successfully!")
