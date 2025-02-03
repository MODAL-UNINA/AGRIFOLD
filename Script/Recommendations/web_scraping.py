import requests
import csv
from bs4 import BeautifulSoup


def extract_links(url):
    response = requests.get(url)

    html = BeautifulSoup(response.text, 'html.parser')

    sections = html.find_all('section', attrs={'data-level': '1', 'id': lambda x: x and x.startswith('ref')})

    extracted_links = []

    for section in sections:
        links = section.find_all('a', href=True)
        
        for link in links:
            name = link.text.strip()  
            href = link['href']      
            extracted_links.append((name, href))

    return extracted_links

def save_to_csv(data, file_name="links.csv"):
    with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Link Name", "URL"]) 
        csv_writer.writerows(data)
    print(f"Dati salvati in {file_name}")

# Main URL 
url = "https://www.example_url.com/"

links = extract_links(url)

for name, link in links:
    print(f"{name} : {link}")

save_to_csv(links, "links.csv")
