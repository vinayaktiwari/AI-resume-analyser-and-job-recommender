import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_job_data(job_role):
    companies = []
    job_descriptions = []
    locations = []
    jobs_data = {}

    for page_num in range(1, 2):
        link = f'https://in.indeed.com/jobs/{job_role}-jobs?start={page_num}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(link, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract job data
            company_tags = soup.find_all('span', class_='css-92r8pb eu4oa1w0')
            per_page_companies = [company.text.strip() for company in company_tags]
            companies.extend(per_page_companies)

            jd_tags = soup.find_all('div', class_='css-9446fg eu4oa1w0')
            job_descriptions.extend([jd.text.strip() for jd in jd_tags])

            location_tags = soup.find_all('div', class_='css-1p0sjhy eu4oa1w0')
            locations.extend([location.text.strip() for location in location_tags])

            # job_url.extend([link])
            jobs_data[link] = companies
        else:
            print('Invalid Response')
    return companies, job_descriptions, locations, jobs_data,link

def create_data_dict(companies, job_descriptions, locations):
    data_dict = {}
    for idx, (company, job_description, location) in enumerate(zip(companies, job_descriptions, locations), start=1):
        data_dict[company] = [job_description, company, location]
    return data_dict



from sentence_transformers import SentenceTransformer, util

def create_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    return model

def retrieve_query2(data_dict, query_text,city,state, k=5):
    model = create_embeddings()
    user_input = [query_text,city,state]
    # Encode the query text
    query_emb = model.encode(user_input, convert_to_tensor=True)

    # Calculate similarity scores
    similarities = {}
    for idx, texts in data_dict.items():
        text_embs = model.encode(texts, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(query_emb, text_embs).numpy()
        avg_sim = cos_sim.mean()  # You can use other aggregation methods as well
        similarities[idx] = avg_sim

    print("similarity", similarities)
    # Sort similarities
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Get top-k similar text indices
    top_k_indices = [idx for idx, _ in sorted_similarities[:k]]
    
    return top_k_indices


    