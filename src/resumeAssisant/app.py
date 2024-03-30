import streamlit as st
from main import Agent
import pdfplumber
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import spacy
load_dotenv()
import plotly.graph_objs as go
from sentence_transformers import SentenceTransformer, util
from data_ingestion import fetch_job_data,create_data_dict,retrieve_query2


def create_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    return model

def retrieve_query(jd_text,resume_extracted_text):
    model = create_embeddings()

    # Encode the query text
    query_emb = model.encode(jd_text, convert_to_tensor=True)

    # Calculate similarity scores
    similarities = {}
    text_embs = model.encode(resume_extracted_text, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_emb, text_embs).numpy()

        # avg_sim = cos_sim.mean()  # You can use other aggregation methods as well
        # similarities[idx] = avg_sim
    print("similarity", cos_sim)

    # # Get top-k similar text indices
    # top_k_indices = [idx for idx, _ in sorted_similarities[:k]]
    return cos_sim[0][0] *100




# Function to plot pie chart
def plot_pie_chart(labels, values, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(marker=dict(colors=['#66c2a5', '#fc8d62']), textinfo='percent+label')
    fig.update_layout(title=title)
    st.plotly_chart(fig)

# Function to interact with the chatbot
def doc_similarity_calculator(user_input, resume_text):
    nlp = spacy.load("en_core_web_md")
    doc1_text = user_input
    doc2_text = resume_text
    # Preprocess the documents
    doc1 = nlp(doc1_text)
    doc2 = nlp(doc2_text)
    # Calculate similarity
    similarity_score = doc1.similarity(doc2)

    print("Similarity score:", similarity_score)
    return similarity_score*100

def chatbot_response(user_input, resume_text):
    ats_agent = Agent(llm_api_key=os.environ["GOOGLE_PALM_API_KEY"], pdf_text=resume_text, debug=True)
    response = ats_agent.ATS_query(JD=user_input)
    return response

def input_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to count words in the text
def count_words(text):
    words = text.split()
    return len(words)

def main():
 
    st.set_page_config(page_title=" AI Adviser for Resume and Jobs", page_icon=":clipboard:", layout="wide")
          # Animated logo

    st.markdown(
    """
    <style>
        .header-container {
            display: flex;
            align-items: center;
            background-color: #222222; /* Dark background color */
            padding: 20px;
            border-radius: 10px;
        }
        .logo-container {
            margin-right: 20px;
        }
        .logo-container img {
            width: 200px; /* Adjust logo width */
            height: auto; /* Maintain aspect ratio */
        }
        .text-container {
            color: white; /* Text color */
        }
        .title {
            margin: 0;
            font-size: 32px; /* Title font size */
            font-weight: bold;
            color: white; /* Title text color */
        }
        .description {
            margin: 0;
            font-size: 20px; /* Description font size */
            color: white; /* Description text color */
        }
    </style>
    <div class="header-container">
        <div class="logo-container">
            <img src="https://media.licdn.com/dms/image/D4D12AQEgw0xFCvyrew/article-cover_image-shrink_600_2000/0/1700803516393?e=2147483647&v=beta&t=7P2eobWKz9qcvR4_V52cG3pY0jMAmnumlFscBlFb8nU" alt="Logo">
        </div>
        <div class="text-container">
            <h1 class="title">Paul- An AI resume analyzer and job search adviser</h1>
            <ul class="description">
                <li>Upload your resume (PDF) and JD where you are applying.</li>
                <li>Paul will use its AI capabilities to analyze your resume and suggests keywords to add for better resume.
                <li>It also gives suggestions for improving the missing skills and recommend courses and books.</li>
                <li>It also recommends you the top 5 jobs after analyzing your resume.</li>
                </ul>
        </div>
    </div>
    <style>
        .sidebar .sidebar-content {
            background-color: #2a2a2a; /* Dark background color */
            padding: 20px;
            border-radius: 10px;
            color: white; /* Text color */
        }
    </style>
    """,
    unsafe_allow_html=True
)
    st.sidebar.title("🤖 Hi !! I am Paul, Your AI assistant")
    role = st.sidebar.text_input("Enter the desired job role: ")
    # Text input for job description
    user_input = st.sidebar.text_area("Enter the Job Description Here:", height=300)

    offices_in_cities = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata",
        "Ahmedabad", "Gurgaon", "Noida", "Jaipur", "Chandigarh", "Lucknow", "Indore",
        "Coimbatore", "Kochi", "Surat", "Vadodara", "Nagpur", "Visakhapatnam",
        "Bhopal", "Patna", "Ludhiana", "Agra", "Nashik", "Madurai", "Kanpur", "Raipur",
        "Guwahati", "Mysore", "Bhubaneswar", "Mangalore", "Vijayawada", "Jodhpur",
        "Amritsar", "Ranchi", "Gwalior", "Jammu", "Jalandhar", "Kota", "Ajmer", "Srinagar",
        "Shimla", "Udaipur", "Allahabad", "Meerut", "Asansol", "Durgapur", "Dhanbad"
    ]
        # List of states in India
    states_in_india = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
        "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
        "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
        "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
        "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu", "Lakshadweep", "Delhi",
        "Puducherry"
    ]
    job_role = role.replace(" ","%20")
    selected_city = st.selectbox("Select a city:", offices_in_cities)
    selected_state = st.selectbox("Select a state:", states_in_india)

    # File uploader for resume
    uploaded_file = st.sidebar.file_uploader("Upload Your Resume", type="pdf", help="Please upload the PDF")

    submit = st.sidebar.button("Submit", key="submit_button", help="Submit your job application")
    col1, col2 = st.columns(2)  # Adjust the column widths as needed

    if submit==True and count_words(user_input)<50:
        st.warning("Please enter a Job Description of at least 50 words for accurate analysis.")
    
    elif submit==True and count_words(user_input)>50:
        with col1:  
            if uploaded_file is not None:
                text = input_pdf_text(uploaded_file)
                response = chatbot_response(user_input=user_input, resume_text=text)
                st.success("Analysis Completed !!!!.")
                # Define the color of the card (you can use any color code or name)
                card_color =  "fff5e6" 

                # Render the response in a colored card
                st.markdown(
                    f'<div style="background-color:: #fff5e6; padding:20px; border-radius:10px;">'
                    f'<p style="color:#fff5e6; font-size:16px;">{response}</p>'
                    f'</div>',
                    unsafe_allow_html=True)
                
            # Calculate and display JD match percentage using a bar plot
            try:
                response_lines = response.split("\n")
                match_percentage_str = None
                for line in response_lines:
                    if "Skills Match" in line:
                        line = line.replace("**", "")
                        match_percentage_str = line.split(":")[-1].strip().replace("%", "")
                        break

                if match_percentage_str == '':
                    match_percentage = 0
                else:
                    match_percentage = float(match_percentage_str) 
                unmatched_percentage = 100 - match_percentage
                
                # Calculate document similarity
                # similarity = doc_similarity_calculator(user_input=user_input,resume_text=text)
                similarity= retrieve_query(jd_text=user_input,resume_extracted_text=text)

                # Plot pie chart for JD skill percentage
                plot_pie_chart(["Matched", "Unmatched"], [match_percentage, unmatched_percentage], "JD Skill Percentage")

                # Plot pie chart for document similarity
                plot_pie_chart(["Skills Match", "Skills Unmatch"], [similarity, 100 - similarity], "Document Similarity")
            except Exception as e:
                print("An error occurred:", e)


        with col2:
            companies, job_descriptions, locations, _ ,link= fetch_job_data(job_role)
            data_dict = create_data_dict(companies, job_descriptions,locations=locations)
            # top_recom_jobs = retrieve_query2(data_dict=data_dict,query_text=user_input,city=selected_city,state=selected_state)
            top_recom_jobs = retrieve_query2(data_dict=data_dict,query_text=text,city=selected_city,state=selected_state)

            # Define a custom CSS style for the job information box
            custom_css = """
            <style>
            .job-info-box {
                background-color: #fff5e6; /* light cream color */
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            </style>
            """

            # Display the custom CSS style
            st.markdown(custom_css, unsafe_allow_html=True)


            custom= """
                custom-header-3 {
                font-family: 'Georgia', serif; /* Use any desired font */
                font-size: 22px; /* Adjust the font size */
                color: #e83e8c; /* Adjust the font color */
                text-transform: capitalize; /* Convert text to capitalize */
            }
            """

            # Display the custom CSS
            st.success("Job recommendations fetched !!!!.")

            # Render the header with the custom styling
            st.markdown("<h1 class='custom-header'>Roles that match your resume !!!</h1>", unsafe_allow_html=True)


            # Assuming top_recom_jobs is a list of tuples where each tuple contains job information and link
            for idx in top_recom_jobs:
                job_info = data_dict[idx]
                st.subheader(idx)
                # Display job information in a styled box

                with st.container():
                    st.subheader(f"summary")
                    st.markdown(f"<div class='job-info-box'>{job_info[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='job-info-box'>{job_info[2]}</div>", unsafe_allow_html=True)
                    button_color = "#4CAF50"  # Green color for example

                    st.markdown(
                    f'<a href="{link}" target="_blank" style="background-color:{button_color};color:#ffffff;text-align:center;padding:10px;border-radius:5px;text-decoration:none;display:inline-block;">Apply</a>',
                    unsafe_allow_html=True)
                                

if __name__ == "__main__":
    main()


