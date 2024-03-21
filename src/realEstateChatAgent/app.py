import streamlit as st
from main import Agent
import pdfplumber
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import spacy
load_dotenv()

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

def main():
 
    st.set_page_config(page_title="Smart Resume Tracker", page_icon=":clipboard:", layout="wide")
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
            <h1 class="title">Smart Resume Tracker</h1>
            <p class="description">Upload your resume (PDF) and JD where you are applying</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    # Input text box and PDF upload block in the sidebar
    user_input = st.sidebar.text_input("Enter the Job description here:")
    uploaded_file = st.sidebar.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")
    submit = st.sidebar.button("Submit")


    # user_input = st.text_input("Enter the Job description here:")
    # uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")
    # submit = st.button("Submit")
    # Layout: Divide the UI into two columns
    col1, col2 = st.columns([2, 3])  # Adjust the column widths as needed
    if submit:
        with col1:  
            if uploaded_file is not None:
                text = input_pdf_text(uploaded_file)
                response = chatbot_response(user_input=user_input, resume_text=text)
                st.subheader(response)
        with col2:
            # Calculate and display JD match percentage using a bar plot
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

            # Plotting the bar plot for JD match percentage
            fig, ax = plt.subplots(figsize=(4, 2))  # Adjust the figure size here
            bars = ax.bar(["Matched", "Unmatched"], [match_percentage, unmatched_percentage], color=['#66c2a5', '#fc8d62'])
            ax.set_ylabel('Percentage')
            ax.set_title('JD Skill Percentage')
            ax.set_ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 2),
                            textcoords='offset points', ha='center', va='bottom')
            st.pyplot(fig)

            # Calculate document similarity
            similarity = doc_similarity_calculator(user_input=user_input,resume_text=text)

            # Plotting the bar plot for document similarity
            fig2, ax2 = plt.subplots(figsize=(4, 2))  # Adjust the figure size here
            bars2 = ax2.bar(["Skills Match", "Skills Unmatch"], [similarity, 100-similarity], color=['#66c2a5', '#fc8d62'])
            ax2.set_ylabel('Percentage')
            ax2.set_title('Document Similarity')
            ax2.set_ylim(0, 100)
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 2),
                            textcoords='offset points', ha='center', va='bottom')
            st.pyplot(fig2)



if __name__ == "__main__":
    main()
