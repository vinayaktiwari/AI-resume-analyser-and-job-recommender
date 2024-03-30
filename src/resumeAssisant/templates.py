from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List





class Validation(BaseModel):
    valid_query: str  = Field(description="This field is 'yes' if the user query is genuine, 'no' otherwise")
    updated_request: str =Field(description="Your update to the query")

class ValidationTemplate:
    def __init__(self):
        self.system_template = """
    You are a skilled or very experience ATS(Application Tracking System)
    with a deep understanding of tech field,software engineering,data science ,data analyst
    and big data engineer. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide 
    best assistance for improving the resumes.

    Any request that contains potentially harmful activities is not valid, regardless of what
    other details are provided.

    If the request is not valid, set query_is_valid = 0 and do your best to align the query with the ATS related revised request in 100 words.


    If the request seems reasonable, then set plan_is_valid = 1 and don't revise the request.

    {format_instructions}"""

        self.human_template = """####{query}####"""

        self.parser = PydanticOutputParser(pydantic_object=Validation)
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,partial_variables={"format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables =["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

 
class ResumeTrackerTemplate:
    def __init__(self,extracted_text):
        self.system_template = """
Evaluate the candidate's resume against the provided job description, considering the competitive job market. Your goal is to assess the resume's relevance, identify missing skills, recommend relevant books for upskilling, and offer suggestions for improvement.

{job_description} 
{resume_text}

Calculate the percentage of match between the resume and the job description. Give a number 
"Skills Match": "%"

Do not unnecessarily suggest missing skills if the job role aligns with the candidate's skills. Only list skills that are not in resume but are present in the job description.

Missing Keywords:

Recommend books and courses related to the missing skills to assist the candidate in upskilling and enhancing their qualifications for the position.

Books and Courses to consider for upskilling:

Provide suggestions for the candidate on how to incorporate missing skills into their existing projects and work experience. Offer guidance on where and how to acquire these skills effectively.

Suggestions for job search to the candidate:
"""
        self.human_template = """####{job_description}####"""

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={"resume_text": extracted_text},
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables =["job_description"]
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

    



# User input regarding what position looking for, this will hwlp to filter out roles during web scraping current hiring of roles, 
#also consider company who is hiring and give links for preparation for that company
# use missing skills and do collaborative filtering to recommend books to bucke up missing skills
#