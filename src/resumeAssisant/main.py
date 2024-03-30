import logging
from langchain.llms import GooglePalm
import logging
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from templates import ValidationTemplate,ResumeTrackerTemplate
from dotenv import load_dotenv
import PyPDF2 as pdf
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
from langchain.chains import LLMChain, SequentialChain

class Agent:
    def __init__(self,
                 llm_api_key,
                 pdf_text,
                 temperature =0.1,
                 debug=True):
        
        self.pdf_text =pdf_text

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Base LLM is Google Palm")
        self.chat_model = GooglePalm(
                temperature=temperature,
                google_api_key=llm_api_key,
            )
        
        self.palm_key = llm_api_key
        self.validation_prompt = ValidationTemplate()
        self.ATS_prompt = ResumeTrackerTemplate(extracted_text=self.pdf_text)

        self.validation_chain = self.set_up_validation_chain(debug)
        self.ATS_chain = self.set_up_resume_tracker_chain(debug)
    
    
    def set_up_validation_chain(self,debug= True):
        validation_agent = LLMChain(llm=self.chat_model,
                                    prompt=self.validation_prompt.chat_prompt,
                                    output_parser=self.validation_prompt.parser,
                                    output_key="validation_output",
                                    verbose = debug)
        
        # add to sequential chain 
        overall_chain = SequentialChain(
            chains=[validation_agent],
            input_variables=["query", "format_instructions"],
            output_variables=["validation_output"],
            verbose=debug,
        )
        return overall_chain
    

    def set_up_resume_tracker_chain(self,debug= True):
        resume_tracker = LLMChain(llm=self.chat_model,
                                    prompt=self.ATS_prompt.chat_prompt,
                                    output_key="ATS_response",
                                    verbose = debug)
        
        # add to sequential chain 
        overall_chain = SequentialChain(
            chains=[resume_tracker],
            input_variables=["job_description", "resume_text"],
            output_variables=["ATS_response"],
            verbose=debug,
        )
        return overall_chain
    



    
    def validate_query(self,query):
        self.logger.info("Validating query")
        t1 = time.time()
        self.logger.info("Calling validation (model is {}) on user input".format(
                self.chat_model.model_name
            )
        )
        validation_result = self.validation_chain(
            {
                "query": query,
                "format_instructions": self.validation_prompt.parser.get_format_instructions(),
            }
        )

        validation_test = validation_result["validation_output"].dict()
        t2 = time.time()
        self.logger.info("Time to validate request: {}".format(round(t2 - t1, 2)))
        self.logger.info(validation_test)
        return validation_test
        


    def ATS_query(self,JD):
        self.logger.info("ATS query")
        t1 = time.time()
        self.logger.info("Calling ATS (model is {}) on user input".format(
                self.chat_model.model_name
            )
        )
        ATS_result = self.ATS_chain(
            {
                "job_description": JD,
                "resume_text": self.ATS_prompt,
            }
        )

        ATS_test = ATS_result["ATS_response"]
        t2 = time.time()
        self.logger.info("Time to validate request: {}".format(round(t2 - t1, 2)))
        self.logger.info(ATS_test)
        return ATS_test
    
