import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import chromadb
import uuid


# Initialize AI Model
llm = ChatGroq(
    temperature=0, 
    groq_api_key='Your api key',  
    model_name="llama-3.3-70b-versatile"
)

# Streamlit UI
st.set_page_config(page_title="AI-Powered Job Finder", page_icon="📧")
st.title("📧AI-Powered Job Finder & Cold Email Generator")

# Initialize session state variables
if "job_suggestions" not in st.session_state:
    st.session_state.job_suggestions = []
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "generated_email" not in st.session_state:
    st.session_state.generated_email = None

user_skills = st.text_area("Enter your skills (comma-separated):")
job_posting = st.text_area("Enter the url of relevant job posting")

# Button to find jobs
if st.button("Find Jobs"):
    if user_skills:
        skills_list = [skill.strip() for skill in user_skills.split(",") if skill.strip()]

        # Scrape job postings
        loader = WebBaseLoader(job_posting)
        page_data = loader.load().pop().page_content

        # Extract job details using AI
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            Extract job postings and return them in JSON format with `role`, `experience`, `skills`, and `description`.
            Only return valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
        )

        chain_extract = prompt_extract | llm 
        res = chain_extract.invoke(input={'page_data': page_data})
        json_parser = JsonOutputParser()
        job_data = json_parser.parse(res.content)

        # Initialize ChromaDB
        client = chromadb.PersistentClient('vectorstore')
        collection = client.get_or_create_collection(name="jobs")

        # Store job data in ChromaDB
        for job in job_data:
            collection.add(
                documents=str(job),
                metadatas={"role": job.get("role", ""), "description": job.get("description", "")},
                ids=[str(uuid.uuid4())]
            )

        # Find jobs based on user skills
        job_suggestions = collection.query(query_texts=skills_list, n_results=3).get('metadatas', [])
        
        # Flatten nested lists and store in session state
        st.session_state.job_suggestions = [job for sublist in job_suggestions for job in sublist]

# Display job suggestions
if st.session_state.job_suggestions:
    st.subheader("✅ Matching Job Opportunities:")
    
    for idx, job in enumerate(st.session_state.job_suggestions):
        if isinstance(job, dict):  # Ensure it's a dictionary
            job_title = job.get("role", "Unknown Role")
            job_description = job.get("description", "No description available")

            # Display job details
            st.markdown(f"**📌 {job_title}**")  
            st.write(f"📝 {job_description}")
            st.write("---")

            # Button to select a job for email generation
            if st.button(f"Generate Email for {job_title}", key=f"email_{idx}"):
                st.session_state.selected_job = job
                st.session_state.generated_email = None  # Reset email content

# Generate cold email
if st.session_state.selected_job:
    st.subheader("📩 Generating Cold Email...")

    # AI prompt for email generation
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Faiz, a business development executive at Techify.Techify is an AI & Software Consulting company dedicated to 
        integrating business processes through automated tools. Over time, AtliQ has empowered enterprises with tailored solutions 
        that improve scalability, process optimization, cost reduction, and overall efficiency. 

        Write a cold email to the client regarding the job mentioned above, showcasing Techify's expertise in fulfilling their needs.

        You are Faiz, BDE at Techify.
        Do not include a preamble.

        ### EMAIL (NO PREAMBLE):
        """
    )

    if st.button("Generate Cold Email"):
        chain_email = prompt_email | llm
        email_response = chain_email.invoke({"job_description": st.session_state.selected_job["description"]})
        st.session_state.generated_email = email_response.content

# Display the generated email
if st.session_state.generated_email:
    st.subheader("📩 Generated Cold Email:")
    st.write(st.session_state.generated_email)
