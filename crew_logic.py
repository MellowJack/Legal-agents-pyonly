from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool  # Updated import for tool decorator
import os
from dotenv import load_dotenv
import requests
import time
import re
from typing import List, Dict
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import random
from pathlib import Path
import tempfile
import base64
import mimetypes
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Initialize LLM
llm = LLM(
    model="groq/llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

class IndianKanoonAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.indiankanoon.org"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json"
        })

    def search_cases(self, query: str, max_pages: int = 1) -> List[Dict]:
        q = quote_plus(query.encode("utf8"))
        url = f"{self.base_url}/search/?formInput={q}&pagenum=0&maxpages={max_pages}"
        res = self.session.post(url)
        res.raise_for_status()
        data = res.json()
        cases = []
        for doc in data.get("docs", []):
            cases.append({
                "doc_id": doc["tid"],
                "title": doc["title"],
                "court": doc.get("docsource", "Unknown"),
                "year": doc.get("publishdate", "Unknown")[:4],
                "snippet": doc.get("headline", "")[:200] + "...",
            })
        return cases

    def fetch_document(self, doc_id: int) -> Dict:
        url = f"{self.base_url}/doc/{doc_id}/"
        res = self.session.post(url)
        res.raise_for_status()
        return res.json()

    def fetch_original_document(self, doc_id: int) -> Path | None:
        url = f"{self.base_url}/origdoc/{doc_id}/"
        res = self.session.post(url)
        res.raise_for_status()
        data = res.json()
        if "doc" not in data:
            return None
        decoded = base64.b64decode(data["doc"])
        content_type = data.get("Content-Type", "application/octet-stream")
        suffix = mimetypes.guess_extension(content_type) or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(decoded)
            return Path(f.name)

    def summarize_file(self, path: Path) -> str:
        if path.suffix == ".pdf":
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        else:
            text = path.read_text(errors="ignore")
        summary_prompt = f"Summarize the following legal document:\n\n{text[:4000]}"
        summary = llm(summary_prompt)
        path.unlink(missing_ok=True)
        return summary
    
ik_client = IndianKanoonAPIClient(api_key=os.getenv("INDIANKANOON_API_KEY"))

# Updated tool definitions using @tool decorator
@tool
def search_cases(query: str) -> str:
    """Search Indian Kanoon for relevant legal cases given a query string"""
    results = ik_client.search_cases(query)
    if not results:
        return "No results found."
    return "\n".join(
        f"{r['title']} | {r['court']} | {r['year']} | id={r['doc_id']}" 
        for r in results
    )

@tool
def fetch_document(doc_id: str) -> str:
    """Fetch full document content by doc ID from Indian Kanoon"""
    doc = ik_client.fetch_document(int(doc_id))
    text = doc.get("text", "")
    return text[:4000] if text else "No text content found."

@tool
def summarize_original(doc_id: str) -> str:
    """Download and summarize original document (PDF/TXT) by doc ID"""
    path = ik_client.fetch_original_document(int(doc_id))
    if not path:
        return "Original document not found."
    return ik_client.summarize_file(path)

# Simplified agents
query_analyst = Agent(
    role="Legal Query Analyst",
    goal="Extract key legal terms and create search queries",
    backstory="Expert in analyzing legal queries and creating effective search terms",
    verbose=True,
    llm=llm,
)

case_researcher = Agent(
    role="Case Researcher",
    goal="Search and find relevant legal cases",
    backstory="Specialist in finding relevant cases using Indian Kanoon API",
    verbose=True,
    llm=llm,
    tools=[search_cases, fetch_document, summarize_original]
)

legal_analyst = Agent(
    role="Legal Analyst",
    goal="Analyze cases and provide legal insights",
    backstory="Senior legal expert providing detailed case analysis",
    verbose=True,
    llm=llm,
    tools=[fetch_document, summarize_original]
)

def create_tasks(user_query: str):
    """Create simplified task sequence"""
    
    analyze_task = Task(
        description=f"""
        Analyze this legal query: "{user_query}"
        
        Extract:
        1. Key legal terms
        2. Areas of law involved
        3. Relevant search keywords
        
        Create 2-3 search term variations for better results.
        """,
        expected_output="Key terms and search variations",
        agent=query_analyst
    )
    
    search_task = Task(
        description=f"""
        Use the search_cases tool to find cases related to: "{user_query}"

        Then, for top cases, use fetch_document or summarize_original if needed.

        Focus on cases highly relevant to the query.
        """,
        expected_output="List of relevant cases with basic details",
        agent=case_researcher,
        context=[analyze_task]
    )
    
    analysis_task = Task(
        description=f"""
        Provide legal analysis for the found cases.
        
        For each case provide:
        1. Case title and court
        2. Year and document ID
        3. Relevance to query
        4. Key legal points
        5. Case summary
        
        Focus on cases most relevant to: "{user_query}"
        """,
        expected_output="Detailed legal analysis of relevant cases",
        agent=legal_analyst,
        context=[analyze_task, search_task]
    )
    
    return [analyze_task, search_task, analysis_task]

class SimplifiedLegalResearchCrew:
    def kickoff(self, inputs: dict) -> str:
        """Execute the legal research crew"""
        user_query = inputs.get("argument", "")
        
        if not user_query.strip():
            return "Please provide a legal query to search for cases."
        
        try:
            tasks = create_tasks(user_query)
            
            crew = Crew(
                agents=[query_analyst, case_researcher, legal_analyst],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            print(f"🚀 Starting legal research for: {user_query}")
            result = crew.kickoff()
            
            return str(result)
            
        except Exception as e:
            return f"Error during legal research: {str(e)}"

# Create the crew instance
crew = SimplifiedLegalResearchCrew()

if __name__ == "__main__":
    print("✅ Simplified Legal Research Crew loaded!")
    print("📋 Ready to search Indian Kanoon for legal cases")