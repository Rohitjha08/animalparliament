import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import newspaper
import asyncio 
import pandas as pd 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# logging purpose
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# gemini llm use
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Any

class AnalyzedArticle(BaseModel):
    title: str
    is_animal_related: bool
    summary: str
    ministry: Optional[str] = None
    animal_agriculture_relevant: bool
    animal_context_summary: Optional[str] = None
    mentions_of_lobbying: List[str] = Field(default_factory=list)
    sentiment_towards_animal_welfare: str
    affect_summary: str
    date_detected: str
    source_url: str
    raw_text_snippet: str

class GeminiAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.prompt_template = """
You are an AI policy watchdog expert analyzing news articles for an organization focused on animal welfare in India.
Your task is to extract key information from the provided article content and return it as a structured JSON object.

**Article Content:**
\"\"\"
{content}
\"\"\"

**Instructions:**
Analyze the article content and return a single, valid JSON object with the following fields. Do not include any text outside the JSON object.
- "title": The official title of the news article.
- "is_animal_related": true if the article is related to animal welfare, wildlife, animal testing, livestock, dairy, the meat industry, or animal-related policy. Otherwise, false.
- "summary": A concise 3-4 line summary of the article's key points.
- "ministry": If a government body or ministry (e.g., "Ministry of Environment, Forest and Climate Change") is mentioned, state its full name. If not, use null.
- "animal_agriculture_relevant": true if the article involves sectors like livestock, poultry, dairy, fisheries, etc. Otherwise, false.
- "animal_context_summary": If the article is animal-related, summarize the specific portions that discuss animals, wildlife, laws, protection, cruelty, or testing. If not applicable, use null.
- "mentions_of_lobbying": A JSON array of strings listing any organizations, companies, or individuals explicitly mentioned as trying to influence or lobby for policy changes. If none, return an empty array [].
- "sentiment_towards_animal_welfare": Classify the article's sentiment towards animal welfare as "Positive", "Negative", or "Neutral".
- "affect_summary": Briefly explain how this news could potentially affect animal welfare policy, public perception, or industry practices.
- "date_detected": "{current_date}"
- "source_url": "{source_url}"
- "raw_text_snippet": "{raw_text_snippet}"
"""

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """More reliably extracts a JSON object from a string."""
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"JSON Decode Error after regex extraction: {e}")
                return None
        # for debugging
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logging.error("Could not parse JSON from the response.")
            return None

    async def analyze(self, title: str, content: str, source_url: str) -> Optional[AnalyzedArticle]:
        """Sends content to Gemini and validates the response with Pydantic."""
        prompt = self.prompt_template.format(
            content=content[:6000], #only 6000 text
            current_date=datetime.now().strftime("%Y-%m-%d"),
            source_url=source_url,
            raw_text_snippet=content[:200].replace('"', "'") + "..."
        )
        try:
            response = await self.model.generate_content_async(prompt)
            json_data = self._extract_json(response.text)

            if not json_data:
                return None

            article = AnalyzedArticle.model_validate(json_data)
            return article
        except ValidationError as e:
            logging.error(f"Pydantic Validation Error for {title}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during Gemini analysis for {title}: {e}")
            return None


SCRAPER_CONFIG = {
    "times_of_india": {
        "url": "https://timesofindia.indiatimes.com/topic/wildlife",
        "link_selector": "div.uwU81 a",
        "base_url": "https://timesofindia.indiatimes.com"
    },
    "indian_express": {
        "url": "https://indianexpress.com/about/animals/",
        "link_selector": "h2.title a",
        "base_url": "https://indianexpress.com"
    }
}

class HybridScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetches the HTML content of a given URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP Error fetching {url}: {e}")
            return None

    def get_article_links(self, config: Dict[str, str]) -> List[str]:
        """Fetches the main topic page and extracts article links using BeautifulSoup."""
        html_content = self._fetch_page(config["url"])
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        links = set()
        for a_tag in soup.select(config["link_selector"]):
            href = a_tag.get("href")
            if not href:
                continue

            if href.startswith("/"):
                full_link = config["base_url"] + href
            elif href.startswith("http"):
                full_link = href
            else:
                continue

            links.add(full_link)
        return list(links)

    def scrape_article_with_newspaper(self, url: str) -> Optional[Dict[str, str]]:
        
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()

            if not article.text or len(article.text) < 100:
                logging.warning(f"⚠️ Skipped {url} (newspaper3k couldn't extract enough content)")
                return None

            return {"title": article.title, "text": article.text, "url": url}
        except Exception as e:
            logging.error(f"Error scraping {url} with newspaper3k: {e}")
            return None


async def scrape_and_analyze_news(progress_callback=None):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    analyzer = GeminiAnalyzer(api_key=gemini_api_key)

    scraper = HybridScraper()

    all_links = []
    if progress_callback: progress_callback("Getting links...", 0)
    for source_name, config in SCRAPER_CONFIG.items():
        if progress_callback: progress_callback(f"🔍 Getting links from {source_name.replace('_', ' ').title()}...", 0)
        links = scraper.get_article_links(config)
        all_links.extend(links)

    unique_links = list(set(all_links))
    if progress_callback: progress_callback(f"Found {len(unique_links)} unique article links.", 0.1)

    scraped_articles = []
    num_to_process = min(20,len(unique_links)) # Limit for demo

    for i, link in enumerate(unique_links[:num_to_process]):
        if progress_callback:
            progress = 0.1 + (i / num_to_process) * 0.4 # 10% for links, 40% for scraping
            progress_callback(f"⚙️ Scraping article {i+1}/{num_to_process}: {link}", progress)
        
        article_data = scraper.scrape_article_with_newspaper(link)
        if article_data:
            scraped_articles.append(article_data)

    if progress_callback: progress_callback(f"📦 Successfully scraped content from {len(scraped_articles)} articles.", 0.5)

    final_data = []
    for i, article in enumerate(scraped_articles):
        if progress_callback:
            progress = 0.5 + (i / len(scraped_articles)) * 0.5 # 50% for scraping, 50% for analysis
            progress_callback(f"🧠 Analyzing article {i+1}/{len(scraped_articles)} with LLM: {article['title']}", progress)
        
        analyzed_result = await analyzer.analyze(
            title=article['title'],
            content=article['text'],
            source_url=article['url']
        )
        if analyzed_result:
            final_data.append(analyzed_result.model_dump())

    if progress_callback: progress_callback(f"✅ LLM successfully analyzed {len(final_data)} articles.", 1.0)
    return final_data


def send_custom_email(bill_title, summary, ministry, link, receiver_email, custom_text=""):
    sender_email = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")

    if not sender_email or not password:
        return "SMTP credentials not set in .env. Cannot send email."

    subject = f"🚨 New Bill Alert: {bill_title}"

    body = f"""
Hello,

A new bill has been identified that may impact animal welfare or agriculture.

🔖 Title: {bill_title}
🏛️ Ministry: {ministry if ministry else 'N/A'}

📄 Summary:
{summary}

🔗 Full Bill Details: {link}
"""

    if custom_text:
        body += f"\n📢 Custom Message:\n{custom_text}\n"

    body += "\nPlease review and consider advocacy or awareness actions.\n\nBest,\nBill Tracker System \nMade with love By rohit Jha"

    message = MIMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(message)
        return True
    except Exception as e:
        return str(e)



st.set_page_config(layout="wide", page_title="Animal Welfare News Analyzer")

st.title("🐾 Animal Welfare News Scraper & Analyzer")
st.markdown("---")

if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = []

if st.button("🚀 Start Scraping and Analyzing"):
    st.session_state.scraped_data = [] 

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    def update_progress(message, progress_val):
        status_text.info(message)
        progress_bar.progress(progress_val)

    with st.spinner("Initializing..."):
        try:
            
            st.session_state.scraped_data = asyncio.run(scrape_and_analyze_news(update_progress))
            status_text.success("✅ Analysis Complete!")
            progress_bar.empty() 

        except ValueError as ve:
            status_text.error(f"Configuration Error: {ve}")
            progress_bar.empty()
        except Exception as e:
            status_text.error(f"An error occurred: {e}")
            progress_bar.empty()

st.markdown("---")

if st.session_state.scraped_data:
    st.header("📊 Analysis Results")
    
    df = pd.DataFrame(st.session_state.scraped_data)
    
   
    st.subheader("Filter and Display Articles")
    filter_animal_related = st.checkbox("Show only animal-related articles", value=True)
    
    display_df = df
    if filter_animal_related:
        display_df = df[df['is_animal_related'] == True]

    if not display_df.empty:
        st.dataframe(display_df, use_container_width=True, height=300)

        st.markdown("---")
        st.subheader("Individual Article Details & Email Alert")

        
        article_titles = display_df['title'].tolist()
        selected_title = st.selectbox("Select an article to view details:", article_titles)

        if selected_title:
            selected_article = display_df[display_df['title'] == selected_title].iloc[0] 
            
            st.json(selected_article.to_dict()) 

            st.markdown("---")
            st.subheader("📧 Send Email Alert for this Article")
            receiver_email = st.text_input("Receiver Email:", "your_email@example.com")
            custom_message = st.text_area("Add a custom message to the email (optional):")

            if st.button("Send Email"):
                email_result = send_custom_email(
                    bill_title=selected_article['title'],
                    summary=selected_article['summary'],
                    ministry=selected_article['ministry'],
                    link=selected_article['source_url'],
                    receiver_email=receiver_email,
                    custom_text=custom_message
                )
                if email_result is True:
                    st.success("Email sent successfully! 📧")
                else:
                    st.error(f"Failed to send email: {email_result}")
    else:
        st.info("No articles to display based on current filters.")
else:
    st.info("Click 'Start Scraping and Analyzing' to fetch and process news articles.")

st.markdown("---")
st.caption("Developed for Animal Welfare Policy Watchdog.")
st.markdown("""
<div style="text-align:center; font-size: 13px; color: gray;">
Made with ❤️ using Streamlit<br>
 By Rohit and Om<br> 
Last Updated: {date}
</div>
""".format(date=datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)