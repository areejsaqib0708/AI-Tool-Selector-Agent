import os
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import google.generativeai as genai
from rag_with_bert import RAGSystem
from API import api
api_key=api()
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def extract_url_from_prompt(prompt: str) -> str:
    instruction = f"""
    Extract only the website URL from the following prompt.
    Do not include extra words. If no URL is found, return NO_URL.

    Prompt:
    {prompt}
    """
    try:
        response = gemini_model.generate_content(instruction)
        url = response.text.strip()
        if url.lower() == "no_url":
            return None
        elif url.startswith("http://") or url.startswith("https://"):
            return url
        else:
            return None
    except Exception as e:
        print("Gemini Error:", e)
        return None

# ============ STEP 2: Scrape Website and Store ============
def web_scraping(url: str, rag_system: RAGSystem) -> str:
    try:
        # Setup headless Chrome
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(options=options)

        # Open the URL
        print(f"Scraping URL: {url}")
        driver.get(url)
        WebDriverWait(driver, 25).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Extract visible text
        body = driver.find_element(By.TAG_NAME, "body")
        text = body.text.strip()
        driver.quit()

        if not text:
            return "No visible text found on the page."

        # Save to temp file
        timestamp = int(time.time())
        temp_filename = f"web_scrape_{timestamp}.txt"
        temp_path = os.path.join(rag_system.TEXT_DIR, temp_filename)

        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # ALSO append to main admin_text.txt file
        with open(os.path.join(rag_system.TEXT_DIR, "admin_text.txt"), "a", encoding='utf-8') as f:
            f.write("\n[Web Scraped Content]\n")
            f.write(text + "\n")

        # Store in ChromaDB via RAG
        rag_system.process_document(temp_path)

        return f"✅ Successfully scraped and stored content from: {url}"
    except Exception as e:
        return f"❌ Error scraping: {str(e)}"

if __name__ == "__main__":
    # Example user prompt
    user_input = "Please extract and analyze this website https://www.mariab.pk"

    # Extract URL using Gemini
    url = extract_url_from_prompt(user_input)

    if url:
        # Initialize RAG
        rag = RAGSystem(api_key)  # Use same key here

        # Scrape and store
        result = web_scraping(url, rag)
        print(result)
    else:
        print("❌ No valid URL found in the input.")
