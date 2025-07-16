import socket
import dns.resolver
import google.generativeai as genai
from API import api
api_key=api()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

def dns_lookup(prompt_text):
    try:
        extract_prompt = f"""
        Extract only the domain (like google.com) from the following sentence:
        "{prompt_text}"
        Respond with only the domain, nothing else.
        """
        response = model.generate_content(extract_prompt)
        website = response.text.strip()
    except Exception as e:
        return f"Error extracting domain: {e}"

    result = ""
    try:
        ip_address = socket.gethostbyname(website)
        result += f"IP Address of {website}: {ip_address}\n\n"
        result += "DNS Records:\n"
        for record_type in ['A', 'MX', 'NS', 'CNAME']:
            try:
                answers = dns.resolver.resolve(website, record_type)
                result += f"{record_type} Records:\n"
                for rdata in answers:
                    result += f" - {rdata.to_text()}\n"
            except dns.resolver.NoAnswer:
                result += f"{record_type} Records: No Answer\n"
            except dns.resolver.NXDOMAIN:
                result += f"{record_type} Records: Domain does not exist\n"
            except Exception as e:
                result += f"{record_type} Records: Error - {e}\n"

    except socket.gaierror:
        result += "Invalid website or unable to resolve host.\n"
    except Exception as e:
        result += f"Error occurred: {e}\n"

    return result
