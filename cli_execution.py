import google.generativeai as genai
import subprocess
from API import api
api_key=api()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

def execute_cmd_from_prompt(user_prompt):
    """
    Convert a user prompt into a CMD command using Gemini, execute it, and return the command and result.
    """
    # Gemini prompt for command generation
    full_prompt = f"""
    You are a command-only assistant.
    You will convert the user's request into a valid Windows CMD command.
    Only return the exact command. Do not explain anything. Do not apologize. Do not say what cannot be done.
    User request: {user_prompt}
    """
    try:
        response = model.generate_content(full_prompt)
        cmd = response.text.strip()
        cmd = cmd.replace("```", "").replace("cmd", "").strip()

        # Run the command
        completed_process = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Return command and output
        result = completed_process.stdout + completed_process.stderr
        return f"🔧 Command: {cmd}\n{result.strip()}"

    except Exception as e:
        return f"❌ Error: {e}"
