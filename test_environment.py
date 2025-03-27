import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv(override=True)

def test_openai():
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "OpenAI API test failed: No API key found in environment variables"
            
        print(f"API Key loaded: {'Yes' if api_key else 'No'}")
        print(f"API Key length: {len(api_key) if api_key else 0}")
        
        client = OpenAI()  # It will automatically use the environment variable
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello! This is a test message."}]
        )
        return "OpenAI API test successful! Response: " + response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API test failed: {str(e)}"

def test_gradio():
    try:
        demo = gr.Interface(
            fn=lambda x: f"Gradio test successful! You typed: {x}",
            inputs="text",
            outputs="text"
        )
        return "Gradio test successful!"
    except Exception as e:
        return f"Gradio test failed: {str(e)}"

if __name__ == "__main__":
    print("Testing environment setup...")
    print("\nTesting OpenAI API:")
    print(test_openai())
    print("\nTesting Gradio:")
    print(test_gradio()) 