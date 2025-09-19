import openai

def get_chat_completion(prompt_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # You can choose a different model like "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,  # Limit the response length
            temperature=0.7  # Control creativity (0.0-1.0)
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main():
    openai.api_key = "pZYfEYo6VrzYBhR0I9mHMThDkpa27yfnpV7NXUzNh2e3fbsH1O-_sYja9PWzgF2NKQIBWdEMtvT3BlbkFJFlhpJFLbjGJoix5QdQ3yZcW0GbYPQdK1CuL-toixbQdN_3zCWNtO7RWN7gpJBy_0lIHYD3QycA"
    if not openai.api_key:
        print("API key not provided. Exiting.")
        return

    question = "What is the capital city of Bangladesh?"
    print(f"Asking OpenAI: '{question}'")

    answer = get_chat_completion(question)

    if answer:
        print("\nOpenAI's Answer:")
        print(answer)
    else:
        print("Could not get an answer from OpenAI.")

if __name__ == "__main__":
    main()