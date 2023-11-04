import os
import openai
import pickle
import argparse

from langchain.vectorstores import FAISS as BaseFAISS

from dotenv import load_dotenv
from celery import Celery

from langchain.embeddings import OpenAIEmbeddings
from flask import Flask, request, jsonify

parser = argparse.ArgumentParser()
parser.add_argument("env", help="name of env", default="")
args = parser.parse_args()

load_dotenv('./.env.' + args.env)

app = Flask(__name__)
app.json.ensure_ascii = False
celery = Celery('api', broker=os.getenv('CELERY_BROKER_URL'))

OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
DOCUMENTATION_NAME = os.getenv('DOCUMENTATION_NAME')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT')
K_COUNT = int(os.getenv('K_COUNT'))
COUNT_FROM_SAME_SOURCE = os.getenv('COUNT_FROM_SAME_SOURCE')
if not COUNT_FROM_SAME_SOURCE:
    COUNT_FROM_SAME_SOURCE = K_COUNT
else:
    COUNT_FROM_SAME_SOURCE = int(COUNT_FROM_SAME_SOURCE)
ALL_COUNT = os.getenv('ALL_COUNT')
if not ALL_COUNT:
    ALL_COUNT = K_COUNT
else:
    ALL_COUNT = int(ALL_COUNT)

OPEN_AI_MODEL = os.getenv('OPEN_AI_MODEL')
if not OPEN_AI_MODEL:
    OPEN_AI_MODEL = 'gpt-4'

API_KEY = os.getenv('API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Store the last 10 conversations for each user
conversations = {}


class FAISS(BaseFAISS):
    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


# Load the FAISS index
faiss_obj_path = "models/" + MODEL_NAME + ".pickle"
faiss_index = FAISS.load(faiss_obj_path)


# @celery.task
def generate_response_chat(message_list):
    if faiss_index:
        # Add extra text to the content of the last message
        last_message = message_list[-1]

        # Get the most similar documents to the last message
        try:
            source_counts = dict()
            all_count = 0
            docs = faiss_index.similarity_search(query=last_message["content"], k=K_COUNT)

            updated_content = "Begin of " + DOCUMENTATION_NAME + "\n\n"
            for doc in docs:
                current_source = doc.metadata['source']
                source_counts[current_source] = source_counts.get(current_source, 0) + 1
                if source_counts[current_source] > COUNT_FROM_SAME_SOURCE:
                    continue
                if all_count > ALL_COUNT:
                    break
                all_count = all_count + 1
                updated_content += doc.page_content + "\n\n"
            updated_content += "End of " + DOCUMENTATION_NAME + "\n\nQuestion: " + last_message["content"]
        except Exception as e:
            print(f"Error while fetching : {e}")
            updated_content = last_message["content"]

        print(updated_content)

        # Create a new HumanMessage object with the updated content
        # updated_message = HumanMessage(content=updated_content)
        updated_message = {"role": "user", "content": updated_content}

        # Replace the last message in message_list with the updated message
        message_list[-1] = updated_message

    openai.api_key = OPENAI_API_KEY
    # Send request to GPT-3 (replace with actual GPT-3 API call)
    gpt3_response = openai.ChatCompletion.create(
        model=OPEN_AI_MODEL,
        temperature=0,
        messages=[
                     {"role": "system",
                      "content": SYSTEM_PROMPT},
                 ] + message_list
    )

    assistant_response = gpt3_response["choices"][0]["message"]["content"].strip().replace("Ответ: ", "")

    return assistant_response


def conversation_tracking(text_message, user_id):
    """
    Make remember all the conversation
    :param user_id: telegram user id
    :param text_message: text message
    :return: str
    """
    # Get the last 10 conversations and responses for this user
    user_conversations = conversations.get(user_id, {'conversations': [], 'responses': []})
    user_messages = user_conversations['conversations'][-9:] + [text_message]
    user_responses = user_conversations['responses'][-9:]

    # Store the updated conversations and responses for this user
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}

    # Construct the full conversation history in the user:assistant, " format
    conversation_history = []

    for i in range(min(len(user_messages), len(user_responses))):
        conversation_history.append({
            "role": "user", "content": user_messages[i]
        })
        conversation_history.append({
            "role": "assistant", "content": user_responses[i]
        })

    # Add last prompt
    conversation_history.append({
        "role": "user", "content": text_message
    })
    # Generate response
    response = generate_response_chat(conversation_history)
    # task = generate_response_chat.apply_async(args=[conversation_history])
    # response = task.get()

    # Add the response to the user's responses
    user_responses.append(response)

    # Store the updated conversations and responses for this user
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}

    return response


@app.route('/api/reply', methods=["POST"])
def api_reply():
    api_key = request.json.get('api_key', None)
    if api_key != API_KEY:
        return jsonify({'success': False, 'error': 'wrong api key'})

    embeddings_name = request.json.get('embeddings', None)
    user_id = request.json.get('user_id', None)
    question_id = request.json.get('question_id', None)
    question_text = request.json.get('question_text', None)

    if not user_id or not question_text:
        return jsonify({'success': False, 'error': 'wrong params'})

    if embeddings_name != MODEL_NAME:
        return jsonify({'success': False, 'error': 'wrong embeddings'})

    # Handle /clear command
    if question_text == '/clear':
        conversations[user_id] = {'conversations': [], 'responses': []}
        jsonify({'success': True, 'question_id': question_id, 'answer_text': "Conversations and responses cleared!"})
        return

    response = conversation_tracking(question_text, user_id)

    # Reply to message
    return ({'success': True, 'question_id': question_id, 'answer_text': response})


if __name__ == "__main__":
    print("Starting API...")
    print("API Started")
    print("Press Ctrl + C to stop API")
    app.run(host='0.0.0.0', debug=True)
