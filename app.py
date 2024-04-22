import asyncio
import logging
import openai
import json
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.dispatcher.router import Router
import whisper

# Set up logging to display information in the console.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot token obtained from BotFather in Telegram.
TOKEN = 'API-Token'
bot = Bot(token=TOKEN)
router = Router()


model = whisper.load_model("tiny")

# Set your OpenAI API key here
openai.api_key = 'Open-api-key'

# Load company data from JSON
with open('filtered_keywords_data_crystal_tax.json', 'r') as file:
    company_data = json.load(file)

greeted_users = set()

# Define a message handler for the "/start" command.
@router.message(Command("start"))
async def start_message(message: Message):
    await message.answer("Start message")


# General message handler to differentiate between text and voice messages.
@router.message()
async def handle_message(message: Message):
    if message.voice:
        await handle_voice(message)
    elif message.text:
        await handle_text_message(message, message.text)


# Handle voice messages by converting them to text using Whisper.
async def handle_voice(message: Message):
    file_info = await bot.get_file(message.voice.file_id)
    file_path = await bot.download_file(file_info.file_path)
    with open("voice_message.ogg", "wb") as f:
        f.write(file_path.read())  # Save the file locally
    result = model.transcribe("voice_message.ogg")
    text = result['text']
    logger.info(f"Transcribed text from voice: {text}")
    await handle_text_message(message, text)


# Process text messages using GPT-3 based on company data.
async def handle_text_message(message: Message, text):
    user_question = text.lower()
    logger.info(f"Received message: {user_question}")

    prompt = f"Here is a customer question: {user_question}\n\n" \
             f"Provide a detailed answer based on the company information: site or json"
    answer = await fetch_gpt_response(prompt)

    await send_long_message(message.chat.id, answer)


async def fetch_gpt_response(prompt):
    try:
        # Определение роли и контекста для GPT-3.5 Turbo
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system",
                 "content": "Your promt here"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000  # Установка максимального количества токенов
        )

        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error generating GPT-3 response: {str(e)}")
        return "Sorry, I encountered an error while generating a response."


# Split long messages into parts if they exceed Telegram's maximum message length.
def split_message(text, size=4096):
    return [text[i:i+size] for i in range(0, len(text), size)]


# Send long messages part by part to avoid hitting Telegram's message length limit.
async def send_long_message(chat_id, text):
    parts = split_message(text)
    for part in parts:
        await bot.send_message(chat_id, part)


# Main function to start the bot.
async def main():
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
