from lingo import Chatbot
import dotenv

dotenv.load_dotenv()

bot = Chatbot(
    "Lingo",
    "A friendly chatbot.",
)

bot.loop()
