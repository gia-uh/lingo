from pydantic import BaseModel
from lingo import Chatbot
import dotenv

from lingo.context import Context

dotenv.load_dotenv()


bot = Chatbot(
    "Lingo",
    "A friendly chatbot.",
)


bot.loop()
