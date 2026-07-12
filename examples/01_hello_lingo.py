import dotenv
from lingo import Lingo
from lingo.cli import loop

dotenv.load_dotenv()

bot = Lingo(
    name="PyTutor",
    description="A friendly Python tutor. Explain concepts clearly and use short code examples.",
)

if __name__ == "__main__":
    loop(bot)
