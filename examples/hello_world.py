from lingo import Lingo
from lingo.cli import loop

import dotenv

# Load MODEL, BASE_URL, and API_KEY from .env file
dotenv.load_dotenv()

# Instantiate a raw Lingo instance
# which comes preconfigured with basic chat functionality
bot = Lingo()

# Run the bot in a CLI loop


def main():
    loop(bot)


if __name__ == "__main__":
    main()
