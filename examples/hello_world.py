from lingo import Lingo
import dotenv

dotenv.load_dotenv()

bot = Lingo()

bot.loop()
