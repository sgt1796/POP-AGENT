from perplexity import Perplexity
from dotenv import load_dotenv

load_dotenv()
client = Perplexity()

search = client.search.create(
    query="stock market today",
    max_results=5,
    max_tokens_per_page=4096
)

for result in search.results:
    print(result)
    print(f"{result.title}: {result.url}")