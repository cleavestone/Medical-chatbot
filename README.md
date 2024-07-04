# End-to-end Medical Chatbot

## Steps

**create virtual environment**
```bash
virtualenv mchaatbot
```
**activate virtual environment**
```bash
source mchatbot/scripts/activate
```
**install requirements.txt**
```bash
pip install -r requirements.txt
```
**Create  a .env file in the root directory and store your Pinecone API key and openai key**
```ini
PINECONE_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXX"
OPENAI_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXX"
```

```bash
python store_index.py
```

```bash
python app.py
```
```bash
open localhost
```
### Tech stack
- Python
- Langchain
- OpenAI
- Pinecone


