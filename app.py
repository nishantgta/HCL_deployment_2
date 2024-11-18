from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from groq import AsyncGroq
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained BERT tokenizer and model
model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = BertTokenizer.from_pretrained(model_name) #BERT uses wordpiece tokenizer
#BERT is trained on two models, Masked Language Modelling(MLM) and Next Sentence Prediction(NSP)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

gorq_key="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"

# Define the request body using a Pydantic model
class RetentionRequest(BaseModel):
    feedback: str

@app.post("/generate-retention-strategy/")
async def retention_strategy(request: RetentionRequest):

    # Extract customer feedback
    feedback = request.feedback

    print("Feedback=",feedback)

    #Sentiment Analysis
    sentiment_result = sentiment_analyzer(feedback)
    sentiment = sentiment_result[0]['label']

    # Generate personalized retention strategy using GPT (GORQ)
    customer_context = (
        f"Customer Retention Plan:\n"
        f"- Feedback: {feedback}\n"
        "Write a professional email to the customer addressing their concerns and offering a retention plan. The mail should be brief and concise"
    )

    client = AsyncGroq(api_key="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")
    completion =await client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "system",
            "content": "You are an expert customer retention specialist. The mail should be concise and easy to understand"
        },
        {
            "role": "user",
            "content": customer_context
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
    )

    response_content = ""
    async for chunk in completion:
        #print(chunk.choices[0].delta.content, end="")
        if chunk.choices[0].delta.content:
            response_content+=chunk.choices[0].delta.content+" "

    # Return the response
    return {
        "sentiment": sentiment,
        "retention_strategy": response_content.strip()
    }

