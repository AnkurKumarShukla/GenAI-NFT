from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
from typing import Optional
import aiohttp
from typing import List
from openai import OpenAI, APIError

# load the .env file
from dotenv import load_dotenv

load_dotenv()

description = """
# NFT Generation and Analysis API

🚀 This API allows you to create NFTs based on currently trending NFTs or user-provided descriptions. It also features advanced market analysis to accurately predict the value of NFTs with high precision.

## Summary

This API leverages the latest advancements in AI and blockchain technology to facilitate the creation and analysis of NFTs. By utilizing models like DALL-E and GPT-4, along with vector databases and embeddings, it ensures high-quality and relevant NFT generation. The API can also analyze the NFT market to provide accurate value predictions for NFTs.

## Tech Stack

- **OpenAI**: GPT-4 for generating NFT descriptions and market analysis.
- **Vector DB**: For storing and querying vector embeddings.
- **Embedding**: To handle semantic understanding and similarity searches.
- **FastAPI**: The web framework used for building this API.
- **DALL-E**: For generating images based on textual descriptions.

## Developer

### About the Developer

- **Name**: [Ankur Kumar Shukla]
- **LinkedIn**: [Connect me](https://www.linkedin.com/in/ankur-shukla-iiitv/)
- **Email**: [Any Querry](mailto:shuklaankur@gmail.com)

"""

app = FastAPI(
    title="GenAI NFT",
    description=description,
    summary="API made and maintained by Ankur Shukla",
    version="0.0.1",
    # terms_of_service="http://example.com/terms/",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:5000",
    "http://localhost:5500",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = OpenAI(api_key="sk-OidJwj7keEHJjSHz6vFhT3Blb")


# Sample list of events
events = [
    {
        "contractAddress": "0x123...abc",
        "name": "Demo Name 1",
        "tokenId": "1",
        "price": "1ETH",
        "image_url": "https://i.seadn.io/gae/v-1OvYmQg24tCc1cKsXjCei_9cwsr2jXrn-XXtv4Tdv3WIWwve4Ic2-W5QuqNvmxv8xftGkCyNEAZsac7wKIzD45ZYE3kSZUycoxxg?auto=format&dpr=1&w=1000",
    },
    {
        "contractAddress": "0x456...def",
        "name": "Demo Name 2",
        "tokenId": "39",
        "price": "2ETH",
        "image_url": "https://i.seadn.io/gae/HjbvSqZ7Cmr5fNZSX6WNPKyKCjOhg9GpxcYw62MoaDs1RMN15JXLtl3J49feThQX800S0uznyy_VJITF_e6oLGK3uCHcvqNpmR4g?auto=format&dpr=1&w=1000",
    },
]


# PYDANTIC MODELS  -  SCHEMA VALIDATION
class ImageResponse(BaseModel):
    contractAddress: str
    name: str
    tokenId: str
    price: str
    image_url: str
    description: str


class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1


class Event(BaseModel):
    contractAddress: str
    name: str
    tokenId: str
    price: str
    image_url: str


class EventsRequest(BaseModel):
    events: List[Event]


# API ROUTES & ENDPOINTS


@app.get("/get_event_data")
async def get_event_data():
    return EventsRequest(events=events)


@app.post("/get_trends")
async def get_trending_nft_desc() -> ImageRequest:
    events_request = await get_event_data()  # Adjust as needed for your actual request
    trending_nfts = await analyze_nft(events_request)

    combined_description = " ".join(nft["description"] for nft in trending_nfts)
    send_description = [nft["description"] for nft in trending_nfts]
    # print(combined_description)
    # Use GPT-4 to generate a description based on combined descriptions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "give name and description only. nothing else. Be to thew point always",
            },
            {
                "role": "user",
                "content": f""" You're an experienced NFT description writer with a keen eye for creativity and detail. Your specialty lies in crafting compelling descriptions for unique NFT artworks, ensuring that each piece is showcased in the best light possible. 

Your task is to create a new creative NFT description inspired by the phrase {combined_description}. Be sure to emphasize that it's an NFT art piece and should reflect that in its appearance. 

When writing the description, focus on capturing the essence of the artwork, highlighting its uniqueness, style, and any notable features that set it apart. Consider incorporating engaging language that sparks interest and curiosity in potential buyers. 

For example, when describing an NFT inspired by nature, you might mention how the artwork seamlessly blends vibrant colors to create a mesmerizing visual experience, inviting viewers to immerse themselves in a digital realm of beauty and tranquility. """,
            },
        ],
        max_tokens=150,
    )

    generated_description = response.choices[0].message.content

    return send_description,ImageRequest(
        prompt=generated_description, size="1024x1024", quality="standard", n=1
    )

# price prediction

# @app.post("/predict_price")
# async def predict_price





@app.post("/generate-image")
async def generate_image(request: Optional[ImageRequest] = None):
    try:
        if request is None:
            request = await get_trending_nft_desc()
        print(request)
        # prompt = f"{request.prompt} it's Board Ape Yacht Club NFT... so make it in that style make only one image"
        response = client.images.generate(
            model="dall-e-3",
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            n=request.n,
        )
        return {"image_url": response.data[0].url}
    except APIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-nft", response_model=List[ImageResponse])
async def analyze_nft(events_request: EventsRequest):
    results = []
    for event in events_request.events:
        try:
            print(event)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "give description detail (only) of nft so that image model can generate similar image  from that decription no additional salutation or explaination",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": event.image_url, "detail": "low"},
                            },
                        ],
                    },
                ],
                max_tokens=300,
            )
            description = response.choices[0].message.content
            
            results.append(
                {
                    "contractAddress": event.contractAddress,
                    "name": event.name,
                    "tokenId": event.tokenId,
                    "price": event.price,
                    "image_url": event.image_url,
                    "description": description,
                }
            )
        except APIError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return results
