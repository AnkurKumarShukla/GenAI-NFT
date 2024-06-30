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
import random
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
    "https://onchain-summer-joy.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = OpenAI(api_key="opEN ai key")


# Sample list of events
# https://i.seadn.io/gae/Pb7g49FxOMubuijHE2jop8zoeoo5NiJF445dG_1X9lAb4z7bTnASzG2d1I7kbkr2Ta0zeVzUT2YvdgSb4VvPRzOEFmIBWdidvmDh?auto=format&dpr=1&w=1000
# 13 eath 

# https://i.seadn.io/gae/JuQHPkErFScGb7O61h69BIbhjRAX9TswDvfxXPi1t4B08HNev5jCmErr9e1sS2v_flBX6PieMLJXDSUcsUr23VLZ_df9IWksgpm3?auto=format&dpr=1&w=1000
# 14.95 eath 
events = [
    {
        "contractAddress": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
        "name": "Punked Primate",
        "tokenId": "198",
        "price": "9ETH",
        "image_url": "https://i.seadn.io/gae/tu9y1zlTXFUuA62_Sjlwe1bhw0VfFEVqPlDBxydF2wZuinMwoq4_-X0rbb4IV0N6CnnB8daeLWM7VqfyHtTZjQE-383qt4GLB4WueLk?auto=format&dpr=1&w=128",
    },
    {
        "contractAddress": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
        "name": "Bubblegum Captain",
        "tokenId": "319",
        "price": "8ETH",
        "image_url": "https://i.seadn.io/gae/HjbvSqZ7Cmr5fNZSX6WNPKyKCjOhg9GpxcYw62MoaDs1RMN15JXLtl3J49feThQX800S0uznyy_VJITF_e6oLGK3uCHcvqNpmR4g?auto=format&dpr=1&w=1000",
    },
    {
        "contractAddress": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
        "name": "Cyber Chic Chimp",
        "tokenId": "112",
        "price": "12ETH",
        "image_url": "https://i.seadn.io/gae/Pb7g49FxOMubuijHE2jop8zoeoo5NiJF445dG_1X9lAb4z7bTnASzG2d1I7kbkr2Ta0zeVzUT2YvdgSb4VvPRzOEFmIBWdidvmDh?auto=format&dpr=1&w=1000",
    },
    {
        "contractAddress": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
        "name": "Imperial Ape",
        "tokenId": "997",
        "price": "14ETH",
        "image_url": "https://i.seadn.io/gae/JuQHPkErFScGb7O61h69BIbhjRAX9TswDvfxXPi1t4B08HNev5jCmErr9e1sS2v_flBX6PieMLJXDSUcsUr23VLZ_df9IWksgpm3?auto=format&dpr=1&w=1000",
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
    # import pdb; pdb.set_trace()
    events_request = await get_event_data()  # Adjust as needed for your actual request
    trending_nfts = await analyze_nft(events_request)

    combined_description = " ".join(nft["description"] for nft in trending_nfts)
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

    return ImageRequest(
        prompt=generated_description, size="1024x1024", quality="standard", n=1
    )

# # price prediction

# @app.post("/predict_price")
# async def predict_price(request: EventsRequest):





@app.post("/generate-image")
async def generate_image(request: Optional[ImageRequest] = None):
    # import pdb; pdb.set_trace()
    try:
        if request is None:
            request = await get_trending_nft_desc()
        print(request)
        prompt = f"it's Board Ape Yacht Club NFT... so make it in that style , make one entity in image , do not make multiple ape or monkey face in one image {request.prompt} "
        
        response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "give one name of two words  of given nft description. nothing else. Be to thew point always",
            },
            {
                "role": "user",
                "content": f"""give only one name(only ) of two  words  of this nft desctiption. this is not collection so do not use word collection  and do not put / and " simple 2 word string   : -  {prompt}.""",
            },
        ],
        max_tokens=150,
    )
        name= response1.choices[0].message.content
        random_number = random.uniform(8, 14)

# Format the number to two decimal places
        price = f"{random_number:.2f}"
        



        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=request.size,
            quality=request.quality,
            response_format='b64_json',
            n=request.n,
        )
        return {"image_url": response,
                "name": name,
                "price":price,}
    except APIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-nft", response_model=List[ImageResponse])
async def analyze_nft(events_request: EventsRequest):
    results = []
    # import pdb; pdb.set_trace()
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
