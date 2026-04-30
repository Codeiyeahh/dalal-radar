import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

load_dotenv()

def test_proxy():
    user = os.getenv("WEBSHARE_USERNAME")
    password = os.getenv("WEBSHARE_PASSWORD")
    
    print(f"Testing with user: {user}")
    
    if not user or not password:
        print("Error: WEBSHARE credentials missing from .env")
        return

    proxy_config = WebshareProxyConfig(
        proxy_username=user,
        proxy_password=password
    )
    
    # Try a very popular video that definitely has transcripts
    video_id = "dVd9kdTTLLo" 
    
    print(f"Attempting to fetch transcript for {video_id} via proxy...")
    try:
        api = YouTubeTranscriptApi(proxy_config=proxy_config)
        srt_obj = api.fetch(video_id, languages=['en'])
        print("✅ SUCCESS! Proxy is working.")
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_proxy()
