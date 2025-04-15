import requests
import os
import json

def download_openi_data(output_dir, limit=7470):
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://openi.nlm.nih.gov"
    api_url = f"{base_url}/api/search"
    params = {
        "query": "chest x-ray",
        "m": 1,         # Start index
        "n": 100,       # Results per page
        "it": "xr"      # X-ray modality
    }

    data = []
    for i in range(1, limit + 1, 100):
        print(f"Fetching records {i} to {i + 99}")
        params["m"] = i
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            try:
                results = response.json().get("list", [])
                if not results:
                    print("No results found in this batch.")
                    continue
                # Print the first item to inspect its structure (optional after debugging)
                if i == 1 and results:
                    print("Sample API response item:", json.dumps(results[0], indent=4))
                for item in results:
                    img_url = item.get("imgLarge")
                    # Access caption from the "image" dictionary
                    caption = item.get("image", {}).get("caption", "No caption available")
                    report = caption  # Use caption directly as the report
                    img_id = item.get("uid") or item.get("id") or None
                    if img_url and img_id:
                        full_img_url = f"{base_url}{img_url}"
                        img_path = os.path.join(output_dir, f"{img_id}.jpg")
                        try:
                            img_data = requests.get(full_img_url, timeout=10)
                            img_data.raise_for_status()
                            with open(img_path, "wb") as f:
                                f.write(img_data.content)
                            data.append({
                                "id": img_id,
                                "image": img_path,
                                "report": report.strip()  # Remove leading/trailing spaces
                            })
                        except requests.exceptions.RequestException as e:
                            print(f"Error downloading image {img_id}: {e}")
                            continue
            except Exception as e:
                print("Failed to parse response JSON:", e)
        else:
            print(f"Request failed at index {i} with status code {response.status_code}")

    metadata_path = os.path.join(output_dir, "openi_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Downloaded {len(data)} X-rays and saved metadata to {metadata_path}")

if __name__ == "__main__":
    output_dir = "data/raw/openi/"
    download_openi_data(output_dir)