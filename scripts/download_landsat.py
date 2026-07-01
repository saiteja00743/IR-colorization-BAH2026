import os
import argparse
import requests
import sys

def download_file(url, local_path, headers=None):
    """Downloads a file from a URL with a console progress bar."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    temp_path = local_path + ".tmp"
    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_written = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        if total_size > 0:
                            percent = (bytes_written / total_size) * 100
                            mb_written = bytes_written / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            sys.stdout.write(f"\rDownloading: {mb_written:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
                            sys.stdout.flush()
            sys.stdout.write("\n")
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(temp_path, local_path)
            return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"\nError downloading {url}: {e}")
        return False

def search_and_download_landsat(limit=2, output_dir="input"):
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Search for Landsat 9 L2SP scenes with low cloud cover
    query = {
        "collections": ["landsat-c2-l2"],
        "limit": limit,
        "query": {
            "platform": {"eq": "landsat-9"},
            "landsat:correction": {"eq": "L2SP"},
            "eo:cloud_cover": {"lt": 5}
        }
    }
    
    print(f"Searching Planetary Computer for {limit} clean Landsat 9 scenes...")
    try:
        response = requests.post(url, json=query, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        items = data.get("features", [])
        print(f"Found {len(items)} scenes.")
        
        if not items:
            print("No matching Landsat 9 scenes found.")
            return False
            
        success_count = 0
        for idx, item in enumerate(items):
            product_id = item["id"]
            print(f"\n[{idx + 1}/{len(items)}] Processing scene: {product_id}")
            
            # Map of STAC asset keys to expected band suffixes
            band_mapping = {
                "blue": "B2",
                "green": "B3",
                "red": "B4",
                "lwir11": "B10"
            }
            
            scene_dir = os.path.join(output_dir, product_id)
            all_bands_ok = True
            
            for asset_key, band_suffix in band_mapping.items():
                asset = item["assets"].get(asset_key)
                if not asset:
                    print(f"Warning: Asset {asset_key} not found in STAC metadata. Skipping scene.")
                    all_bands_ok = False
                    break
                
                raw_href = asset["href"]
                # Request a signed URL from the token/SAS signing API
                sign_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={raw_href}"
                sign_res = requests.get(sign_url, headers=headers, timeout=20)
                if sign_res.status_code != 200:
                    print(f"Failed to sign URL for asset {asset_key}: {sign_res.status_code}")
                    all_bands_ok = False
                    break
                
                signed_href = sign_res.json()["href"]
                dest_filename = f"{product_id}_{band_suffix}.TIF"
                dest_path = os.path.join(scene_dir, dest_filename)
                
                print(f"  Downloading band {band_suffix} ({asset_key}) -> {dest_filename}")
                if os.path.exists(dest_path):
                    print(f"  File already exists: {dest_filename}. Skipping download.")
                else:
                    ok = download_file(signed_href, dest_path, headers=headers)
                    if not ok:
                        all_bands_ok = False
                        break
            
            if all_bands_ok:
                print(f"Successfully processed all bands for scene {product_id}.")
                success_count += 1
            else:
                print(f"Failed to process all bands for scene {product_id}.")
        
        print(f"\nCompleted! Successfully downloaded {success_count} scenes to {output_dir}/")
        return success_count > 0
    except Exception as e:
        print(f"Error searching or downloading Landsat data: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Landsat 9 data from Microsoft Planetary Computer.")
    parser.add_argument('--limit', type=int, default=2, help="Number of scenes to download.")
    parser.add_argument('--output_dir', type=str, default="input", help="Directory to save the downloaded data.")
    args = parser.parse_args()
    
    success = search_and_download_landsat(limit=args.limit, output_dir=args.output_dir)
    sys.exit(0 if success else 1)
