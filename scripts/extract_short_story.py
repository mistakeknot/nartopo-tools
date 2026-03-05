import sys
import os
import json
import asyncio
import argparse

async def run_gemini_cli(prompt, stdin_data=None):
    if stdin_data:
        process = await asyncio.create_subprocess_exec(
            "gemini", "-y", "-p", prompt,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=stdin_data.encode('utf-8'))
    else:
        process = await asyncio.create_subprocess_exec(
            "gemini", "-y", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
    return stdout.decode('utf-8', errors='replace')

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collection_file")
    parser.add_argument("title")
    parser.add_argument("output_file")
    args = parser.parse_args()

    with open(args.collection_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Asking Gemini to find boundaries for '{args.title}' in {args.collection_file}...")
    prompt = f"""You are a precise text extraction assistant.
Find the exact start and end of the short story titled "{args.title}" in the collection text provided.
Return your answer EXACTLY as a JSON object with two keys: "start_string" and "end_string".
The "start_string" should be a unique sequence of exactly 50 characters at the very beginning of the story (which may just be the title itself and the first lines).
The "end_string" should be a unique sequence of exactly 50 characters at the very end of the story.
DO NOT output any other text or markdown besides the raw JSON object.
"""
    result = await run_gemini_cli(prompt, text)
    
    try:
        # Clean up result if it contains markdown code blocks
        clean_result = result.strip()
        if clean_result.startswith("```json"):
            clean_result = clean_result[7:]
        if clean_result.startswith("```"):
            clean_result = clean_result[3:]
        if clean_result.endswith("```"):
            clean_result = clean_result[:-3]
            
        boundaries = json.loads(clean_result.strip())
        start_str = boundaries["start_string"]
        end_str = boundaries["end_string"]
        
        start_idx = text.find(start_str)
        if start_idx == -1:
            print(f"Error: Could not find start string: {start_str}")
            sys.exit(1)
            
        end_idx = text.find(end_str, start_idx)
        if end_idx == -1:
            print(f"Error: Could not find end string: {end_str}")
            sys.exit(1)
            
        # Include the end string in the output
        end_idx += len(end_str)
        
        story_text = text[start_idx:end_idx]
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(story_text)
            
        print(f"Successfully extracted {len(story_text)} characters to {args.output_file}")
        
    except Exception as e:
        print(f"Failed to parse or extract: {e}")
        print(f"Gemini output was:\n{result}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
