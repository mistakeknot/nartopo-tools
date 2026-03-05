import sys
import os
import json
import asyncio
import time
import argparse

async def run_gemini_cli(prompt):
    process = await asyncio.create_subprocess_exec(
        "gemini", "-y", "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode('utf-8', errors='replace')

async def synthesize_frameworks(title, author):
    print(f"\n--- Synthesizing Frameworks for {title} by {author} ---")

    frameworks = [
        "Todorov's Equilibrium",
        "Actantial Model",
        "Quadrant Scores",
        "Lévi-Strauss's Binary Oppositions",
        "Cognitive Estrangement",
        "Bakhtin's Chronotope",
        "Aristotelian Poetics",
        "Jungian Archetypal Analysis",
        "Genette's Transtextuality"
    ]

    async def synthesize_single(fw):
        print(f"Synthesizing: {fw}...")
        prompt = f"""You are a Synthesis Sub-Agent specializing in {fw}.

Analyze the well-known short story "{title}" by {author}.
Output ONLY the {fw} mapping in valid JSON format.
"""
        if fw == "Quadrant Scores":
            prompt += """
Specifically, output exactly 6 floats between 0.0 and 1.0 for these metrics based on the story's pacing, conflict types, and plot structure:
- time_linearity: 0.0=Linear, 1.0=Fractured
- pacing_velocity: 0.0=Action-Driven, 1.0=Observational
- threat_scale: 0.0=Individual, 1.0=Systemic
- protagonist_fate: 0.0=Victory, 1.0=Assimilation
- conflict_style: 0.0=Western Combat, 1.0=Kishōtenketsu
- price_type: 0.0=Physical, 1.0=Ideological

Format the output strictly as JSON:
{
  "time_linearity": 0.0,
  "pacing_velocity": 0.0,
  "threat_scale": 0.0,
  "protagonist_fate": 0.0,
  "conflict_style": 0.0,
  "price_type": 0.0
}
"""
        res = await run_gemini_cli(prompt)
        return f"### {fw}\n{res}\n"

    tasks = [synthesize_single(fw) for fw in frameworks]
    results = await asyncio.gather(*tasks)
    return "\n".join(results)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("author")
    parser.add_argument("output_file")
    args = parser.parse_args()

    synthesis_result = await synthesize_frameworks(args.title, args.author)
    
    # Fake timeline for short stories
    timeline = '{"type": "exposition", "summary": "Initial status quo."}\n{"type": "action", "summary": "Main plot points."}\n'
    final_output = timeline + "\n\n=== SYNTHESIS ===\n" + synthesis_result

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_output)

if __name__ == "__main__":
    asyncio.run(main())
