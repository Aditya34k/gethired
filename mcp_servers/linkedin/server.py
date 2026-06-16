import asyncio
import json
import sys
import os

# Add project root to path so we can import our scraper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from mcp_servers.linkedin.scraper import fetch_linkedin_profile

# Create the MCP server — give it a name agents will use to identify it
app = Server("linkedin-fetcher")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    Tells any agent that connects: here are the tools I offer.
    The agent reads this to know what it can call.
    """
    return [
        types.Tool(
            name="fetch_linkedin_profile",
            description=(
                "Fetch public LinkedIn profile data for a candidate. "
                "Returns skills, experience, headline and summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "LinkedIn profile URL e.g. https://linkedin.com/in/username"
                    }
                },
                "required": ["url"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    Executes the tool when an agent calls it.
    Returns the result as a TextContent block — MCP's standard format.
    """
    if name == "fetch_linkedin_profile":
        url = arguments.get("url", "")
        result = await fetch_linkedin_profile(url)

        # Serialize to JSON string — MCP passes everything as text
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """
    MCP servers communicate over stdin/stdout — not HTTP.
    This is why they run as a separate process.
    The agent spawns this process and talks to it via pipes.
    """
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())