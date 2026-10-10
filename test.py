from mcp.server.fastmcp import FastMCP

mcp = FastMCP("company-tools")

@mcp.tool()
async def get_logs(service: str):
    return {
        "service": service,
        "errors": [...]
    }

@mcp.tool()
async def search_sop(query: str):
    return {
        "documents": [...]
    }

@mcp.tool()
async def send_slack(message: str):
    return {
        "status": "sent"
    }