"""MCP servers for AcademicOS.

`qbank_server` exposes the question bank over the Model Context Protocol so AI
agents and the backend query the same corpus through the same query engine.

Note: this package deliberately does NOT import `qbank_server` here. Doing so
makes `python -m academicos.mcp.qbank_server` emit "found in sys.modules after
import of package, but prior to execution", which is a real warning about a
module being loaded twice -- the entry point must be importable without its own
package pulling it in.
"""

__all__ = ["qbank_server"]
