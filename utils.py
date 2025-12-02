#!/usr/bin/env python3
"""
Utility functions for working with The Graph Protocol subgraphs and Web3.

This module provides helper functions for:
- Fetching subgraph schemas via GraphQL introspection
- Parsing subgraph IDs from URLs
- Creating Web3 clients for Infura (HTTP and WebSocket)

Usage:
    # Subgraph schema
    from utils import get_subgraph_schema
    schema = get_subgraph_schema("HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1")

    # Web3 client
    from utils import get_infura_web3
    w3 = get_infura_web3()
    print(w3.eth.block_number)

    # From command line
    python utils.py HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1
"""

import argparse
import json
import os
import re
from typing import Any, Optional, Literal

import requests
from web3 import Web3


# GraphQL introspection query to fetch the full schema
INTROSPECTION_QUERY = """
query IntrospectionQuery {
  __schema {
    queryType {
      name
    }
    mutationType {
      name
    }
    subscriptionType {
      name
    }
    types {
      ...FullType
    }
    directives {
      name
      description
      locations
      args {
        ...InputValue
      }
    }
  }
}

fragment FullType on __Type {
  kind
  name
  description
  fields(includeDeprecated: true) {
    name
    description
    args {
      ...InputValue
    }
    type {
      ...TypeRef
    }
    isDeprecated
    deprecationReason
  }
  inputFields {
    ...InputValue
  }
  interfaces {
    ...TypeRef
  }
  enumValues(includeDeprecated: true) {
    name
    description
    isDeprecated
    deprecationReason
  }
  possibleTypes {
    ...TypeRef
  }
}

fragment InputValue on __InputValue {
  name
  description
  type {
    ...TypeRef
  }
  defaultValue
}

fragment TypeRef on __Type {
  kind
  name
  ofType {
    kind
    name
    ofType {
      kind
      name
      ofType {
        kind
        name
        ofType {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
              }
            }
          }
        }
      }
    }
  }
}
"""


def extract_subgraph_id(subgraph_input: str) -> str:
    """
    Extract a subgraph ID from either a full URL or a bare ID.

    Accepts:
    - Full URL: "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/HMuAwuf..."
    - Subgraph ID only: "HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1"

    Args:
        subgraph_input: Either a full subgraph URL or just the subgraph ID

    Returns:
        The extracted subgraph ID

    Raises:
        ValueError: If the input cannot be parsed as a valid subgraph ID or URL

    Example:
        >>> extract_subgraph_id("HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1")
        'HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1'

        >>> extract_subgraph_id("https://gateway.thegraph.com/api/subgraphs/id/HMuAwuf...")
        'HMuAwuf...'
    """
    # Check if it's a URL
    if subgraph_input.startswith("http"):
        # Extract ID from URL pattern: .../subgraphs/id/<ID>
        match = re.search(r"/subgraphs/id/([^/?\s]+)", subgraph_input)
        if match:
            return match.group(1)
        raise ValueError(
            f"Could not extract subgraph ID from URL: {subgraph_input}\n"
            "Expected URL format: https://gateway.thegraph.com/api/.../subgraphs/id/<SUBGRAPH_ID>"
        )

    # Assume it's a bare ID - validate it looks reasonable
    # Subgraph IDs are typically alphanumeric strings of 40+ characters
    if re.match(r"^[A-Za-z0-9]{30,}$", subgraph_input):
        return subgraph_input

    raise ValueError(
        f"Invalid subgraph ID format: {subgraph_input}\n"
        "Expected either a full URL or an alphanumeric subgraph ID"
    )


def get_subgraph_schema(
    subgraph_input: str,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fetch the GraphQL schema for a subgraph using introspection.

    This function queries The Graph's decentralized network to retrieve
    the full schema of a subgraph, which includes all available types,
    queries, and their fields.

    Args:
        subgraph_input: Either a full subgraph URL or just the subgraph ID
            - URL example: "https://gateway.thegraph.com/api/subgraphs/id/HMuAwuf..."
            - ID example: "HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1"
        api_key: The Graph API key. If not provided, uses GRAPH_API_KEY env var.

    Returns:
        Dictionary containing the full GraphQL schema with structure:
        {
            "__schema": {
                "queryType": {"name": "Query"},
                "types": [...],  # All available types
                "directives": [...]
            }
        }

    Raises:
        ValueError: If API key is missing or subgraph ID is invalid
        requests.RequestException: If the HTTP request fails

    Example:
        >>> schema = get_subgraph_schema("5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV")
        >>> print([t["name"] for t in schema["__schema"]["types"] if t["name"] == "Pool"])
        ['Pool']
    """
    # Get API key
    api_key = api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set GRAPH_API_KEY environment variable or pass api_key parameter.\n"
            "Get a free API key at https://thegraph.com/studio/"
        )

    # Extract the subgraph ID
    subgraph_id = extract_subgraph_id(subgraph_input)

    # Build the endpoint URL
    endpoint = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"

    # Execute the introspection query
    response = requests.post(
        endpoint,
        json={"query": INTROSPECTION_QUERY},
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    response.raise_for_status()

    result = response.json()

    # Check for GraphQL errors
    if "errors" in result:
        error_messages = [e.get("message", str(e)) for e in result["errors"]]
        raise ValueError(f"GraphQL errors: {'; '.join(error_messages)}")

    return result.get("data", {})


def print_schema_summary(schema: dict[str, Any]) -> None:
    """
    Print a human-readable summary of a GraphQL schema.

    Args:
        schema: The schema dictionary returned by get_subgraph_schema()
    """
    schema_data = schema.get("__schema", {})
    types = schema_data.get("types", [])

    # Filter out internal GraphQL types (those starting with __)
    user_types = [t for t in types if not t.get("name", "").startswith("__")]

    # Group types by kind
    type_groups = {}
    for t in user_types:
        kind = t.get("kind", "UNKNOWN")
        if kind not in type_groups:
            type_groups[kind] = []
        type_groups[kind].append(t)

    print("\n" + "=" * 60)
    print("SCHEMA SUMMARY")
    print("=" * 60)

    # Print query type info
    query_type = schema_data.get("queryType", {})
    if query_type:
        print(f"\nQuery Type: {query_type.get('name', 'N/A')}")

    # Print type counts by kind
    print("\nType Counts:")
    for kind, types_list in sorted(type_groups.items()):
        print(f"  {kind}: {len(types_list)}")

    # Print object types with their fields
    print("\n" + "-" * 60)
    print("OBJECT TYPES (Entities)")
    print("-" * 60)

    object_types = type_groups.get("OBJECT", [])
    for obj_type in sorted(object_types, key=lambda x: x.get("name", "")):
        name = obj_type.get("name", "")
        if name in ("Query", "Subscription"):
            continue

        fields = obj_type.get("fields", []) or []
        print(f"\n{name} ({len(fields)} fields)")

        # Print first few fields as preview
        for field in fields[:5]:
            field_name = field.get("name", "")
            field_type = _format_type(field.get("type", {}))
            print(f"  - {field_name}: {field_type}")

        if len(fields) > 5:
            print(f"  ... and {len(fields) - 5} more fields")


def _format_type(type_obj: dict) -> str:
    """Format a GraphQL type object as a string."""
    if not type_obj:
        return "Unknown"

    kind = type_obj.get("kind", "")
    name = type_obj.get("name", "")

    if kind == "NON_NULL":
        inner = _format_type(type_obj.get("ofType", {}))
        return f"{inner}!"
    elif kind == "LIST":
        inner = _format_type(type_obj.get("ofType", {}))
        return f"[{inner}]"
    elif name:
        return name
    else:
        return "Unknown"


def get_infura_web3(
    network: str = "mainnet",
    api_key: Optional[str] = None,
) -> Web3:
    """
    Create a Web3 HTTP client connected to Infura.

    Args:
        network: Ethereum network name (mainnet, goerli, sepolia, etc.)
        api_key: Infura API key. If not provided, uses INFURA_API_KEY env var.

    Returns:
        Web3 instance connected to Infura via HTTP.

    Raises:
        ValueError: If API key is missing.

    Example:
        >>> w3 = get_infura_web3()
        >>> w3.eth.block_number
        18500000
    """
    api_key = api_key or os.environ.get("INFURA_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set INFURA_API_KEY environment variable or pass api_key parameter.\n"
            "Get a free API key at https://infura.io/"
        )

    endpoint = f"https://{network}.infura.io/v3/{api_key}"
    provider = Web3.HTTPProvider(endpoint)
    w3 = Web3(provider)

    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Infura {network}")

    return w3


def get_infura_endpoint(
    use_websocket: bool = False,
    network: str = "mainnet",
    api_key: Optional[str] = None,
) -> str:
    """
    Get the Infura endpoint URL for HTTP or WebSocket connections.

    Args:
        use_websocket: If True, return WebSocket URL; otherwise HTTP URL.
        network: Ethereum network name (mainnet, goerli, sepolia, etc.)
        api_key: Infura API key. If not provided, uses INFURA_API_KEY env var.

    Returns:
        Infura endpoint URL string.

    Example:
        >>> get_infura_endpoint()
        'https://mainnet.infura.io/v3/YOUR_API_KEY'

        >>> get_infura_endpoint(use_websocket=True)
        'wss://mainnet.infura.io/ws/v3/YOUR_API_KEY'
    """
    api_key = api_key or os.environ.get("INFURA_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set INFURA_API_KEY environment variable or pass api_key parameter.\n"
            "Get a free API key at https://infura.io/"
        )

    if use_websocket:
        return f"wss://{network}.infura.io/ws/v3/{api_key}"
    else:
        return f"https://{network}.infura.io/v3/{api_key}"


def main():
    """
    Command-line interface for fetching subgraph schemas.

    Usage:
        python utils.py <subgraph_id_or_url> [options]

    Examples:
        python utils.py HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1
        python utils.py https://gateway.thegraph.com/api/subgraphs/id/HMuAwuf... --output schema.json
    """
    parser = argparse.ArgumentParser(
        description="Fetch and display GraphQL schema for a subgraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print schema summary for a subgraph ID
  python utils.py 5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV

  # Fetch schema from a full URL
  python utils.py https://gateway.thegraph.com/api/subgraphs/id/HMuAwuf...

  # Save full schema to a JSON file
  python utils.py 5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV --output schema.json

  # Print full JSON schema to stdout
  python utils.py 5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV --json
        """,
    )

    parser.add_argument(
        "subgraph",
        type=str,
        help="Subgraph ID or full URL to fetch schema for",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The Graph API key (or set GRAPH_API_KEY env var)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save full schema to a JSON file",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON schema instead of summary",
    )

    args = parser.parse_args()

    try:
        print(f"Fetching schema for: {args.subgraph}")
        schema = get_subgraph_schema(args.subgraph, api_key=args.api_key)

        if args.output:
            # Save to file
            with open(args.output, "w") as f:
                json.dump(schema, f, indent=2)
            print(f"Schema saved to: {args.output}")
        elif args.json:
            # Print full JSON
            print(json.dumps(schema, indent=2))
        else:
            # Print summary
            print_schema_summary(schema)

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except requests.RequestException as e:
        print(f"HTTP Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
