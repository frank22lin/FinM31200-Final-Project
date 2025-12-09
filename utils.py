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
import time
import random
from datetime import datetime
from typing import Any, Optional, Literal, Union, List

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


# Known contract addresses for AAVE and Compound on Ethereum mainnet
# These can be overridden by users
KNOWN_CONTRACT_ADDRESSES = {
    "aave_v3_pool": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",  # AAVE V3 Pool
    "compound_v3_comet": "0xc3d688B66703497DAA19211EEdff47f25384cdc3",  # Compound V3 USDC Comet
    "compound_v2_comptroller": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",  # Compound V2 Comptroller
}


def retry_with_backoff(func, max_retries: int = 5, base_delay: float = 0.5, max_delay: float = 30):
    """
    Retry a function with exponential backoff.

    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries

    Returns:
        Result of the function call

    Raises:
        Exception: If all retries are exhausted
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            time.sleep(delay)


def decode_liquidation_event(
    log: dict,
    event_hash: str,
    w3: Web3,
) -> dict[str, Any]:
    """
    Decode a liquidation event log based on the event hash.

    Supports:
    - AAVE V3 LiquidationCall event
    - Compound LiquidationCall event

    Args:
        log: Event log dictionary from get_logs()
        event_hash: The event signature hash (topics[0])
        w3: Web3 instance for address conversion

    Returns:
        Dictionary with decoded event fields:
        - For AAVE: collateralAsset, debtAsset, user, debtToCover,
          liquidatedCollateralAmount, liquidator, receiveAToken
        - For Compound: liquidator, borrower, repayAmount, cTokenCollateral, seizeTokens
    """
    # AAVE V3 LiquidationCall event hash
    # Event: LiquidationCall(address indexed collateralAsset, address indexed debtAsset,
    #                        address indexed user, uint256 debtToCover, uint256 liquidatedCollateralAmount,
    #                        address liquidator, bool receiveAToken)
    AAVE_LIQUIDATION_HASH = "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286"

    # Compound LiquidationCall event hash
    # Event: LiquidationCall(address indexed liquidator, address indexed borrower,
    #                        uint256 repayAmount, address indexed cTokenCollateral, uint256 seizeTokens)
    COMPOUND_LIQUIDATION_HASH = "0x298637f684da70674f26509b10f07ec2fbc77a335ab1e7d6215a4b2484d8bb52"

    topics = log.get("topics", [])
    data = log.get("data", "0x")
    
    # Convert data to string if it's HexBytes or bytes
    if hasattr(data, 'hex') and not isinstance(data, str):
        # HexBytes or bytes-like object
        data = "0x" + data.hex()
    elif isinstance(data, bytes):
        data = "0x" + data.hex()
    elif not isinstance(data, str):
        # Fallback: try to convert to string
        data = str(data)
    # If it's already a string, keep it as is

    if event_hash.lower() == AAVE_LIQUIDATION_HASH.lower():
        # AAVE V3 LiquidationCall decoding
        # topics[0] = event hash
        # topics[1] = collateralAsset (indexed)
        # topics[2] = debtAsset (indexed)
        # topics[3] = user (indexed)
        # data = debtToCover (uint256) + liquidatedCollateralAmount (uint256) + liquidator (address) + receiveAToken (bool)

        if len(topics) < 4:
            raise ValueError("Invalid AAVE liquidation event: insufficient topics")

        # Extract addresses from topics (topics are HexBytes objects from Web3.py, or strings)
        # Addresses are stored as 32-byte values, right-aligned (last 20 bytes)
        def extract_address(topic):
            if isinstance(topic, str):
                # String - remove 0x prefix if present
                hex_str = topic[2:] if topic.startswith('0x') else topic
            elif hasattr(topic, 'hex'):
                # HexBytes or bytes-like object with hex() method
                hex_str = topic.hex()
            elif isinstance(topic, bytes):
                # Raw bytes
                hex_str = topic.hex()
            else:
                # Fallback: convert to bytes then hex
                hex_str = bytes(topic).hex()
            # Ensure we have at least 40 hex chars (20 bytes for address)
            if len(hex_str) < 40:
                raise ValueError(f"Topic too short for address extraction: {hex_str}")
            return w3.to_checksum_address("0x" + hex_str[-40:])
        
        collateral_asset = extract_address(topics[1])
        debt_asset = extract_address(topics[2])
        user = extract_address(topics[3])

        # Decode data (remove 0x prefix)
        data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)

        # debtToCover (uint256, 32 bytes)
        debt_to_cover = int.from_bytes(data_bytes[0:32], byteorder="big")
        # liquidatedCollateralAmount (uint256, 32 bytes)
        liquidated_collateral_amount = int.from_bytes(data_bytes[32:64], byteorder="big")
        # liquidator (address, 32 bytes, right-aligned)
        liquidator = w3.to_checksum_address(data_bytes[64:96][-20:].hex())
        # receiveAToken (bool, 32 bytes)
        receive_a_token = bool(int.from_bytes(data_bytes[96:128], byteorder="big"))

        return {
            "protocol": "AAVE",
            "collateralAsset": collateral_asset,
            "debtAsset": debt_asset,
            "user": user,
            "debtToCover": debt_to_cover,
            "liquidatedCollateralAmount": liquidated_collateral_amount,
            "liquidator": liquidator,
            "receiveAToken": receive_a_token,
        }

    elif event_hash.lower() == COMPOUND_LIQUIDATION_HASH.lower():
        # Compound LiquidationCall decoding
        # topics[0] = event hash
        # topics[1] = liquidator (indexed)
        # topics[2] = borrower (indexed)
        # topics[3] = cTokenCollateral (indexed)
        # data = repayAmount (uint256) + seizeTokens (uint256)

        if len(topics) < 4:
            raise ValueError("Invalid Compound liquidation event: insufficient topics")

        # Extract addresses from topics
        def extract_address(topic):
            if isinstance(topic, str):
                # String - remove 0x prefix if present
                hex_str = topic[2:] if topic.startswith('0x') else topic
            elif hasattr(topic, 'hex'):
                # HexBytes or bytes-like object with hex() method
                hex_str = topic.hex()
            else:
                # Fallback: convert to bytes then hex
                hex_str = bytes(topic).hex()
            return w3.to_checksum_address("0x" + hex_str[-40:])
        
        liquidator = extract_address(topics[1])
        borrower = extract_address(topics[2])
        c_token_collateral = extract_address(topics[3])

        # Decode data (data is already converted to string above)
        data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)

        # repayAmount (uint256, 32 bytes)
        repay_amount = int.from_bytes(data_bytes[0:32], byteorder="big")
        # seizeTokens (uint256, 32 bytes)
        seize_tokens = int.from_bytes(data_bytes[32:64], byteorder="big")

        return {
            "protocol": "Compound",
            "liquidator": liquidator,
            "borrower": borrower,
            "repayAmount": repay_amount,
            "cTokenCollateral": c_token_collateral,
            "seizeTokens": seize_tokens,
        }

    else:
        raise ValueError(f"Unknown event hash: {event_hash}")


def get_transactions_by_event_hash(
    w3: Web3,
    event_hashes: Union[str, List[str]],
    from_block: int,
    to_block: int,
    contract_addresses: Optional[Union[str, List[str]]] = None,
) -> List[dict[str, Any]]:
    """
    Find transactions containing specific event hashes within a block range.

    Args:
        w3: Web3 instance
        event_hashes: Single event hash string or list of event hash strings
        from_block: Starting block number (inclusive)
        to_block: Ending block number (inclusive)
        contract_addresses: Optional contract address(es) to filter by.
                           If None, searches all contracts.

    Returns:
        List of dictionaries, each containing:
        - transactionHash: Transaction hash
        - blockNumber: Block number
        - blockHash: Block hash
        - logIndex: Log index within the transaction
        - contractAddress: Address of the contract that emitted the event
        - eventHash: The event hash that matched
        - decodedEvent: Decoded event data (if decode_liquidation_event succeeds)
        - transaction: Full transaction data
        - receipt: Transaction receipt

    Example:
        >>> w3 = get_infura_web3()
        >>> latest = w3.eth.block_number
        >>> events = get_transactions_by_event_hash(
        ...     w3,
        ...     "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286",
        ...     latest - 100,
        ...     latest
        ... )
        >>> print(f"Found {len(events)} AAVE liquidation events")
    """
    # Normalize event_hashes to list
    if isinstance(event_hashes, str):
        event_hashes = [event_hashes]

    # Normalize contract_addresses to list or None
    if isinstance(contract_addresses, str):
        contract_addresses = [contract_addresses]

    results = []

    # Search for each event hash
    for event_hash_str in event_hashes:
        # Build filter parameters
        # Ensure event hash has 0x prefix for Web3.py
        topic_hex = event_hash_str if event_hash_str.startswith("0x") else "0x" + event_hash_str
        filter_params = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "topics": [topic_hex],  # topics[0] is the event signature
        }

        if contract_addresses:
            filter_params["address"] = contract_addresses

        # Get logs with retry logic
        try:
            logs = retry_with_backoff(
                lambda: w3.eth.get_logs(filter_params)
            )
        except Exception as e:
            print(f"Error fetching logs for event {event_hash_str}: {e}")
            continue

        # Process each log
        for log in logs:
            try:
                # Get transaction and receipt
                tx_hash = log["transactionHash"]
                tx = retry_with_backoff(lambda: w3.eth.get_transaction(tx_hash))
                receipt = retry_with_backoff(lambda: w3.eth.get_transaction_receipt(tx_hash))

                # Try to decode the event
                decoded_event = None
                try:
                    # Skip decoding for flash loan events (they're decoded separately in get_flash_loans_in_block)
                    if event_hash_str.lower() in [h.lower() for h in FLASH_LOAN_EVENT_HASHES.values()]:
                        decoded_event = None  # Don't decode flash loans here
                    else:
                        # Ensure log has topics and data fields
                        if "topics" not in log:
                            raise ValueError(f"Log missing topics field. Log keys: {list(log.keys())}")
                        if "data" not in log:
                            raise ValueError(f"Log missing data field. Log keys: {list(log.keys())}")
                        if len(log.get("topics", [])) < 4:
                            raise ValueError(f"Insufficient topics: {len(log.get('topics', []))}")
                        
                    decoded_event = decode_liquidation_event(log, event_hash_str, w3)
                except Exception as e:
                    # If decoding fails, still include the log but without decoded data
                    # Only print warning for non-flash-loan events to avoid spam
                    if event_hash_str.lower() not in [h.lower() for h in FLASH_LOAN_EVENT_HASHES.values()]:
                        # Only print full traceback for first error to avoid spam
                        if len(results) == 0:
                            import traceback
                            traceback.print_exc()

                result = {
                    "transactionHash": tx_hash.hex(),
                    "blockNumber": log["blockNumber"],
                    "blockHash": log["blockHash"].hex(),
                    "logIndex": log["logIndex"],
                    "contractAddress": log["address"],
                    "eventHash": event_hash_str,
                    "decodedEvent": decoded_event,
                    # Preserve raw log data for debugging
                    "topics": [t.hex() if hasattr(t, 'hex') else str(t) for t in log.get("topics", [])],
                    "data": log.get("data", "0x").hex() if hasattr(log.get("data", "0x"), 'hex') else str(log.get("data", "0x")),
                    "transaction": {
                        "from": tx["from"],
                        "to": tx.get("to"),
                        "value": tx["value"],
                        "gas": tx["gas"],
                        "gasPrice": tx.get("gasPrice"),
                        "nonce": tx["nonce"],
                    },
                    "receipt": {
                        "gasUsed": receipt["gasUsed"],
                        "status": receipt["status"],
                        "effectiveGasPrice": receipt.get("effectiveGasPrice"),
                    },
                }

                results.append(result)

            except Exception as e:
                print(f"Error processing log: {e}")
                continue

    return results


def get_liquidation_from_tx_hash(
    w3: Web3,
    tx_hash: str,
    event_hash: str,
) -> Optional[dict[str, Any]]:
    """
    Extract liquidation event information from a transaction hash.
    
    This function fetches the transaction receipt and searches through
    the logs to find and decode the liquidation event.
    
    Args:
        w3: Web3 instance
        tx_hash: Transaction hash (hex string with or without 0x prefix)
        event_hash: The event signature hash to search for
        
    Returns:
        Dictionary with decoded liquidation event data, or None if not found
        
    Example:
        >>> w3 = get_infura_web3()
        >>> tx_hash = "0x52b3c1cf36b7fbf1bceaa0405fafe08cdc5bb388cafc8811d22394940b1cbb43"
        >>> event_hash = "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286"
        >>> result = get_liquidation_from_tx_hash(w3, tx_hash, event_hash)
        >>> print(result["liquidator"])
    """
    # Normalize tx_hash
    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash
    
    try:
        # Get transaction receipt (contains all logs/events)
        receipt = retry_with_backoff(lambda: w3.eth.get_transaction_receipt(tx_hash))
        
        # Get transaction details
        tx = retry_with_backoff(lambda: w3.eth.get_transaction(tx_hash))
        
        # Search through logs for the matching event
        event_hash_lower = event_hash.lower()
        if not event_hash_lower.startswith("0x"):
            event_hash_lower = "0x" + event_hash_lower
        
        for log in receipt.get("logs", []):
            # Check if this log matches our event hash
            log_topics = log.get("topics", [])
            if len(log_topics) == 0:
                continue
                
            # Compare topic[0] (event signature) with our event hash
            topic0 = log_topics[0]
            if hasattr(topic0, 'hex'):
                topic0_hex = "0x" + topic0.hex()
            elif isinstance(topic0, bytes):
                topic0_hex = "0x" + topic0.hex()
            else:
                topic0_hex = topic0 if isinstance(topic0, str) else str(topic0)
            
            if topic0_hex.lower() == event_hash_lower:
                # Found matching event! Decode it
                try:
                    decoded_event = decode_liquidation_event(log, event_hash, w3)
                    
                    # Get block info for timestamp
                    block = w3.eth.get_block(receipt["blockNumber"])
                    
                    # Build result
                    result = {
                        "protocol": decoded_event.get("protocol"),
                        "transactionHash": tx_hash,
                        "blockNumber": receipt["blockNumber"],
                        "blockHash": receipt["blockHash"].hex(),
                        "logIndex": log["logIndex"],
                        "contractAddress": log["address"],
                        "timestamp": block["timestamp"],
                        "datetime": datetime.fromtimestamp(block["timestamp"]).isoformat(),
                    }
                    
                    # Add protocol-specific fields
                    if decoded_event["protocol"] == "AAVE":
                        result.update({
                            "liquidator": decoded_event.get("liquidator"),
                            "liquidatedUser": decoded_event.get("user"),
                            "debtAsset": decoded_event.get("debtAsset"),
                            "collateralAsset": decoded_event.get("collateralAsset"),
                            "debtToCover": str(decoded_event.get("debtToCover", 0)),
                            "liquidatedCollateralAmount": str(decoded_event.get("liquidatedCollateralAmount", 0)),
                            "receiveAToken": decoded_event.get("receiveAToken"),
                        })
                    elif decoded_event["protocol"] == "Compound":
                        result.update({
                            "liquidator": decoded_event.get("liquidator"),
                            "borrower": decoded_event.get("borrower"),
                            "repayAmount": str(decoded_event.get("repayAmount", 0)),
                            "cTokenCollateral": decoded_event.get("cTokenCollateral"),
                            "seizeTokens": str(decoded_event.get("seizeTokens", 0)),
                        })
                    
                    # Add transaction details
                    result.update({
                        "gasUsed": receipt["gasUsed"],
                        "gasPrice": tx.get("gasPrice"),
                        "effectiveGasPrice": receipt.get("effectiveGasPrice"),
                        "transactionFrom": tx["from"],
                        "transactionTo": tx.get("to"),
                        "transactionValue": tx["value"],
                    })
                    
                    return result
                    
                except Exception as e:
                    print(f"Warning: Could not decode event from transaction {tx_hash}: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        
        # Event not found in this transaction
        return None
        
    except Exception as e:
        print(f"Error fetching transaction {tx_hash}: {e}")
        return None


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


# Common token address to CoinGecko ID mapping
TOKEN_COINGECKO_MAP = {
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "weth",  # WETH
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": "usd-coin",  # USDC
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": "tether",  # USDT
    "0x6B175474E89094C44Da98b954EedeAC495271d0F": "dai",  # DAI
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": "wrapped-bitcoin",  # WBTC
    "0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72": "ethereum-name-service",  # ENS
    "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf": "compound-governance-token",  # COMP
    "0x8236a87084f8B84306f72007F36F2618A5634494": "aave",  # AAVE
    "0x657e8C867D8B37dCC18fA4Caead9C45EB088C642": "usd-coin",  # USDC (different address)
}

# Token address to decimals mapping
TOKEN_DECIMALS_MAP = {
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": 18,  # WETH
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": 6,  # USDC
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": 6,  # USDT
    "0x6B175474E89094C44Da98b954EedeAC495271d0F": 18,  # DAI
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": 8,  # WBTC (8 decimals!)
    "0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72": 18,  # ENS
    "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf": 18,  # COMP
    "0x8236a87084f8B84306f72007F36F2618A5634494": 18,  # AAVE
    "0x657e8C867D8B37dCC18fA4Caead9C45EB088C642": 6,  # USDC (different address)
}


# Cache for fetched decimals to avoid repeated API calls
_DECIMALS_CACHE: dict[str, int] = {}


def get_token_decimals(token_address: str, default: int = 18, w3: Optional[Web3] = None) -> int:
    """
    Get token decimals from cache, mapping, or fetch from API.
    
    Args:
        token_address: Token address
        default: Default decimals if not found (default 18)
        w3: Optional Web3 instance for on-chain fetching (not used if API available)
        
    Returns:
        Number of decimals
    """
    normalized = token_address.lower()
    
    # Check cache first
    if normalized in _DECIMALS_CACHE:
        return _DECIMALS_CACHE[normalized]
    
    # Check hardcoded mapping
    if normalized in TOKEN_DECIMALS_MAP:
        decimals = TOKEN_DECIMALS_MAP[normalized]
        _DECIMALS_CACHE[normalized] = decimals
        return decimals
    
    # Try to fetch from Etherscan API (free, no API key needed for basic calls)
    try:
        decimals = _fetch_decimals_etherscan(token_address)
        if decimals is not None:
            _DECIMALS_CACHE[normalized] = decimals
            return decimals
    except Exception as e:
        pass  # Fall through to default
    
    # Fallback to default and cache it
    _DECIMALS_CACHE[normalized] = default
    return default


def _fetch_decimals_etherscan(token_address: str) -> Optional[int]:
    """
    Fetch token decimals from Etherscan API by calling the decimals() function.
    
    Args:
        token_address: Token contract address
        
    Returns:
        Number of decimals, or None if not found
    """
    try:
        # ERC20 decimals() function signature: 0x313ce567
        # This is the function selector for decimals()
        url = "https://api.etherscan.io/api"
        params = {
            "module": "proxy",
            "action": "eth_call",
            "to": token_address,
            "data": "0x313ce567",  # decimals() function selector
            "tag": "latest",
            "apikey": "YourApiKeyToken"  # Can be empty for free tier
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("result") and data["result"] != "0x":
            # Result is hex string, convert to int
            decimals_hex = data["result"]
            decimals = int(decimals_hex, 16)
            if 0 <= decimals <= 18:  # Sanity check
                return decimals
    except Exception:
        pass
    
    # Fallback: Try CoinGecko API for token metadata
    try:
        url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{token_address}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # CoinGecko returns platform details with decimals
        platforms = data.get("platforms", {})
        if "ethereum" in platforms:
            detail_url = f"https://api.coingecko.com/api/v3/coins/{data.get('id')}"
            detail_response = requests.get(detail_url, timeout=10)
            detail_response.raise_for_status()
            detail_data = detail_response.json()
            
            # Check detail data for decimals
            if "detail_platforms" in detail_data:
                eth_platform = detail_data["detail_platforms"].get("ethereum", {})
                if "decimal_place" in eth_platform:
                    return int(eth_platform["decimal_place"])
    except Exception:
        pass
    
    return None


def get_token_price_coingecko(token_address: str, timestamp: Optional[int] = None) -> Optional[float]:
    """
    Get token price in USD from CoinGecko API.
    
    Args:
        token_address: Ethereum token address (checksummed or lowercase)
        timestamp: Optional Unix timestamp for historical price. If None, returns current price.
        
    Returns:
        Price in USD, or None if not found
        
    Example:
        >>> price = get_token_price_coingecko("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        >>> print(f"WETH price: ${price}")
        
        >>> # Get historical price
        >>> hist_price = get_token_price_coingecko("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", timestamp=1704067200)
    """
    # Normalize address
    token_address_lower = token_address.lower()
    
    # Get CoinGecko ID
    coingecko_id = TOKEN_COINGECKO_MAP.get(token_address_lower)
    
    # If not in mapping, try to get ID from contract address
    if not coingecko_id:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{token_address_lower}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            coingecko_id = data.get("id")
        except Exception as e:
            # Silently continue if lookup fails
            pass
    
    if not coingecko_id:
        # Fallback: Try direct contract address API (current price only)
        try:
            url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum"
            params = {
                "contract_addresses": token_address_lower,
                "vs_currencies": "usd"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if token_address_lower in data:
                return float(data[token_address_lower].get("usd", 0))
        except Exception as e:
            print(f"Warning: Could not fetch price for {token_address}: {e}")
        return None
    
    # If timestamp provided, get historical price
    if timestamp:
        return _get_historical_price_coingecko(coingecko_id, timestamp)
    
    # Get current price
    global _last_coingecko_call_time
    
    try:
        # Rate limiting: ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - _last_coingecko_call_time
        if time_since_last_call < _coingecko_min_delay:
            sleep_time = _coingecko_min_delay - time_since_last_call
            time.sleep(sleep_time)
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coingecko_id,
            "vs_currencies": "usd"
        }
        
        # Retry with exponential backoff for 429 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                _last_coingecko_call_time = time.time()
                response = requests.get(url, params=params, timeout=10)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * _coingecko_min_delay * 2
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Warning: Rate limited after {max_retries} attempts for {coingecko_id}")
                        return None
                
                response.raise_for_status()
                data = response.json()
                if coingecko_id in data:
                    return float(data[coingecko_id].get("usd", 0))
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * _coingecko_min_delay * 2
                    time.sleep(wait_time)
                    continue
                raise
                
    except Exception as e:
        if "429" not in str(e):
            print(f"Warning: Could not fetch price for {coingecko_id} ({token_address}): {e}")
    
    return None


# Global variable to track last CoinGecko API call time for rate limiting
_last_coingecko_call_time = 0.0
_coingecko_min_delay = 1.2  # Minimum delay between CoinGecko API calls (seconds)


def _get_historical_price_coingecko(coingecko_id: str, timestamp: int) -> Optional[float]:
    """
    Get historical token price from CoinGecko API for a specific timestamp.
    Includes rate limiting to avoid 429 errors.
    
    Args:
        coingecko_id: CoinGecko coin ID
        timestamp: Unix timestamp
        
    Returns:
        Price in USD at that timestamp, or None if not found
    """
    global _last_coingecko_call_time
    
    try:
        # Rate limiting: ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - _last_coingecko_call_time
        if time_since_last_call < _coingecko_min_delay:
            sleep_time = _coingecko_min_delay - time_since_last_call
            time.sleep(sleep_time)
        
        # Convert timestamp to date string (DD-MM-YYYY)
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime("%d-%m-%Y")
        
        # CoinGecko history endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/history"
        params = {
            "date": date_str,
            "localization": "false"
        }
        
        # Retry with exponential backoff for 429 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                _last_coingecko_call_time = time.time()
                response = requests.get(url, params=params, timeout=10)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Exponential backoff: wait longer for each retry
                        wait_time = (2 ** attempt) * _coingecko_min_delay * 2
                        print(f"Rate limited. Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Warning: Rate limited after {max_retries} attempts for {coingecko_id}")
                        return None
                
                response.raise_for_status()
                data = response.json()
                
                # Extract price from market_data
                if "market_data" in data and "current_price" in data["market_data"]:
                    usd_price = data["market_data"]["current_price"].get("usd")
                    if usd_price:
                        return float(usd_price)
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * _coingecko_min_delay * 2
                    print(f"Rate limited. Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue
                raise
                
    except Exception as e:
        # Only print warning if it's not a rate limit (already handled above)
        if "429" not in str(e):
            print(f"Warning: Could not fetch historical price for {coingecko_id} at {timestamp}: {e}")
    
    return None


def calculate_liquidation_pnl_usd(
    debt_asset: str,
    debt_amount: int,
    collateral_asset: str,
    collateral_amount: int,
    debt_decimals: int = 18,
    collateral_decimals: int = 18,
    timestamp: Optional[int] = None,
) -> dict[str, Optional[float]]:
    """
    Calculate liquidation PNL in USD using historical prices.
    
    Args:
        debt_asset: Debt token address
        debt_amount: Debt amount in raw units (wei)
        collateral_asset: Collateral token address
        collateral_amount: Collateral amount in raw units (wei)
        debt_decimals: Decimals for debt token (default 18)
        collateral_decimals: Decimals for collateral token (default 18)
        timestamp: Optional Unix timestamp for historical prices. If None, uses current prices.
        
    Returns:
        Dictionary with:
        - debt_usd: Debt amount in USD
        - collateral_usd: Collateral amount in USD
        - pnl_usd: Profit/Loss in USD
    """
    # Get prices (historical if timestamp provided)
    debt_price = get_token_price_coingecko(debt_asset, timestamp=timestamp)
    collateral_price = get_token_price_coingecko(collateral_asset, timestamp=timestamp)
    
    # Convert to human-readable amounts
    debt_human = debt_amount / (10 ** debt_decimals)
    collateral_human = collateral_amount / (10 ** collateral_decimals)
    
    # Calculate USD values
    debt_usd = debt_human * debt_price if debt_price else None
    collateral_usd = collateral_human * collateral_price if collateral_price else None
    
    # Calculate PNL
    pnl_usd = None
    if debt_usd is not None and collateral_usd is not None:
        pnl_usd = collateral_usd - debt_usd
    
    return {
        "debt_usd": debt_usd,
        "collateral_usd": collateral_usd,
        "pnl_usd": pnl_usd,
    }


# Flash Loan Event Hashes
FLASH_LOAN_EVENT_HASHES = {
    "aave_v1": "0x5b8f46461c1dd69fb968f1a003acee221ea3e19540e350233b612ddb43433b55",
    "aave_v2": "0x631042c832607452973831137f2d73e395028b44b250dedc5abb0ee766e168ac",
    "aave_v3": "0xefefaba5e921573100900a3ad9cf29f222d995fb3b6045797eaea7521bd8d6f0",
    "balancer": "0x0d7d75e01ab95780d3cd1c8ec0dd6c2ce19e3a20427eec8bf53283b6fb8e95f0",
}


def decode_flash_loan_event(log: dict, event_hash: str, w3: Web3) -> dict[str, Any]:
    """
    Decode a flash loan event log.
    
    Aave V3 FlashLoan event signature:
    FlashLoan(address indexed target, address indexed initiator, address indexed asset, 
              uint256 amount, uint256 premium, uint16 referralCode)
    
    Args:
        log: Event log dictionary from get_logs()
        event_hash: The event signature hash (topics[0])
        w3: Web3 instance for address conversion
        
    Returns:
        Dictionary with decoded flash loan fields:
        - protocol: Protocol name (Aave V1/V2/V3, Balancer)
        - target: Target address
        - initiator: Initiator/borrower address
        - asset: Token address
        - amount: Flash loan amount
        - premium: Premium paid
    """
    topics = log.get("topics", [])
    data = log.get("data", "0x")
    
    # Convert data to string if needed
    if hasattr(data, 'hex') and not isinstance(data, str):
        data = "0x" + data.hex()
    elif isinstance(data, bytes):
        data = "0x" + data.hex()
    elif not isinstance(data, str):
        data = str(data)
    
    # Determine protocol from event hash
    protocol = None
    for proto_name, proto_hash in FLASH_LOAN_EVENT_HASHES.items():
        if event_hash.lower() == proto_hash.lower():
            protocol = proto_name.replace("_", " ").title()
            break
    
    if not protocol:
        protocol = "Unknown"
    
    # Extract addresses from topics
    def extract_address(topic):
        if isinstance(topic, str):
            # Remove 0x prefix if present
            hex_str = topic[2:] if topic.startswith('0x') else topic
        elif hasattr(topic, 'hex'):
            hex_str = topic.hex()
        elif isinstance(topic, bytes):
            hex_str = topic.hex()
        else:
            hex_str = bytes(topic).hex()
        if len(hex_str) < 40:
            raise ValueError(f"Topic too short for address extraction: {hex_str}")
        return w3.to_checksum_address("0x" + hex_str[-40:])
    
    # Balancer FlashLoan structure (different!)
    # topics[0] = event hash
    # topics[1] = recipient (indexed)
    # topics[2] = token (indexed)
    # data = amount (uint256)
    
    # Check if this is Balancer (3 topics)
    balancer_hash = FLASH_LOAN_EVENT_HASHES.get("balancer", "").lower()
    if len(topics) == 3 and event_hash.lower() == balancer_hash:
        recipient = extract_address(topics[1])
        asset = extract_address(topics[2])
        
        # Decode data (just amount for Balancer)
        data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)
        if len(data_bytes) < 32:
            raise ValueError(f"Insufficient data bytes for Balancer flash loan: {len(data_bytes)}")
        amount = int.from_bytes(data_bytes[0:32], byteorder="big")
        
        return {
            "protocol": protocol,
            "target": recipient,  # Use recipient as target for Balancer
            "initiator": recipient,  # Balancer doesn't have separate initiator
            "asset": asset,
            "amount": amount,
            "premium": 0,  # Balancer doesn't have premium in the event
            "referralCode": 0,
        }
    
    # Aave V1/V2/V3 FlashLoan structure (4 topics)
    # topics[0] = event hash
    # topics[1] = target (indexed)
    # topics[2] = initiator (indexed) 
    # topics[3] = asset (indexed)
    # data = amount (uint256) + premium (uint256) + referralCode (uint16)
    
    if len(topics) >= 4:
        target = extract_address(topics[1])
        initiator = extract_address(topics[2])
        asset = extract_address(topics[3])
        
        # Decode data
        data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)
        
        if len(data_bytes) < 64:
            raise ValueError(f"Insufficient data bytes for Aave flash loan: {len(data_bytes)}")
        
        # amount (uint256, 32 bytes)
        amount = int.from_bytes(data_bytes[0:32], byteorder="big")
        # premium (uint256, 32 bytes)
        premium = int.from_bytes(data_bytes[32:64], byteorder="big")
        # referralCode (uint16, 32 bytes, right-aligned) - optional, may not be present
        referral_code = 0
        if len(data_bytes) >= 96:
            referral_code = int.from_bytes(data_bytes[64:96][-2:], byteorder="big")
        
        return {
            "protocol": protocol,
            "target": target,
            "initiator": initiator,
            "asset": asset,
            "amount": amount,
            "premium": premium,
            "referralCode": referral_code,
        }
    
    raise ValueError(f"Invalid flash loan event: unexpected topic count ({len(topics)}) for protocol {protocol}")


def get_flash_loans_in_block(
    w3: Web3,
    block_number: int,
    flash_loan_hashes: Optional[List[str]] = None,
) -> List[dict[str, Any]]:
    """
    Get all flash loan events in a specific block.
    
    Args:
        w3: Web3 instance
        block_number: Block number to search
        flash_loan_hashes: Optional list of flash loan event hashes.
                         If None, uses all known flash loan hashes.
        
    Returns:
        List of dictionaries with flash loan event data
    """
    if flash_loan_hashes is None:
        flash_loan_hashes = list(FLASH_LOAN_EVENT_HASHES.values())
    
    all_flash_loans = []
    
    for event_hash in flash_loan_hashes:
        try:
            events = get_transactions_by_event_hash(
                w3,
                event_hash,
                block_number,
                block_number,
                contract_addresses=None
            )
            
            # Decode flash loan events
            for event in events:
                # Reconstruct log from event data
                # Topics are stored as hex strings in the event (from get_transactions_by_event_hash)
                topics_hex = event.get("topics", [])
                data_hex = event.get("data", "0x")
                
                # Ensure topics are in the right format (hex strings)
                # The decode function expects hex strings or HexBytes
                log = {
                    "topics": topics_hex,  # Already hex strings
                    "data": data_hex,  # Already hex string
                    "address": event.get("contractAddress"),
                }
                
                try:
                    decoded = decode_flash_loan_event(log, event_hash, w3)
                    decoded.update({
                        "transactionHash": event.get("transactionHash"),
                        "blockNumber": event.get("blockNumber"),
                        "logIndex": event.get("logIndex"),
                    })
                    all_flash_loans.append(decoded)
                except Exception as e:
                    # Silently skip if decoding fails (might be different event structure)
                    continue
                    
        except Exception as e:
            # Silently continue if block search fails
            continue
    
    return all_flash_loans


# DEX Swap Event Hashes
UNISWAP_V2_SWAP_HASH = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
UNISWAP_V3_SWAP_HASH = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# Chainlink Oracle Event Hashes
CHAINLINK_ANSWER_UPDATED_HASH = "0x0559884fd3a460db3073b7fc896cc77986f16e378210ded43186175bf646fc5f"

# Known Chainlink Price Feed addresses (can be extended)
CHAINLINK_PRICE_FEEDS = {
    "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419": "ETH/USD",
    "0xF9680D99D6C9589e2a93a78A04A279e509205945": "ETH/USD",  # Polygon
    "0x8A753747A1Fa494EC906cE90E9f37563A8AF630e": "ETH/USD",  # Rinkeby
}


def get_swaps_in_transaction(w3: Web3, tx_hash: str) -> List[dict[str, Any]]:
    """
    Get all DEX swaps in a transaction.
    
    Args:
        w3: Web3 instance
        tx_hash: Transaction hash
        
    Returns:
        List of dictionaries with swap event details
    """
    try:
        receipt = retry_with_backoff(lambda: w3.eth.get_transaction_receipt(tx_hash))
        swaps = []
        
        for log in receipt.get("logs", []):
            topics = log.get("topics", [])
            if not topics:
                continue
                
            topic0 = topics[0]
            if hasattr(topic0, 'hex'):
                topic0_hex = "0x" + topic0.hex()
            elif isinstance(topic0, bytes):
                topic0_hex = "0x" + topic0.hex()
            else:
                topic0_hex = topic0 if isinstance(topic0, str) else str(topic0)
            
            # Check if it's a Uniswap swap event
            if topic0_hex.lower() == UNISWAP_V2_SWAP_HASH.lower():
                swaps.append({
                    "protocol": "Uniswap V2",
                    "contract": log["address"],
                    "topics": [t.hex() if hasattr(t, 'hex') else str(t) for t in topics],
                    "data": log.get("data", "0x").hex() if hasattr(log.get("data", "0x"), 'hex') else str(log.get("data", "0x")),
                    "logIndex": log.get("logIndex"),
                })
            elif topic0_hex.lower() == UNISWAP_V3_SWAP_HASH.lower():
                swaps.append({
                    "protocol": "Uniswap V3",
                    "contract": log["address"],
                    "topics": [t.hex() if hasattr(t, 'hex') else str(t) for t in topics],
                    "data": log.get("data", "0x").hex() if hasattr(log.get("data", "0x"), 'hex') else str(log.get("data", "0x")),
                    "logIndex": log.get("logIndex"),
                })
        
        return swaps
    except Exception as e:
        print(f"Warning: Could not fetch swaps for transaction {tx_hash}: {e}")
        return []


def get_oracle_updates_in_transaction(w3: Web3, tx_hash: str) -> List[dict[str, Any]]:
    """
    Get oracle price feed updates in a transaction.
    
    Args:
        w3: Web3 instance
        tx_hash: Transaction hash
        
    Returns:
        List of dictionaries with oracle update details
    """
    try:
        receipt = retry_with_backoff(lambda: w3.eth.get_transaction_receipt(tx_hash))
        oracle_updates = []
        
        for log in receipt.get("logs", []):
            topics = log.get("topics", [])
            if not topics:
                continue
                
            topic0 = topics[0]
            if hasattr(topic0, 'hex'):
                topic0_hex = "0x" + topic0.hex()
            elif isinstance(topic0, bytes):
                topic0_hex = "0x" + topic0.hex()
            else:
                topic0_hex = topic0 if isinstance(topic0, str) else str(topic0)
            
            # Check if it's a Chainlink AnswerUpdated event
            if topic0_hex.lower() == CHAINLINK_ANSWER_UPDATED_HASH.lower():
                # AnswerUpdated(int256 current, uint256 updatedAt, uint256 roundId)
                data = log.get("data", "0x")
                if hasattr(data, 'hex'):
                    data = "0x" + data.hex()
                elif isinstance(data, bytes):
                    data = "0x" + data.hex()
                
                oracle_address = log["address"]
                feed_name = CHAINLINK_PRICE_FEEDS.get(oracle_address, "Unknown Feed")
                
                oracle_updates.append({
                    "oracle": oracle_address,
                    "feed": feed_name,
                    "data": data,
                    "logIndex": log.get("logIndex"),
                })
        
        return oracle_updates
    except Exception as e:
        print(f"Warning: Could not fetch oracle updates for transaction {tx_hash}: {e}")
        return []


def get_swaps_in_block(w3: Web3, block_number: int) -> List[dict[str, Any]]:
    """
    Get all DEX swaps in a block using efficient log filtering.
    
    Args:
        w3: Web3 instance
        block_number: Block number to scan
        
    Returns:
        List of dictionaries with swap event details, including transaction hash
    """
    all_swaps = []
    
    try:
        # Use efficient log filtering instead of fetching receipts for each transaction
        swap_hashes = [UNISWAP_V2_SWAP_HASH, UNISWAP_V3_SWAP_HASH]
        
        for swap_hash in swap_hashes:
            try:
                # Build filter parameters
                filter_params = {
                    "fromBlock": block_number,
                    "toBlock": block_number,
                    "topics": [swap_hash],  # topics[0] is the event signature
                }
                
                # Get logs with retry logic
                logs = retry_with_backoff(lambda: w3.eth.get_logs(filter_params))
                
                # Process each log
                for log in logs:
                    topics = log.get("topics", [])
                    if not topics:
                        continue
                    
                    # Determine protocol based on event hash
                    protocol = "Uniswap V2" if swap_hash == UNISWAP_V2_SWAP_HASH else "Uniswap V3"
                    
                    # Extract transaction hash
                    tx_hash = log.get("transactionHash")
                    if hasattr(tx_hash, 'hex'):
                        tx_hash = "0x" + tx_hash.hex()
                    elif isinstance(tx_hash, bytes):
                        tx_hash = "0x" + tx_hash.hex()
                    
                    swap = {
                        "protocol": protocol,
                        "contract": log["address"],
                        "transactionHash": tx_hash,
                        "topics": [t.hex() if hasattr(t, 'hex') else str(t) for t in topics],
                        "data": log.get("data", "0x").hex() if hasattr(log.get("data", "0x"), 'hex') else str(log.get("data", "0x")),
                        "logIndex": log.get("logIndex"),
                        "blockNumber": log.get("blockNumber"),
                    }
                    all_swaps.append(swap)
                    
            except Exception as e:
                protocol = "Uniswap V2" if swap_hash == UNISWAP_V2_SWAP_HASH else "Uniswap V3"
                print(f"Warning: Could not fetch {protocol} swaps for block {block_number}: {e}")
                continue
                
    except Exception as e:
        print(f"Warning: Could not fetch swaps for block {block_number}: {e}")
    
    return all_swaps


def get_oracle_updates_in_block(w3: Web3, block_number: int) -> List[dict[str, Any]]:
    """
    Get all oracle price feed updates in a block using efficient log filtering.
    
    Args:
        w3: Web3 instance
        block_number: Block number to scan
        
    Returns:
        List of dictionaries with oracle update details, including transaction hash
    """
    all_oracle_updates = []
    
    try:
        # Use efficient log filtering instead of fetching receipts for each transaction
        filter_params = {
            "fromBlock": block_number,
            "toBlock": block_number,
            "topics": [CHAINLINK_ANSWER_UPDATED_HASH],  # topics[0] is the event signature
        }
        
        # Get logs with retry logic
        logs = retry_with_backoff(lambda: w3.eth.get_logs(filter_params))
        
        # Process each log
        for log in logs:
            topics = log.get("topics", [])
            if not topics:
                continue
            
            # Extract data
            data = log.get("data", "0x")
            if hasattr(data, 'hex'):
                data = "0x" + data.hex()
            elif isinstance(data, bytes):
                data = "0x" + data.hex()
            
            # Extract transaction hash
            tx_hash = log.get("transactionHash")
            if hasattr(tx_hash, 'hex'):
                tx_hash = "0x" + tx_hash.hex()
            elif isinstance(tx_hash, bytes):
                tx_hash = "0x" + tx_hash.hex()
            
            oracle_address = log["address"]
            feed_name = CHAINLINK_PRICE_FEEDS.get(oracle_address, "Unknown Feed")
            
            oracle_update = {
                "oracle": oracle_address,
                "feed": feed_name,
                "transactionHash": tx_hash,
                "data": data,
                "logIndex": log.get("logIndex"),
                "blockNumber": log.get("blockNumber"),
            }
            all_oracle_updates.append(oracle_update)
            
    except Exception as e:
        print(f"Warning: Could not fetch oracle updates for block {block_number}: {e}")
    
    return all_oracle_updates


def detect_oracle_manipulation(
    w3: Web3,
    liquidation_tx_hash: str,
    liquidation_data: dict[str, Any],
    block_number: int,
    flash_loans: Optional[List[dict[str, Any]]] = None,
    scan_block: bool = True,
) -> dict[str, Any]:
    """
    Detect oracle price manipulation indicators for a liquidation.
    Can scan the entire block or just the transaction.
    
    Args:
        w3: Web3 instance
        liquidation_tx_hash: Transaction hash of the liquidation
        liquidation_data: Dictionary with liquidation details
        block_number: Block number containing the liquidation
        flash_loans: Optional list of flash loans (from same block)
        scan_block: If True, scan entire block; if False, only scan transaction
        
    Returns:
        Dictionary with manipulation indicators and score
    """
    indicators = {
        "has_flash_loan": False,
        "has_large_swaps": False,
        "has_oracle_updates": False,
        "swap_details": [],
        "oracle_update_details": [],
        "flash_loan_details": [],
        "swapCount": 0,
        "oracleUpdateCount": 0,
        "flashLoanCount": 0,
        "manipulation_score": 0,
        "flags": [],
        "likely_manipulation": False,
        "scan_scope": "block" if scan_block else "transaction",
    }
    
    try:
        if scan_block:
            # Scan entire block for manipulation indicators
            
            # 1. Check for flash loans in block
            if flash_loans:
                block_flash_loans = [
                    fl for fl in flash_loans 
                    if fl.get("blockNumber") == block_number
                ]
                if block_flash_loans:
                    indicators["has_flash_loan"] = True
                    indicators["flash_loan_details"] = block_flash_loans
                    indicators["flashLoanCount"] = len(block_flash_loans)
                    
                    # Check if flash loan is in same transaction or different transaction
                    tx_flash_loans = [
                        fl for fl in block_flash_loans
                        if fl.get("transactionHash", "").lower() == liquidation_tx_hash.lower()
                    ]
                    other_tx_flash_loans = [
                        fl for fl in block_flash_loans
                        if fl.get("transactionHash", "").lower() != liquidation_tx_hash.lower()
                    ]
                    
                    if tx_flash_loans:
                        indicators["manipulation_score"] += 3
                        indicators["flags"].append(f"Flash loan in same transaction ({len(tx_flash_loans)} loan(s))")
                    if other_tx_flash_loans:
                        indicators["manipulation_score"] += 2
                        indicators["flags"].append(f"Flash loan in same block, different transaction ({len(other_tx_flash_loans)} loan(s))")
            
            # 2. Check for DEX swaps in block
            block_swaps = get_swaps_in_block(w3, block_number)
            if block_swaps:
                indicators["swap_details"] = block_swaps
                indicators["has_large_swaps"] = True
                indicators["swapCount"] = len(block_swaps)
                
                # Separate swaps in same transaction vs other transactions
                tx_swaps = [
                    swap for swap in block_swaps
                    if swap.get("transactionHash", "").lower() == liquidation_tx_hash.lower()
                ]
                other_tx_swaps = [
                    swap for swap in block_swaps
                    if swap.get("transactionHash", "").lower() != liquidation_tx_hash.lower()
                ]
                
                if tx_swaps:
                    indicators["manipulation_score"] += 2
                    indicators["flags"].append(f"{len(tx_swaps)} DEX swap(s) in same transaction")
                if other_tx_swaps:
                    indicators["manipulation_score"] += 1
                    indicators["flags"].append(f"{len(other_tx_swaps)} DEX swap(s) in same block, different transaction(s)")
            
            # 3. Check for oracle updates in block
            block_oracle_updates = get_oracle_updates_in_block(w3, block_number)
            if block_oracle_updates:
                indicators["has_oracle_updates"] = True
                indicators["oracle_update_details"] = block_oracle_updates
                indicators["oracleUpdateCount"] = len(block_oracle_updates)
                
                # Separate oracle updates in same transaction vs other transactions
                tx_oracle_updates = [
                    update for update in block_oracle_updates
                    if update.get("transactionHash", "").lower() == liquidation_tx_hash.lower()
                ]
                other_tx_oracle_updates = [
                    update for update in block_oracle_updates
                    if update.get("transactionHash", "").lower() != liquidation_tx_hash.lower()
                ]
                
                if tx_oracle_updates:
                    indicators["manipulation_score"] += 4  # Very suspicious!
                    indicators["flags"].append(f"{len(tx_oracle_updates)} oracle update(s) in same transaction")
                if other_tx_oracle_updates:
                    indicators["manipulation_score"] += 3  # Also very suspicious!
                    indicators["flags"].append(f"{len(other_tx_oracle_updates)} oracle update(s) in same block, different transaction(s)")
        
        else:
            # Original transaction-only scanning
            receipt = retry_with_backoff(lambda: w3.eth.get_transaction_receipt(liquidation_tx_hash))
            
            # 1. Check for flash loans in same transaction
            if flash_loans:
                tx_flash_loans = [
                    fl for fl in flash_loans 
                    if fl.get("transactionHash", "").lower() == liquidation_tx_hash.lower()
                ]
                if tx_flash_loans:
                    indicators["has_flash_loan"] = True
                    indicators["flash_loan_details"] = tx_flash_loans
                    indicators["flashLoanCount"] = len(tx_flash_loans)
                    indicators["manipulation_score"] += 3
                    indicators["flags"].append(f"Flash loan detected ({len(tx_flash_loans)} loan(s))")
            
            # 2. Check for DEX swaps in same transaction
            swaps = get_swaps_in_transaction(w3, liquidation_tx_hash)
            if swaps:
                indicators["swap_details"] = swaps
                indicators["has_large_swaps"] = True
                indicators["swapCount"] = len(swaps)
                indicators["manipulation_score"] += 2
                indicators["flags"].append(f"{len(swaps)} DEX swap(s) in same transaction")
            
            # 3. Check for oracle updates in same transaction
            oracle_updates = get_oracle_updates_in_transaction(w3, liquidation_tx_hash)
            if oracle_updates:
                indicators["has_oracle_updates"] = True
                indicators["oracle_update_details"] = oracle_updates
                indicators["oracleUpdateCount"] = len(oracle_updates)
                indicators["manipulation_score"] += 4
                indicators["flags"].append(f"{len(oracle_updates)} oracle price feed update(s) in same transaction")
        
        # Determine if manipulation is likely
        if indicators["manipulation_score"] >= 5:
            indicators["likely_manipulation"] = True
        else:
            indicators["likely_manipulation"] = False
        
    except Exception as e:
        print(f"Warning: Error analyzing transaction {liquidation_tx_hash}: {e}")
        indicators["flags"].append(f"Error during analysis: {e}")
    
    return indicators


def decode_chainlink_price_update(log: dict) -> Optional[dict[str, Any]]:
    """
    Decode Chainlink AnswerUpdated event to extract price information.
    
    AnswerUpdated(int256 current, uint256 updatedAt, uint256 roundId)
    
    Args:
        log: Event log dictionary with 'data' field
        
    Returns:
        Dictionary with price, timestamp, and roundId, or None if decoding fails
    """
    try:
        data = log.get("data", "0x")
        if hasattr(data, 'hex'):
            data = "0x" + data.hex()
        elif isinstance(data, bytes):
            data = "0x" + data.hex()
        
        if not data or data == "0x":
            return None
        
        # Remove 0x prefix and convert to bytes
        data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)
        
        if len(data_bytes) < 96:  # Need at least 3 * 32 bytes
            return None
        
        # Decode: current (int256, 32 bytes), updatedAt (uint256, 32 bytes), roundId (uint256, 32 bytes)
        current_price = int.from_bytes(data_bytes[0:32], byteorder="big", signed=True)
        updated_at = int.from_bytes(data_bytes[32:64], byteorder="big")
        round_id = int.from_bytes(data_bytes[64:96], byteorder="big")
        
        # Chainlink prices are typically scaled by 10^8
        price_usd = current_price / (10 ** 8) if current_price > 0 else None
        
        return {
            "price": current_price,
            "priceUSD": price_usd,
            "updatedAt": updated_at,
            "roundId": round_id,
        }
    except Exception as e:
        return None


def get_uniswap_pair_tokens(w3: Web3, pair_address: str, protocol: str = "V2", block_number: Optional[int] = None) -> Optional[dict[str, str]]:
    """
    Get token addresses from a Uniswap pair/pool contract.
    
    Args:
        w3: Web3 instance
        pair_address: Pair/pool contract address
        protocol: "V2" or "V3"
        block_number: Optional block number for historical queries
        
    Returns:
        Dictionary with token0 and token1 addresses, or None if query fails
    """
    try:
        pair_address = w3.to_checksum_address(pair_address)
        
        if protocol == "V2":
            # Uniswap V2 Pair: token0() and token1() functions
            # Function selectors: token0() = 0x0dfe1681, token1() = 0xd21220a7
            token0_selector = "0x0dfe1681"
            token1_selector = "0xd21220a7"
            
            block_param = {"block_identifier": block_number} if block_number else {}
            
            try:
                token0_result = retry_with_backoff(
                    lambda: w3.eth.call({
                        "to": pair_address,
                        "data": token0_selector
                    }, **block_param)
                )
                token1_result = retry_with_backoff(
                    lambda: w3.eth.call({
                        "to": pair_address,
                        "data": token1_selector
                    }, **block_param)
                )
                
                if token0_result and token1_result:
                    token0 = w3.to_checksum_address("0x" + token0_result.hex()[-40:])
                    token1 = w3.to_checksum_address("0x" + token1_result.hex()[-40:])
                    return {"token0": token0, "token1": token1}
            except Exception:
                pass
        
        elif protocol == "V3":
            # Uniswap V3 Pool: token0() and token1() functions (same selectors as V2)
            token0_selector = "0x0dfe1681"
            token1_selector = "0xd21220a7"
            
            block_param = {"block_identifier": block_number} if block_number else {}
            
            try:
                token0_result = retry_with_backoff(
                    lambda: w3.eth.call({
                        "to": pair_address,
                        "data": token0_selector
                    }, **block_param)
                )
                token1_result = retry_with_backoff(
                    lambda: w3.eth.call({
                        "to": pair_address,
                        "data": token1_selector
                    }, **block_param)
                )
                
                if token0_result and token1_result:
                    token0 = w3.to_checksum_address("0x" + token0_result.hex()[-40:])
                    token1 = w3.to_checksum_address("0x" + token1_result.hex()[-40:])
                    return {"token0": token0, "token1": token1}
            except Exception:
                pass
        
        return None
    except Exception as e:
        return None


def decode_uniswap_swap_detailed(log: dict, w3: Web3, protocol: str, block_number: Optional[int] = None) -> Optional[dict[str, Any]]:
    """
    Decode Uniswap swap event and extract token addresses by querying the pair contract.
    
    Args:
        log: Event log dictionary (can have "address" or "contract" key, topics as hex strings or HexBytes)
        w3: Web3 instance
        protocol: "Uniswap V2" or "Uniswap V3"
        block_number: Optional block number for historical queries
        
    Returns:
        Dictionary with decoded swap information including token addresses
    """
    try:
        topics = log.get("topics", [])
        if len(topics) < 3:
            return None
        
        # Handle both "address" and "contract" keys
        pair_address = log.get("address") or log.get("contract", "")
        if not pair_address:
            return None
        
        # Normalize pair address to checksum format
        try:
            pair_address = w3.to_checksum_address(pair_address)
        except Exception:
            return None
        
        # Get token addresses from pair contract
        protocol_type = "V2" if "V2" in protocol else "V3"
        try:
            tokens = get_uniswap_pair_tokens(w3, pair_address, protocol_type, block_number)
        except Exception as e:
            # If pair token query fails, return None (might not be a Uniswap pair or RPC error)
            return None
        
        if not tokens or "token0" not in tokens or "token1" not in tokens:
            return None
        
        # Extract data
        data = log.get("data", "0x")
        if hasattr(data, 'hex'):
            data = "0x" + data.hex()
        elif isinstance(data, bytes):
            data = "0x" + data.hex()
        
        # Extract transaction hash
        tx_hash = log.get("transactionHash", "")
        if hasattr(tx_hash, 'hex'):
            tx_hash = "0x" + tx_hash.hex()
        elif isinstance(tx_hash, bytes):
            tx_hash = "0x" + tx_hash.hex()
        
        # Extract sender/recipient from topics
        # Topics can be hex strings (from get_swaps_in_block) or HexBytes (from get_logs)
        def extract_address(topic):
            if isinstance(topic, str):
                # Remove 0x prefix if present
                hex_str = topic[2:] if topic.startswith('0x') else topic
            elif hasattr(topic, 'hex'):
                # HexBytes or bytes-like object with hex() method
                hex_str = topic.hex()
            elif isinstance(topic, bytes):
                # Raw bytes
                hex_str = topic.hex()
            else:
                # Fallback: try to convert to bytes then hex
                try:
                    hex_str = bytes(topic).hex()
                except Exception:
                    return None
            
            # Ensure we have at least 40 hex chars (20 bytes for address)
            if len(hex_str) < 40:
                return None
            
            # Extract last 40 chars (right-aligned address)
            address_hex = hex_str[-40:]
            try:
                return w3.to_checksum_address("0x" + address_hex)
            except Exception:
                return None
        
        if protocol == "Uniswap V2":
            sender = extract_address(topics[1]) if len(topics) > 1 else None
            to = extract_address(topics[2]) if len(topics) > 2 else None
            
            # Decode amounts from data
            if data and data != "0x":
                data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)
                if len(data_bytes) >= 128:  # 4 * 32 bytes
                    amount0_in = int.from_bytes(data_bytes[0:32], byteorder="big")
                    amount1_in = int.from_bytes(data_bytes[32:64], byteorder="big")
                    amount0_out = int.from_bytes(data_bytes[64:96], byteorder="big")
                    amount1_out = int.from_bytes(data_bytes[96:128], byteorder="big")
                else:
                    amount0_in = amount1_in = amount0_out = amount1_out = 0
            else:
                amount0_in = amount1_in = amount0_out = amount1_out = 0
            
            return {
                "protocol": protocol,
                "pair": pair_address,
                "token0": tokens["token0"],
                "token1": tokens["token1"],
                "sender": sender,
                "to": to,
                "amount0In": amount0_in,
                "amount1In": amount1_in,
                "amount0Out": amount0_out,
                "amount1Out": amount1_out,
                "transactionHash": tx_hash,
                "logIndex": log.get("logIndex"),
            }
        
        elif protocol == "Uniswap V3":
            sender = extract_address(topics[1]) if len(topics) > 1 else None
            recipient = extract_address(topics[2]) if len(topics) > 2 else None
            
            # Decode amounts from data
            if data and data != "0x":
                data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)
                if len(data_bytes) >= 64:  # At least amount0 and amount1
                    amount0 = int.from_bytes(data_bytes[0:32], byteorder="big", signed=True)
                    amount1 = int.from_bytes(data_bytes[32:64], byteorder="big", signed=True)
                else:
                    amount0 = amount1 = 0
            else:
                amount0 = amount1 = 0
            
            return {
                "protocol": protocol,
                "pool": pair_address,
                "token0": tokens["token0"],
                "token1": tokens["token1"],
                "sender": sender,
                "recipient": recipient,
                "amount0": amount0,
                "amount1": amount1,
                "transactionHash": tx_hash,
                "logIndex": log.get("logIndex"),
            }
        
        return None
    except Exception as e:
        return None


def detect_detailed_price_manipulation(
    w3: Web3,
    liquidation_tx_hash: str,
    liquidation_data: dict[str, Any],
    block_number: int,
    flash_loans: Optional[List[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Detailed price manipulation detection with price changes and token-specific swaps.
    
    Args:
        w3: Web3 instance
        liquidation_tx_hash: Transaction hash of the liquidation
        liquidation_data: Dictionary with liquidation details (debtAsset, collateralAsset, etc.)
        block_number: Block number containing the liquidation
        flash_loans: Optional list of flash loans (from same block)
        
    Returns:
        Dictionary with detailed manipulation indicators including:
        - Price changes from oracle updates
        - Token-specific swaps
        - Price impact calculations
        - Manipulation score
    """
    debt_asset = liquidation_data.get("debtAsset", "").lower()
    collateral_asset = liquidation_data.get("collateralAsset", "").lower()
    liquidator = liquidation_data.get("liquidator", "").lower()
    
    result = {
        "liquidationTxHash": liquidation_tx_hash,
        "blockNumber": block_number,
        "debtAsset": debt_asset,
        "collateralAsset": collateral_asset,
        "liquidator": liquidator,
        "flashLoans": [],
        "oraclePriceUpdates": [],
        "tokenSwaps": {
            "debtTokenSwaps": [],
            "collateralTokenSwaps": [],
            "allSwaps": [],
        },
        "priceChanges": {
            "debtToken": None,
            "collateralToken": None,
        },
        "manipulationScore": 0,
        "flags": [],
        "likelyManipulation": False,
    }
    
    # 1. Get flash loans in block and check for matches
    if flash_loans:
        block_flash_loans = [fl for fl in flash_loans if fl.get("blockNumber") == block_number]
        result["flashLoans"] = block_flash_loans
        
        for fl in block_flash_loans:
            fl_asset = fl.get("asset", "").lower()
            fl_initiator = fl.get("initiator", "").lower()
            
            token_match = (fl_asset == debt_asset or fl_asset == collateral_asset)
            address_match = (fl_initiator == liquidator)
            
            if token_match:
                result["manipulationScore"] += 2
                result["flags"].append(f"Flash loan for liquidation token ({fl_asset})")
            if address_match:
                result["manipulationScore"] += 1
                result["flags"].append(f"Flash loan initiator matches liquidator")
            if token_match and address_match:
                result["manipulationScore"] += 1  # Bonus for both
    
    # 2. Get oracle updates in block and decode prices
    oracle_updates = get_oracle_updates_in_block(w3, block_number)
    previous_prices = {}  # Track previous prices by oracle address
    
    for update in oracle_updates:
        # Decode price from the update
        decoded_price = decode_chainlink_price_update(update)
        if decoded_price:
            oracle_addr = update.get("oracle", "").lower()
            price_info = {
                **update,
                **decoded_price,
            }
            result["oraclePriceUpdates"].append(price_info)
            
            # Check if this oracle might be related to debt or collateral token
            # Note: We'd need a mapping of tokens to oracle addresses for exact matching
            # For now, flag any oracle update as potentially suspicious
            result["manipulationScore"] += 3
            result["flags"].append(f"Oracle price update: {update.get('feed', 'Unknown')} = ${decoded_price.get('priceUSD', 'N/A')}")
    
    # 3. Get swaps in block and filter by token
    swaps = get_swaps_in_block(w3, block_number)
    
    for swap in swaps:
        protocol = swap.get("protocol", "")
        if not protocol:
            continue
        
        # Convert swap dict from get_swaps_in_block to log format expected by decode_uniswap_swap_detailed
        # get_swaps_in_block returns: {"protocol", "contract", "transactionHash", "topics", "data", "logIndex", "blockNumber"}
        # decode_uniswap_swap_detailed expects: {"address", "topics", "data", ...}
        swap_log = {
            "address": swap.get("contract"),  # Convert "contract" to "address"
            "topics": swap.get("topics", []),
            "data": swap.get("data", "0x"),
            "logIndex": swap.get("logIndex"),
            "transactionHash": swap.get("transactionHash"),
            "blockNumber": swap.get("blockNumber"),
        }
        
        # Decode swap to get token addresses
        try:
            decoded_swap = decode_uniswap_swap_detailed(swap_log, w3, protocol, block_number)
            if decoded_swap:
                result["tokenSwaps"]["allSwaps"].append(decoded_swap)
                
                token0 = decoded_swap.get("token0", "").lower()
                token1 = decoded_swap.get("token1", "").lower()
                
                # Check if swap involves debt or collateral token
                involves_debt = (token0 == debt_asset or token1 == debt_asset)
                involves_collateral = (token0 == collateral_asset or token1 == collateral_asset)
                
                if involves_debt:
                    result["tokenSwaps"]["debtTokenSwaps"].append(decoded_swap)
                    result["manipulationScore"] += 2
                    result["flags"].append(f"Swap involving debt token ({debt_asset})")
                
                if involves_collateral:
                    result["tokenSwaps"]["collateralTokenSwaps"].append(decoded_swap)
                    result["manipulationScore"] += 2
                    result["flags"].append(f"Swap involving collateral token ({collateral_asset})")
                
                # Check if swap is in same transaction
                if decoded_swap.get("transactionHash", "").lower() == liquidation_tx_hash.lower():
                    result["manipulationScore"] += 1
                    result["flags"].append(f"Swap in same transaction as liquidation")
        except Exception as e:
            # Silently continue if decoding fails (might be non-Uniswap swap or pair query failed)
            # Add debug print if needed: print(f"Warning: Could not decode swap: {e}")
            continue
    
    # Calculate manipulation score
    if result["flashLoans"]:
        result["manipulationScore"] += 3
    if result["tokenSwaps"]["debtTokenSwaps"] or result["tokenSwaps"]["collateralTokenSwaps"]:
        result["manipulationScore"] += 2
    if result["oraclePriceUpdates"]:
        result["manipulationScore"] += 4
    
    result["likelyManipulation"] = result["manipulationScore"] >= 5
    
    return result


if __name__ == "__main__":
    exit(main())
