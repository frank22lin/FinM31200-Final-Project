#!/usr/bin/env python3
"""
AAVE Liquidations Data Downloader
=================================

A command-line tool for downloading AAVE protocol liquidation data from The Graph.
Supports multiple AAVE versions and networks via configurable subgraph endpoints.

AAVE Liquidation Overview
-------------------------
In AAVE, when a borrower's health factor drops below 1, their position becomes
eligible for liquidation. Liquidators can repay part of the debt and receive
the corresponding collateral plus a liquidation bonus.

Key liquidation fields:
- collateralAsset: The asset seized as collateral
- debtAsset: The asset being repaid (principal)
- debtToCover: Amount of debt being repaid
- liquidatedCollateralAmount: Amount of collateral seized
- liquidator: Address performing the liquidation
- user: Address being liquidated

Usage
-----
    # Download recent liquidations (default: AAVE V3 Ethereum)
    python aave_liquidations.py --query-type liquidations --limit 100

    # Download liquidations for a specific user
    python aave_liquidations.py --query-type liquidations --user 0x...

    # Download reserve data
    python aave_liquidations.py --query-type reserves

    # Use a different network
    python aave_liquidations.py --subgraph aave_v3_arbitrum --query-type liquidations

Requirements
------------
    pip install requests

Author: Generated with Claude Code
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AaveSubgraphConfig:
    """
    Configuration for an AAVE subgraph endpoint.

    Attributes:
        name: Human-readable name
        subgraph_id: The Graph subgraph ID
        description: Brief description
        version: AAVE version (v2, v3)
        network: Network name (ethereum, arbitrum, etc.)
    """
    name: str
    subgraph_id: str
    description: str
    version: str
    network: str


# Base URL for The Graph decentralized network
GRAPH_BASE_URL = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/"

# AAVE Subgraph configurations
AAVE_SUBGRAPHS = {
    "aave_v3_ethereum": AaveSubgraphConfig(
        name="AAVE V3 Ethereum",
        subgraph_id="9JLB7VbhJaGRtiFVvA6b4vDDwsfWF5rbY8Gd3zAUW1T7",
        description="AAVE V3 protocol on Ethereum mainnet - liquidations, reserves, users",
        version="v3",
        network="ethereum",
    ),
    # Note: Hr4ZdBkwkeENLSXwRLCPUQ1Xh5ep9S36dMz7PMcxwCp3 is an upgrades tracking subgraph,
    # not the main AAVE V2 protocol subgraph. Add more AAVE subgraphs here as needed.
}


# =============================================================================
# GRAPHQL QUERIES
# =============================================================================

QUERIES = {
    # Query liquidation events
    # Schema: id, user, reserve, collateralAsset, debtAsset, debtToCover,
    #         liquidatedCollateralAmount, profit, timestamp
    # Note: Reserve only has id and underlyingAsset fields
    # Note: profit field exists in schema but is always 0 (not populated by handler)
    "liquidations": """
    query GetLiquidations($first: Int!, $skip: Int!, $user: String, $startTime: BigInt, $endTime: BigInt) {
        liquidations(
            first: $first
            skip: $skip
            where: {
                user_: { id: $user }
                timestamp_gte: $startTime
                timestamp_lte: $endTime
            }
            orderBy: timestamp
            orderDirection: desc
        ) {
            id
            timestamp
            user {
                id
                healthFactor
            }
            reserve {
                id
                underlyingAsset
            }
            collateralAsset
            debtAsset
            debtToCover
            liquidatedCollateralAmount
            profit
        }
    }
    """,

    # Query liquidations without user filter
    "liquidations_all": """
    query GetLiquidationsAll($first: Int!, $skip: Int!, $startTime: BigInt, $endTime: BigInt) {
        liquidations(
            first: $first
            skip: $skip
            where: {
                timestamp_gte: $startTime
                timestamp_lte: $endTime
            }
            orderBy: timestamp
            orderDirection: desc
        ) {
            id
            timestamp
            user {
                id
                healthFactor
            }
            reserve {
                id
                underlyingAsset
            }
            collateralAsset
            debtAsset
            debtToCover
            liquidatedCollateralAmount
            profit
        }
    }
    """,

    # Query reserve (market) data
    "reserves": """
    query GetReserves($first: Int!, $skip: Int!) {
        reserves(
            first: $first
            skip: $skip
        ) {
            id
            underlyingAsset
        }
    }
    """,

}


# =============================================================================
# AAVE CLIENT
# =============================================================================

class AaveGraphClient:
    """
    Client for querying AAVE subgraphs on The Graph.

    Handles pagination, rate limiting, and error handling for AAVE-specific
    GraphQL queries.
    """

    def __init__(
        self,
        config: AaveSubgraphConfig,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize the AAVE Graph client.

        Args:
            config: Subgraph configuration
            api_key: The Graph API key
            rate_limit_delay: Delay between paginated requests
        """
        self.config = config
        self.api_key = api_key or os.environ.get("GRAPH_API_KEY")
        self.rate_limit_delay = rate_limit_delay

        if not self.api_key:
            raise ValueError(
                "API key required. Set GRAPH_API_KEY environment variable or pass api_key parameter.\n"
                "Get a free API key at https://thegraph.com/studio/"
            )

        self.url = GRAPH_BASE_URL.format(api_key=self.api_key) + config.subgraph_id

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query response data

        Raises:
            ValueError: If the response contains GraphQL errors
            requests.RequestException: If the HTTP request fails
        """
        payload = {
            "query": query,
            "variables": variables,
        }

        response = self.session.post(self.url, json=payload)
        response.raise_for_status()

        result = response.json()

        if "errors" in result:
            error_messages = [e.get("message", str(e)) for e in result["errors"]]
            raise ValueError(f"GraphQL errors: {'; '.join(error_messages)}")

        return result.get("data", {})

    def query_with_pagination(
        self,
        query_name: str,
        variables: dict[str, Any],
        entity_name: str,
        max_items: Optional[int] = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Execute a paginated query.

        Args:
            query_name: Key in QUERIES dict
            variables: Base query variables
            entity_name: Name of the entity list in response
            max_items: Maximum items to retrieve
            page_size: Items per page

        Returns:
            List of all retrieved entities
        """
        all_items = []
        skip = 0
        query = QUERIES[query_name]

        while True:
            if max_items is not None:
                remaining = max_items - len(all_items)
                if remaining <= 0:
                    break
                current_page_size = min(page_size, remaining)
            else:
                current_page_size = page_size

            page_variables = {**variables, "first": current_page_size, "skip": skip}

            print(f"  Fetching items {skip} to {skip + current_page_size}...")

            try:
                data = self.query(query, page_variables)
                items = data.get(entity_name, [])

                if not items:
                    break

                all_items.extend(items)
                skip += len(items)

                if len(items) < current_page_size:
                    break

                time.sleep(self.rate_limit_delay)

            except Exception as e:
                print(f"  Error during pagination: {e}")
                break

        return all_items


# =============================================================================
# DATA FORMATTING
# =============================================================================

def format_liquidation(liq: dict[str, Any]) -> dict[str, Any]:
    """
    Format a raw liquidation record.

    Schema fields:
    - id, timestamp, user (id, healthFactor), reserve (id, underlyingAsset)
    - collateralAsset, debtAsset (addresses as bytes)
    - debtToCover, liquidatedCollateralAmount, profit (BigInt values)

    Args:
        liq: Raw liquidation data from subgraph

    Returns:
        Formatted liquidation dictionary
    """
    timestamp = int(liq.get("timestamp", 0))
    reserve = liq.get("reserve", {}) or {}
    user = liq.get("user", {}) or {}

    # Parse amounts (BigInt values from subgraph)
    debt_to_cover = int(liq.get("debtToCover", 0))
    liquidated_collateral = int(liq.get("liquidatedCollateralAmount", 0))
    profit = int(liq.get("profit", 0))
    health_factor = int(user.get("healthFactor", 0))

    return {
        "id": liq.get("id"),
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
        "user": user.get("id"),
        "health_factor": health_factor,
        "reserve_id": reserve.get("id"),
        "reserve_underlying_asset": reserve.get("underlyingAsset"),
        "collateral_asset": liq.get("collateralAsset"),
        "debt_asset": liq.get("debtAsset"),
        "debt_to_cover": debt_to_cover,
        "liquidated_collateral_amount": liquidated_collateral,
        "profit": profit,
    }


def format_reserve(reserve: dict[str, Any]) -> dict[str, Any]:
    """
    Format a raw reserve record.

    Args:
        reserve: Raw reserve data from subgraph

    Returns:
        Formatted reserve dictionary
    """
    return {
        "id": reserve.get("id"),
        "underlying_asset": reserve.get("underlyingAsset"),
    }


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_liquidations(
    client: AaveGraphClient,
    limit: int = 100,
    user: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Download liquidation events.

    Args:
        client: AaveGraphClient instance
        limit: Maximum number of liquidations
        user: Optional user address to filter
        start_time: Optional start timestamp
        end_time: Optional end timestamp

    Returns:
        List of formatted liquidation records
    """
    effective_start = start_time if start_time is not None else 0
    effective_end = end_time if end_time is not None else 9999999999

    time_filter = ""
    if start_time is not None or end_time is not None:
        if start_time is not None and end_time is not None:
            time_filter = f" from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}"
        elif start_time is not None:
            time_filter = f" after {datetime.fromtimestamp(start_time)}"
        else:
            time_filter = f" before {datetime.fromtimestamp(end_time)}"

    if user:
        print(f"\nDownloading {limit} liquidations for user {user}{time_filter}...")
        query_name = "liquidations"
        variables = {
            "user": user.lower(),
            "startTime": str(effective_start),
            "endTime": str(effective_end),
        }
    else:
        print(f"\nDownloading {limit} recent liquidations{time_filter}...")
        query_name = "liquidations_all"
        variables = {
            "startTime": str(effective_start),
            "endTime": str(effective_end),
        }

    liquidations = client.query_with_pagination(
        query_name,
        variables,
        entity_name="liquidations",
        max_items=limit,
    )

    return [format_liquidation(l) for l in liquidations]


def download_reserves(
    client: AaveGraphClient,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Download reserve (market) data.

    Args:
        client: AaveGraphClient instance
        limit: Maximum number of reserves

    Returns:
        List of formatted reserve records
    """
    print(f"\nDownloading reserve data...")

    reserves = client.query_with_pagination(
        "reserves",
        {},
        entity_name="reserves",
        max_items=limit,
    )

    return [format_reserve(r) for r in reserves]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_timestamp(time_str: Optional[str]) -> Optional[int]:
    """
    Parse a time string into a Unix timestamp.

    Accepts:
    - Unix timestamp (integer string): "1704067200"
    - ISO date: "2024-01-01"
    - ISO datetime: "2024-01-01T12:00:00"
    """
    if time_str is None:
        return None

    try:
        return int(time_str)
    except ValueError:
        pass

    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            dt = datetime.strptime(time_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue

    raise ValueError(
        f"Cannot parse time '{time_str}'. Use Unix timestamp, YYYY-MM-DD, or YYYY-MM-DDTHH:MM:SS"
    )


def get_data_directory(subgraph: str, query_type: str, base_dir: Optional[str] = None) -> str:
    """
    Get the data directory path.

    Directory structure: data/aave/<subgraph>/<query_type>/

    Args:
        subgraph: Subgraph name
        query_type: Query type
        base_dir: Optional base directory override

    Returns:
        Path to the data directory
    """
    if base_dir:
        data_dir = os.path.join(base_dir, "aave", subgraph, query_type)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data", "aave", subgraph, query_type)

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def save_to_json(data: list[dict], filepath: str) -> None:
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {len(data)} records to {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download AAVE protocol liquidation data from The Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download recent liquidations (default: AAVE V3 Ethereum)
  python aave_liquidations.py --query-type liquidations --limit 100

  # Download liquidations for a specific user
  python aave_liquidations.py --query-type liquidations --user 0x1234...

  # Download reserve data
  python aave_liquidations.py --query-type reserves

  # Time-filtered liquidations
  python aave_liquidations.py --query-type liquidations --start-time 2024-01-01 --end-time 2024-02-01

  # Specify custom output directory (useful for testing)
  python aave_liquidations.py --query-type liquidations --output-dir /tmp/aave_test
        """,
    )

    parser.add_argument(
        "--subgraph",
        type=str,
        default="aave_v3_ethereum",
        choices=list(AAVE_SUBGRAPHS.keys()),
        help="Subgraph to query (default: aave_v3_ethereum)",
    )

    parser.add_argument(
        "--query-type",
        type=str,
        default="liquidations",
        choices=["liquidations", "reserves"],
        help="Type of data to download (default: liquidations)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records (default: 100)",
    )

    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Filter by user address (for liquidations, borrows)",
    )

    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start time filter (ISO format or Unix timestamp)",
    )

    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="End time filter (ISO format or Unix timestamp)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON filename (default: auto-generated)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: ./data)",
    )

    parser.add_argument(
        "--list-subgraphs",
        action="store_true",
        help="List available subgraph configurations",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The Graph API key (or set GRAPH_API_KEY env var)",
    )

    args = parser.parse_args()

    # Handle --list-subgraphs
    if args.list_subgraphs:
        print("\nAvailable AAVE Subgraphs:")
        print("-" * 60)
        for key, config in AAVE_SUBGRAPHS.items():
            print(f"\n  {key}")
            print(f"    Name: {config.name}")
            print(f"    Version: {config.version}")
            print(f"    Network: {config.network}")
            print(f"    Description: {config.description}")
            print(f"    Subgraph ID: {config.subgraph_id}")
        print()
        return

    # Initialize client
    config = AAVE_SUBGRAPHS[args.subgraph]
    print(f"\nConnecting to {config.name}...")

    try:
        client = AaveGraphClient(config, api_key=args.api_key)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    print(f"Endpoint: {client.url[:60]}...")

    # Parse time arguments
    try:
        start_time = parse_timestamp(args.start_time)
        end_time = parse_timestamp(args.end_time)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # Execute query
    if args.query_type == "liquidations":
        data = download_liquidations(
            client,
            limit=args.limit,
            user=args.user,
            start_time=start_time,
            end_time=end_time,
        )
    elif args.query_type == "reserves":
        data = download_reserves(client, limit=args.limit)
    else:
        print(f"Unknown query type: {args.query_type}")
        return

    # Print summary
    print(f"\nDownloaded {len(data)} {args.query_type}")

    if data:
        print("\nSample record:")
        print(json.dumps(data[0], indent=2, default=str))

    # Determine output path
    if args.output:
        output_file = args.output
    else:
        data_dir = get_data_directory(args.subgraph, args.query_type, base_dir=args.output_dir)
        output_file = os.path.join(data_dir, "data.json")

    save_to_json(data, output_file)

    print("\nDone!")


if __name__ == "__main__":
    main()
