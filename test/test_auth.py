import httpx
import json

API_KEY = "my_default_secret_key"  # Should match VALID_API_KEY in main.py
BASE_URL = "http://localhost:10002"

def test_get_agent_card():
    """Test getting the agent card with API key authentication"""
    print("=" * 60)
    print("Test 1: Getting Agent Card with Authentication")
    print("=" * 60)
    
    response = httpx.get(
        f"{BASE_URL}/.well-known/agent-card.json",
        headers={"X-API-Key": API_KEY}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        agent_card = response.json()
        print("\nAgent Card Information:")
        print(f"  Name: {agent_card.get('name')}")
        print(f"  Description: {agent_card.get('description')}")
        print(f"  Version: {agent_card.get('version')}")
        print(f"  URL: {agent_card.get('url')}")
        # Check both snake_case and camelCase for supports_authenticated_extended_card
        supports_extended = agent_card.get('supports_authenticated_extended_card') or agent_card.get('supportsAuthenticatedExtendedCard')
        print(f"  Supports Authenticated Extended Card: {supports_extended}")
        
        # Check both snake_case and camelCase for security_schemes
        security_schemes = agent_card.get('security_schemes') or agent_card.get('securitySchemes')
        if security_schemes:
            print("\nSecurity Schemes:")
            for scheme_name, scheme in security_schemes.items():
                print(f"  - {scheme_name}: {scheme.get('type')} (in: {scheme.get('in') or scheme.get('in_')})")
        
        security = agent_card.get('security')
        if security:
            print(f"\nSecurity Requirements: {security}")
        
        print("\nFull Agent Card JSON:")
        print(json.dumps(agent_card, indent=2))
        return agent_card
    else:
        print(f"Error: {response.text}")
        return None


def test_get_agent_card_without_auth():
    """Test getting the agent card without authentication (should fail)"""
    print("\n" + "=" * 60)
    print("Test 2: Getting Agent Card WITHOUT Authentication (Expected to fail)")
    print("=" * 60)
    
    try:
        response = httpx.get(
            f"{BASE_URL}/.well-known/agent-card.json"
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✓ Correctly rejected request without API key")
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
            except:
                print(f"Error message: {response.text}")
        else:
            print(f"⚠ Unexpected status code. Expected 401, got {response.status_code}")
            print(f"Response: {response.text}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("✓ Correctly rejected request without API key")
        else:
            print(f"Error: {e}")


def test_get_authenticated_extended_card():
    """Test getting the authenticated extended card"""
    print("\n" + "=" * 60)
    print("Test 3: Getting Authenticated Extended Card")
    print("=" * 60)
    
    # Create JSON-RPC request for agent/getAuthenticatedExtendedCard
    request_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "agent/getAuthenticatedExtendedCard"
    }
    
    response = httpx.post(
        f"{BASE_URL}/",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json=request_payload
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nResponse:")
        print(json.dumps(result, indent=2))
        
        if 'result' in result and 'result' in result['result']:
            extended_card = result['result']['result']
            print("\nAuthenticated Extended Card Information:")
            print(f"  Name: {extended_card.get('name')}")
            print(f"  Description: {extended_card.get('description')}")
            print(f"  Version: {extended_card.get('version')}")
        return result
    else:
        print(f"Error: {response.text}")
        return None


def test_invalid_api_key():
    """Test with invalid API key"""
    print("\n" + "=" * 60)
    print("Test 4: Using Invalid API Key (Expected to fail)")
    print("=" * 60)
    
    response = httpx.get(
        f"{BASE_URL}/.well-known/agent-card.json",
        headers={"X-API-Key": "invalid_key"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 401:
        print("✓ Correctly rejected request with invalid API key")
        try:
            error_detail = response.json()
            print(f"Error message: {error_detail}")
        except:
            print(f"Error message: {response.text}")
    else:
        print(f"⚠ Unexpected status code. Expected 401, got {response.status_code}")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("A2A Authentication Test Suite")
    print("=" * 60)
    
    # Run all tests
    agent_card = test_get_agent_card()
    test_get_agent_card_without_auth()
    
    # Only test extended card if agent supports it
    # Check both snake_case and camelCase
    supports_extended = False
    if agent_card:
        supports_extended = agent_card.get('supports_authenticated_extended_card') or agent_card.get('supportsAuthenticatedExtendedCard')
    
    if supports_extended:
        test_get_authenticated_extended_card()
    else:
        print("\n" + "=" * 60)
        print("Skipping Authenticated Extended Card test")
        print("(Agent does not support authenticated extended card)")
        print("=" * 60)
    
    test_invalid_api_key()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
