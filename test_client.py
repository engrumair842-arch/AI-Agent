import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def create_general_session():
    """Create a general conversation session"""
    response = requests.post(f"{BASE_URL}/session/create")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ General session created!")
        print(f"Session ID: {data['session_id']}")
        print(f"Mode: {data['mode']}")
        return data['session_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def upload_file(file_path):
    """Upload a CSV file"""
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    if response.status_code == 200:
        print("✅ File uploaded successfully!")
        data = response.json()
        print(f"Session ID: {data['session_id']}")
        print(f"Filename: {data['filename']}")
        print(f"Columns: {data['columns']}")
        print(f"Row count: {data['row_count']}")
        return data['session_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def send_message(session_id, message):
    """Send a message to the agent"""
    payload = {
        "message": message
    }
    
    # Add session_id only if provided
    if session_id:
        payload["session_id"] = session_id
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Agent Response ({data['mode']} mode):\n{data['response']}")
        
        if data.get('visualization_code'):
            print(f"\n📊 Visualization Code Generated:\n{data['visualization_code']}")
        
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def send_message_without_session(message):
    """Send a message without a session (general mode)"""
    payload = {
        "message": message
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Agent Response ({data['mode']} mode):\n{data['response']}")
        print(f"\nNew Session ID: {data['session_id']}")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def get_session_info(session_id):
    """Get session information"""
    response = requests.get(f"{BASE_URL}/session/{session_id}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n📋 Session Info:")
        print(f"Filename: {data['filename']}")
        print(f"Columns: {data['columns']}")
        print(f"Row count: {data['row_count']}")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def delete_session(session_id):
    """Delete a session"""
    response = requests.delete(f"{BASE_URL}/session/{session_id}")
    
    if response.status_code == 200:
        print("✅ Session deleted successfully!")
    else:
        print(f"❌ Error: {response.text}")

# Example usage
if __name__ == "__main__":
    print("=== CSV Agent API Test - Option 1 (CSV Optional) ===\n")
    
    print("=" * 60)
    print("TEST 1: General Mode (No CSV uploaded)")
    print("=" * 60)
    
    # Test without uploading any file
    print("\n1. Chatting without uploading CSV file...")
    response = send_message_without_session("Hello! What can you help me with?")
    
    if response:
        general_session_id = response['session_id']
        
        # Continue conversation in general mode
        print("\n2. Asking a general question...")
        send_message(general_session_id, "What is the capital of France?")
        
        print("\n3. Asking about data analysis (should inform about upload)...")
        send_message(general_session_id, "Can you analyze some data for me?")
    
    print("\n" + "=" * 60)
    print("TEST 2: Data Analysis Mode (With CSV uploaded)")
    print("=" * 60)
    
    # Test with file upload
    print("\n4. Uploading CSV file...")
    data_session_id = upload_file("data/Titanic-Dataset.csv")
    
    if not data_session_id:
        print("Failed to upload file. Skipping data analysis tests.")
    else:
        # Get session info
        print("\n5. Getting session info...")
        get_session_info(data_session_id)
        
        # Test data analysis features
        print("\n6. Simple data query...")
        send_message(data_session_id, "Show me the first 5 rows of the dataset")
        
        print("\n7. Data analysis...")
        send_message(data_session_id, "What is the survival rate by passenger class?")
        
        print("\n8. Visualization request...")
        send_message(data_session_id, "Create a bar chart showing survivors by gender")
    
    print("\n" + "=" * 60)
    print("TEST 3: Creating Explicit General Session")
    print("=" * 60)
    
    print("\n9. Creating general session explicitly...")
    explicit_session_id = create_general_session()
    
    if explicit_session_id:
        print("\n10. Chatting in explicit general session...")
        send_message(explicit_session_id, "Tell me a joke about programming")
    
    # Cleanup (optional)
    print("\n" + "=" * 60)
    print("Cleanup (optional - uncomment to delete sessions)")
    print("=" * 60)
    # if data_session_id:
    #     print("\nDeleting data session...")
    #     delete_session(data_session_id)
    
    print("\n✅ All tests completed!")
    print("\n📝 Summary:")
    print("   - General mode: Works without CSV upload")
    print("   - Data analysis mode: Works with CSV upload")
    print("   - Both modes maintain conversation history")