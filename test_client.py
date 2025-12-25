import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

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
        "session_id": session_id,
        "message": message
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Agent Response:\n{data['response']}")
        
        if data.get('visualization_code'):
            print(f"\n📊 Visualization Code Generated:\n{data['visualization_code']}")
        
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
    # Test the API
    print("=== CSV Agent API Test ===\n")
    
    # 1. Upload a file
    print("1. Uploading file...")
    session_id = upload_file("data/Titanic-Dataset.csv")
    
    if not session_id:
        print("Failed to upload file. Exiting.")
        exit(1)
    
    # 2. Get session info
    print("\n2. Getting session info...")
    get_session_info(session_id)
    
    # 3. Send messages
    print("\n3. Chatting with agent...")
    
    # Simple query
    send_message(session_id, "Show me the first 5 rows of the dataset")
    
    # Analysis query
    send_message(session_id, "What is the survival rate by passenger class?")
    
    # Visualization query
    send_message(session_id, "Create a bar chart showing the number of survivors by gender")
    
    # 4. Delete session (optional)
    # print("\n4. Deleting session...")
    # delete_session(session_id)
    
    print("\n✅ Test completed!")