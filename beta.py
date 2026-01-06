import json
import ast # <--- TAMBAHAN PENTING
from typing import List, Union # <--- TAMBAHAN PENTING
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, register_function

# ==========================================
# 1. KONFIGURASI
# ==========================================
model = 'gpt-oss:20b'
llm_config = {
    "config_list": [
        {
            "model": model,
            "api_key": 'ollama',
            "base_url": 'http://localhost:11434/v1',
            "price": [0, 0],
        }
    ],
    "temperature": 0,
}

# ==========================================
# 2. TOOLS (DIPERBAIKI)
# ==========================================

def filter_events(genre: str = None, max_price: int = None, start_date: str = None, end_date: str = None) -> str:
    print(f"\n[SYSTEM] Filtering: Genre={genre}, Price={max_price}, Date={start_date}-{end_date}...")
    try:
        with open('events.json', 'r') as f:
            events = json.load(f)
        
        results = []
        for event in events:
            # Filter Genre
            if genre:
                g_in = genre.lower().split()
                g_db = event['genre'].lower()
                if not any(k in g_db for k in g_in):
                    continue
            # Filter Price
            if max_price is not None and event['price'] > max_price:
                continue
            # Filter Date
            event_date = event['date']
            if start_date and event_date < start_date:
                continue
            if end_date and event_date > end_date:
                continue
            
            results.append({"id": event['id'], "name": event['name']})
        
        if not results:
            return "No events found."
        
        return json.dumps(results)
    except Exception as e:
        return f"Error: {str(e)}"

# === FUNGSI INI KITA UBAH AGAR MENERIMA STRING JUGA ===
def check_seat_availability(event_ids: Union[List[int], str]) -> str:
    print(f"\n[SYSTEM] Checking Seats for IDs: {event_ids} (Type: {type(event_ids)})...")
    
    # 1. JIKA INPUT ADALAH STRING (Contoh: "[1, 6]"), UBAH JADI LIST
    if isinstance(event_ids, str):
        try:
            # ast.literal_eval aman mengubah string "[1, 6]" menjadi list [1, 6]
            event_ids = ast.literal_eval(event_ids)
        except:
            return "Error: event_ids format is invalid. Must be a list like [1, 2]."

    # 2. SEKARANG KITA YAKIN 'event_ids' ADALAH LIST
    if not isinstance(event_ids, list):
         return "Error: event_ids must be a list."

    with open('events.json', 'r') as f:
        all_events = json.load(f)
        
    results = []
    
    for eid in event_ids:
        original_event = next((e for e in all_events if e['id'] == eid), None)
        if original_event:
            status = original_event.get("seat_status", "Unknown")
            results.append({
                "event_name": original_event['name'],
                "date": original_event['date'],
                "price": original_event['price'],
                "location": original_event['location'],
                "seat_status": status
            })
            
    return json.dumps(results)

# ==========================================
# 3. AGENTS
# ==========================================

user = UserProxyAgent(
    name="UserProxyAgent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

preference_agent = AssistantAgent(
    name="PreferenceExtractorAgent",
    system_message="""
    Analyze the user's request.
    Extract: 'genre', 'max_price', 'start_date', 'end_date'.
    Convert dates to YYYY-MM-DD if needed.
    Just give the parameters without any explanations to 'EventDataAgent'.
    """,
    llm_config=llm_config
)

data_agent = AssistantAgent(
    name="EventDataAgent",
    system_message="""
    You have the tool 'filter_events'.
    Use the parameters provided by the previous speaker.
    Output ONLY the tool call (e.g., filter_events(genre='jazz')).
    Just give the results without any explainations.
    Then give the results to 'SeatAvailabilityAgent'.
    """,
    llm_config=llm_config
)

seat_agent = AssistantAgent(
    name="SeatAvailabilityAgent",
    system_message="""
    Role: Function Caller.
    Task: Extract event IDs from the previous list and call 'check_seat_availability'.
    
    You MUST output a tool call. Do not output JSON text or chat.

    EXAMPLE:
    Input: [{"id": 1, "name": "Jazz"}, {"id": 5, "name": "Pop"}]
    Output: check_seat_availability(event_ids=[1, 5])
    
    If input is "No events found", output: NO_EVENTS.
    """,
    llm_config=llm_config
)

writer_agent = AssistantAgent(
    name="EventRecommendationWriterAgent",
    system_message="""
    You are a helpful assistant.
    Summarize the event details (Name, Price, Location, Seat Status) provided by the previous step to a user that is looking to buy a ticket.
    And give explanations on why this/these events match the users request.
    Do it in indonesian.
    
    CRITICAL: You MUST end your response with the word: TERMINATE
    """,
    llm_config=llm_config
)

# ==========================================
# 4. REGISTER TOOLS
# ==========================================

register_function(filter_events, caller=data_agent, executor=user, name="filter_events", description="Filters events database.")
register_function(check_seat_availability, caller=seat_agent, executor=user, name="check_seat_availability", description="Checks seat availability.")

# ==========================================
# 5. CUSTOM FLOW CONTROL
# ==========================================

def custom_speaker_selection(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is user:
        if len(messages) < 2: return preference_agent
        last_agent_name = messages[-2]['name']
        
        if last_agent_name == "EventDataAgent": return seat_agent 
        elif last_agent_name == "SeatAvailabilityAgent": return writer_agent 
        elif last_agent_name == "EventRecommendationWriterAgent": return None
        else: return preference_agent

    if last_speaker is preference_agent: return data_agent
    if last_speaker is data_agent: return user 
    if last_speaker is seat_agent: return user 
    if last_speaker is writer_agent: return user 

    return "auto"

# ==========================================
# 6. EKSEKUSI
# ==========================================

groupchat = GroupChat(
    agents=[user, preference_agent, data_agent, seat_agent, writer_agent],
    messages=[],
    max_round=15,
    speaker_selection_method=custom_speaker_selection 
)

groupchat_mgr = GroupChatManager(groupchat, llm_config=llm_config)

if __name__ == "__main__":
    input_text = """
    Tolong carikan tiket konser. Prioritas utama saya jazz.
    Tanggalnya bebas antara 28 Oktober 2023 sampai 1 November 2023.
    Budget maksimal 500.000.
    """
    
    print(f"\n--- User: {input_text} ---\n")
    user.initiate_chat(groupchat_mgr, message=input_text)