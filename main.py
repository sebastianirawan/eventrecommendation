import json
import random
from typing import List
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, register_function

model = 'llama3.2'
llm_config = {
    "model": model,
    "api_key": 'ollama',
    "base_url": 'http://localhost:11434/v1',
    "temperature": 0,
}

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding", 
        "use_docker": False
    }
)

preference_agent = AssistantAgent(
    name="PreferenceExtractorAgent",
    system_message="""You are an expert at understanding user intent.
    Analyze the user's input to extract: Genre, Favorite artists, Feasible date, Location, Budget, and Vibe. 
    Summarize these clearly for the EventDataAgent.""",
    llm_config=llm_config
)

data_agent = AssistantAgent(
    name="EventDataAgent",
    system_message="""You are responsible for finding events. 
    Use the tool 'filter_events' based on the preferences provided by the extractor.
    Once you get the list of events, show them and ask the SeatAvailabilityAgent to check tickets.""",
    llm_config=llm_config
)

seat_agent = AssistantAgent(
    name="SeatAvailabilityAgent",
    system_message="""You check ticket status. 
    Extract the 'id' from the events found by the EventDataAgent.
    Use the tool 'check_seat_availability' with the list of IDs.
    Output the status for each event.""",
    llm_config=llm_config
)

writer_agent = AssistantAgent(
    name="EventRecommendationWriterAgent",
    system_message="""You are the final recommender. 
    Based on the Event List and the Seat Availability provided by previous agents:
    1. Select the top 3 events that are NOT 'Sold Out'.
    2. Write a polite recommendation explaining why these fit the user's vibe and budget.
    3. Include the price and date.
    End your message with 'TERMINATE'.""",
    llm_config=llm_config
)

groupchat = GroupChat(
    agents=[user, preference_agent, data_agent, seat_agent, writer_agent],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"
)

groupchat_mgr = GroupChatManager(groupchat, llm_config=llm_config)

def filter_events(genre: str = None, max_price: int = None) -> str:
    try:
        with open('events.json', 'r') as f:
            events = json.load(f)
        
        results = []
        for event in events:
            if genre and genre.lower() not in event['genre'].lower():
                continue
            if max_price and event['price'] > max_price:
                continue
            results.append(event)
            
        if not results:
            return "No events found matching criteria."
        return json.dumps(results)
    except Exception as e:
        return f"Error reading database: {str(e)}"

def check_seat_availability(event_ids: List[int]) -> str:
    statuses = ["Available", "Limited Seats", "Sold Out"]
    results = {}
    for eid in event_ids:
        results[eid] = random.choice(statuses)
    return json.dumps(results)

register_function(
    filter_events,
    caller=data_agent,
    executor=user,
    name="filter_events",
    description="Filters events database by genre and max price."
)

register_function(
    check_seat_availability,
    caller=seat_agent,
    executor=user,
    name="check_seat_availability",
    description="Checks seat availability for specific event IDs."
)

def get_recommendation(user_prompt):
    print(f"\n--- Processing User Request: {user_prompt} ---\n")
    user.initiate_chat(
        groupchat_mgr,
        message=user_prompt
    )

if __name__ == "__main__":
    input_text = """
    Halo, tolong carikan saya konser musik Jazz untuk tanggal 28 Oktober 2023. 
    Saya sangat ingin menonton 'Sarah & The Band' jika ada. 
    Saya lebih suka lokasi di luar ruangan seperti taman (City Park) karena mencari vibe yang intimate. 
    Budget saya maksimal 500.000 rupiah.
    """
    
    get_recommendation(input_text)