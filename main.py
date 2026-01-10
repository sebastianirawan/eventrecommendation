import json
import ast
import threading
from typing import List, Union
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, register_function
import tkinter as tk
from tkinter import scrolledtext

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

def filter_events(genre: str = None, max_price: int = None, start_date: str = None, end_date: str = None) -> str:
    print(f"\n[SYSTEM] Filtering: Genre={genre}, Price={max_price}, Date={start_date}-{end_date}...")
    try:
        with open('events.json', 'r') as f:
            events = json.load(f)
        
        results = []
        for event in events:
            if genre:
                g_in = genre.lower().split()
                g_db = event['genre'].lower()
                if not any(k in g_db for k in g_in):
                    continue
            if max_price is not None and event['price'] > max_price:
                continue
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

def check_seat_availability(event_ids: Union[List[int], str]) -> str:
    print(f"\n[SYSTEM] Checking Seats for IDs: {event_ids} (Type: {type(event_ids)})...")
    
    if isinstance(event_ids, str):
        try:
            event_ids = ast.literal_eval(event_ids)
        except:
            return "Error: event_ids format is invalid. Must be a list like [1, 2]."

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

register_function(filter_events, caller=data_agent, executor=user, name="filter_events", description="Filters events database.")
register_function(check_seat_availability, caller=seat_agent, executor=user, name="check_seat_availability", description="Checks seat availability.")

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

groupchat = GroupChat(
    agents=[user, preference_agent, data_agent, seat_agent, writer_agent],
    messages=[],
    max_round=15,
    speaker_selection_method=custom_speaker_selection 
)

groupchat_mgr = GroupChatManager(groupchat, llm_config=llm_config)

#UI

def run_gui():
    root = tk.Tk()
    root.title("Event Recommendation")
    root.geometry("600x650")

    lbl_input = tk.Label(root, text="Prompt:", font=("Arial", 10, "bold"))
    lbl_input.pack(pady=(10, 5), anchor="w", padx=10)

    txt_input = scrolledtext.ScrolledText(root, height=5, width=70)
    txt_input.pack(padx=10, pady=5)
    
    default_prompt = "Tolong carikan tiket konser. Prioritas utama saya jazz. Tanggalnya bebas antara 28 Oktober 2023 sampai 1 November 2023. Budget maksimal 500.000."
    txt_input.insert(tk.END, default_prompt)

    lbl_output = tk.Label(root, text="Agent Recommendation:", font=("Arial", 10, "bold"))
    lbl_output.pack(pady=(20, 5), anchor="w", padx=10)

    txt_output = scrolledtext.ScrolledText(root, height=25, width=70, state='disabled')
    txt_output.pack(padx=10, pady=5)

    btn_submit = tk.Button(root, text="Submit")
    btn_submit.pack(pady=15)

    def update_ui_with_result(final_response):
        txt_output.config(state='normal')
        txt_output.delete("1.0", tk.END)
        txt_output.insert(tk.END, final_response)
        txt_output.config(state='disabled')
        
        btn_submit.config(state="normal", text="Submit")

    def run_chat_background(user_prompt):
        final_response = "Error: No recommendation found."
        try:
            chat_result = user.initiate_chat(groupchat_mgr, message=user_prompt)

            history = getattr(chat_result, 'chat_history', groupchat.messages)

            for msg in reversed(history):
                name = msg.get('name', '')
                if name == "EventRecommendationWriterAgent":
                    content = msg.get('content', '')
                    final_response = content.replace("TERMINATE", "").strip()
                    break
                    
        except Exception as e:
            final_response = f"An error occurred: {str(e)}"
            print(final_response)

        root.after(0, update_ui_with_result, final_response)

    def on_submit():
        user_prompt = txt_input.get("1.0", tk.END).strip()
        if not user_prompt:
            return

        btn_submit.config(state="disabled", text="Processing.... This may take a while.")
        t = threading.Thread(target=run_chat_background, args=(user_prompt,), daemon=True)
        t.start()

    btn_submit.config(command=on_submit)

    root.mainloop()

if __name__ == "__main__":
    run_gui()