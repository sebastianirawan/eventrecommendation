import autogen
import json
import os
import re
from typing import List, Optional, Union, Dict, Any


# setup and config
llm_config = {
    "config_list": [
        {
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ],
    "cache_seed": None, 
    "temperature": 0.1,
    "timeout": 120,
}

# mengubah format tanggal menjadi YYYY-MM-DD sesuai format di events.json
def normalize_date(date_input: str) -> Optional[str]:
    if not date_input or not isinstance(date_input, str): 
        return None
    
    # mapping nama bulan ke angka (MM)
    months = {
        "januari": "01", 
        "februari": "02", 
        "maret": "03", 
        "april": "04",
        "mei": "05", 
        "juni": "06", 
        "juli": "07", 
        "agustus": "08",
        "september": "09", 
        "oktober": "10", 
        "november": "11", 
        "desember": "12"
    }

    clean_date = date_input.lower().strip()

    # coba cari pola "tanggal, bulan, tahun" dengan menggunakan regex
    match = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", clean_date)

    if match:
        day, month_name, year = match.groups()
        month_num = months.get(month_name)
        if month_num: 
            return f"{year}-{month_num}-{day.zfill(2)}"
        
    # cek kalo format sudah benar (YYY-MM-DD)
    if re.match(r"\d{4}-\d{2}-\d{2}", clean_date): 
        return clean_date
    
    return None

# tool yang digunakan agen
# terima parameter dari LLM event_data_agent
# filter data dari events.json
# return berupa json
def filter_events(
    genre: Union[str, None] = None, 
    max_price: Union[str, int, None] = None, 
    date_str: Union[str, None] = None, 
    vibe: Union[str, None] = None, 
    keyword: Union[str, None] = None
) -> str:
    print(f"\n[TOOL LOG] Filtering with: Genre={genre}, Price={max_price}, Date={date_str}, Vibe={vibe}, Keyword={keyword}")
    try:
        with open("events.json", "r") as f: events = json.load(f)
    except FileNotFoundError: return "Error: events.json not found"

    # function untuk bersigin string "None" atau "Null" dari LLM
    def clean(val):
        if str(val).lower() in ["none", "null", ""]: 
            return None
        return val

    # bersihin input dari agent
    genre = clean(genre)
    vibe = clean(vibe)
    keyword = clean(keyword)

    # normalisasi tanggal kalau ada input tanggal
    target_date = normalize_date(str(date_str)) if clean(date_str) else None
    
    # parsing harga agar tidak ada "Rp", ".", dll.
    price_limit = float('inf')
    if max_price is not None:
        try:
            # hapus semua karakter kecuali angka
            price_limit = float(re.sub(r'[^\d]', '', str(max_price)))
        
        # kalau gagal parsing, biarkan default
        except: 
            pass

    # proses filtering data
    filtered = []
    for e in events:
        match = True

        # cek satu-satu kriteria
        if genre and genre.lower() not in e['genre'].lower(): 
            match = False
        if vibe and vibe.lower() not in e['vibe'].lower(): 
            match = False
        if e['price'] > price_limit: 
            match = False
        if target_date and e['date'] != target_date: 
            match = False

        # cek keyword seperti nama artis
        if keyword:
            if keyword.lower() not in e['artist'].lower() and keyword.lower() not in e['name'].lower(): 
                match = False
        
        # jika ada kriteria yang cocok, masukkan ke dalam list
        if match: 
            filtered.append(e)

    # jika tidak ditemukan event yang cocok
    if not filtered: 
        return "No events found matching criteria"
    
    return json.dumps(filtered)

# AGEN
# handle input user
# mengeksekusi tool
user_proxy = autogen.UserProxyAgent(
    name="UserProxyAgent",
    system_message="A human user.",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# ekstrak info preferensi dari agen user_proxy
# output berupa json
extractor = autogen.AssistantAgent(
    name="PreferenceExtractorAgent",
    system_message="""
    You are a parameter extractor.
    Extract: genre, max_price, date_str, vibe, keyword.
    
    IMPORTANT: 
    1. Output MUST be a single raw JSON object. NO text before or after.
    2. Set 'keyword' to null if the user mentions generic terms like "perempuan", "laki-laki", "band".
    
    Example: {"genre": "Rock", "max_price": 100000, "date_str": "2 Jan 2023", "vibe": null, "keyword": null}
    """,
    llm_config=llm_config
)

# terima json dari extractor
# panggil fungsi filter_events untuk filter events
event_data_agent = autogen.AssistantAgent(
    name="EventDataAgent",
    system_message="You are a function caller. Call `filter_events` with the parameters provided in the previous JSON.",
    llm_config=llm_config
)

# baca hasil search 
# tulis rekomendasi 
writer = autogen.AssistantAgent(
    name="EventRecommendationWriterAgent",
    system_message="""
    You are a recommender.
    Read the provided event list (which includes seat status).
    Explain why the vibe/artist fits.
    End with "TERMINATE".
    """,
    llm_config=llm_config
)

# register function agar bisa dipanggil agent
autogen.register_function(
    filter_events, 
    caller=event_data_agent, 
    executor=user_proxy, 
    name="filter_events", 
    description="Filters events"
    )

# alur komunikasi di set manual
# mengatur urutan siapa yang ngomong selanjutnya
# user_proxy -> extractor -> event_data_agent (tool) -> writer
def custom_flow(last_speaker: autogen.Agent, groupchat: autogen.GroupChat):
    messages = groupchat.messages
    last_content = messages[-1].get("content", "")
    
    # cek jika speaker terakhir adalah user_proxy
    if last_speaker is user_proxy:
        # jika user_proxy baru mulai (input) lanjut ke agen extractor
        if len(messages) < 2: 
            return extractor 
        
        # jika eror validasi dari tool, suruh ekstrak ulang
        if "Error" in last_content and "validation error" in last_content.lower():
            return extractor

        # cek siapa yang bicara sebelum user_proxy
        prev_speaker_name = messages[-2]["name"]
        
        # jika event_data_agent yang bicara untuk jalanin tool
        if prev_speaker_name == "EventDataAgent":
            # lanjut ke agen writer
            return writer
        
        else:
            return extractor

    # urutan standar/default
    elif last_speaker is extractor:
        return event_data_agent
        
    elif last_speaker is event_data_agent:
        return user_proxy

    elif last_speaker is writer:
        return None

    return None

# setup group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, extractor, event_data_agent, writer],
    messages=[],
    max_round=10,
    speaker_selection_method=custom_flow
)

manager = autogen.GroupChatManager(
    groupchat=groupchat, 
    llm_config=llm_config
)


# INPUT
# user_input = "Saya ingin konser jazz akhir pekan ini, kalau bisa artisnya perempuan dan suasananya intimate. Budget maksimal 500 ribu. Tanggal sekitar 28 Oktober 2023."
# user_input = """
#             Tolong carikan tiket konser. Prioritas utama saya jazz.
#             Tanggalnya bebas antara 28 Oktober 2023 sampai 1 November 2023.
#             Budget maksimal 500.000.
#             """
# user_input = """
#             I want a jazz concert ticket for around 28 of october 2023 to 1 november 2023 with a budget of 500000
#             """
user_input = "aku mau konser jazz"

user_proxy.initiate_chat(manager, message=user_input)