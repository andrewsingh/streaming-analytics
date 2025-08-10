import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import argparse
import webbrowser
from threading import Timer
from rapidfuzz import process, fuzz

# Set default plotly theme to dark
pio.templates.default = "plotly_dark"

# Constants for search and scoring
SEARCH_CONSTANTS = {
    'TRACK_WEIGHT': 1.0,
    'ARTIST_WEIGHT': 0.8,
    'ALBUM_WEIGHT': 0.6,
    'COMBINED_WEIGHT': 0.7,
    'DEFAULT_MIN_SCORE': 75,
    'SEARCH_MIN_SCORE': 70,
    'DEFAULT_SEARCH_LIMIT': 20,
    'TOP_SONGS_LIMIT': 50,
    'MAX_PLAY_BONUS': 5,
    'PLAY_BONUS_DIVISOR': 1000,
    'FLOAT_EPSILON': 1e-10,
    'SEASON_FORMAT_THRESHOLD': 18
}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)

app.title = "Spotify Analytics Dashboard"

# Global variables for data
df_global = None
entries_global = None
song_stats_global = None
artist_rankings_global = None
song_search_index = None
artist_list_global = None
artist_total_minutes_global = None
artist_daily_minutes_global = None

def smart_convert_to_datetime(timestamp: int) -> datetime:
    """Convert timestamp to datetime, handling both seconds and milliseconds."""
    if timestamp is None:
        return None
    # timestamp may be in milliseconds or seconds
    if timestamp < 10000000000:
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime.fromtimestamp(timestamp / 1000)

def load_entries(json_dir: Path) -> List[Dict]:
    """Load entries from JSON files in directory."""
    entries: List[Dict] = []
    for jf in sorted(json_dir.glob("*.json")):
        # only include audio history files
        if "Audio" not in jf.name:
            continue
        
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Skipping {jf}: JSON error {e}", file=sys.stderr)
            continue
        if isinstance(data, list):
            entries.extend(data)
    return entries

def clean_entries(entries: List[Dict]) -> List[Dict]:
    """Clean and standardize entry data."""
    cleaned_entries = []
    for entry in entries:
        if entry["spotify_track_uri"] is None:
            continue

        if entry["offline"]:
            timestamp = smart_convert_to_datetime(entry["offline_timestamp"])
        else:
            timestamp = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
        
        # strip timezone info
        timestamp = timestamp.replace(tzinfo=None)
        
        cleaned_entries.append({
            "timestamp": timestamp,
            "ms_played": entry["ms_played"],
            "track_id": entry["spotify_track_uri"].split(":")[-1],
            "track_name": entry["master_metadata_track_name"],
            "artist_name": entry["master_metadata_album_artist_name"],
            "album_name": entry["master_metadata_album_album_name"],
            "offline": entry["offline"],
            "platform": entry["platform"]
        })
    return cleaned_entries

def load_and_clean_entries(json_dir: str) -> List[Dict]:
    """Load and clean entries from JSON directory."""
    json_dir = Path(json_dir)
    entries = load_entries(json_dir)
    print(f"{len(entries)} entries loaded")
    entries = clean_entries(entries)
    print(f"{len(entries)} entries after cleaning")
    return entries

def precompute_artist_rankings(df: pd.DataFrame) -> Dict[str, Dict[str, any]]:
    """Precompute artist rankings and minutes data for different time periods."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    one_year_ago = now - timedelta(days=365)
    six_months_ago = now - timedelta(days=180)
    one_month_ago = now - timedelta(days=30)
    
    # Calculate total minutes per artist for different time periods
    df['minutes_played'] = df['ms_played'] / (1000 * 60)
    
    # All-time data
    all_time_minutes = df.groupby('artist_name')['minutes_played'].sum().sort_values(ascending=False)
    all_time_rankings = {artist: rank + 1 for rank, artist in enumerate(all_time_minutes.index)}
    all_time_minutes_dict = all_time_minutes.to_dict()
    
    # Past 1 year data
    df_1yr = df[df['timestamp'] >= one_year_ago]
    if len(df_1yr) > 0:
        one_year_minutes = df_1yr.groupby('artist_name')['minutes_played'].sum().sort_values(ascending=False)
        one_year_rankings = {artist: rank + 1 for rank, artist in enumerate(one_year_minutes.index)}
        one_year_minutes_dict = one_year_minutes.to_dict()
    else:
        one_year_rankings = {}
        one_year_minutes_dict = {}
    
    # Past 6 months data
    df_6mo = df[df['timestamp'] >= six_months_ago]
    if len(df_6mo) > 0:
        six_months_minutes = df_6mo.groupby('artist_name')['minutes_played'].sum().sort_values(ascending=False)
        six_months_rankings = {artist: rank + 1 for rank, artist in enumerate(six_months_minutes.index)}
        six_months_minutes_dict = six_months_minutes.to_dict()
    else:
        six_months_rankings = {}
        six_months_minutes_dict = {}
    
    # Past 1 month data
    df_1mo = df[df['timestamp'] >= one_month_ago]
    if len(df_1mo) > 0:
        one_month_minutes = df_1mo.groupby('artist_name')['minutes_played'].sum().sort_values(ascending=False)
        one_month_rankings = {artist: rank + 1 for rank, artist in enumerate(one_month_minutes.index)}
        one_month_minutes_dict = one_month_minutes.to_dict()
    else:
        one_month_rankings = {}
        one_month_minutes_dict = {}
    
    return {
        'all_time': {
            'rankings': all_time_rankings,
            'minutes': all_time_minutes_dict
        },
        'past_1_year': {
            'rankings': one_year_rankings,
            'minutes': one_year_minutes_dict
        },
        'past_6_months': {
            'rankings': six_months_rankings,
            'minutes': six_months_minutes_dict
        },
        'past_1_month': {
            'rankings': one_month_rankings,
            'minutes': one_month_minutes_dict
        }
    }

def precompute_song_stats(df: pd.DataFrame) -> Dict[tuple, Dict]:
    """Precompute detailed statistics for all songs."""
    print("Precomputing song statistics...")
    
    # Group by (track_name, artist_name) to merge duplicate songs with different URIs
    df['minutes_played'] = df['ms_played'] / (1000 * 60)
    df['listened_60s'] = df['ms_played'] >= 60000  # 60 seconds in milliseconds
    
    song_stats = {}
    
    # Group by song key
    for (track_name, artist_name), group in tqdm(df.groupby(['track_name', 'artist_name']), desc="Processing songs"):
        # Get album name (use the most common one if there are multiple)
        album_name = group['album_name'].mode().iloc[0] if len(group['album_name'].mode()) > 0 else group['album_name'].iloc[0]
        
        # Calculate basic stats
        stats = {
            'track_name': track_name,
            'artist_name': artist_name,
            'album_name': album_name,
            'first_streamed': group['timestamp'].min(),
            'last_streamed': group['timestamp'].max(),
            'total_streams': len(group),
            'streams_60s': group['listened_60s'].sum(),
            'total_minutes': group['minutes_played'].sum(),
        }
        
        # Calculate seasonal streaming data with consistent format
        # Determine format based on date range (same logic as other pages)
        min_year = group['timestamp'].min().year
        max_year = group['timestamp'].max().year
        seasons_count = len(generate_seasons(min_year, max_year, short_format=True))
        use_short_format = seasons_count > SEARCH_CONSTANTS['SEASON_FORMAT_THRESHOLD']
        
        group['season'] = group['timestamp'].apply(lambda x: get_season(x, short_format=use_short_format))
        seasonal_minutes = group.groupby('season')['minutes_played'].sum()
        stats['seasonal_data'] = seasonal_minutes.to_dict()
        
        song_stats[(track_name, artist_name)] = stats
    
    # Calculate percentile ranks
    all_minutes = [stats['total_minutes'] for stats in song_stats.values()]
    all_minutes_sorted = sorted(all_minutes)
    
    for song_key, stats in song_stats.items():
        # Calculate percentile rank
        song_minutes = stats['total_minutes']
        rank = sum(1 for minutes in all_minutes_sorted if minutes < song_minutes)
        percentile = (rank / len(all_minutes_sorted)) * 100
        stats['percentile_rank'] = percentile
    
    print(f"✅ Precomputed statistics for {len(song_stats)} unique songs")
    return song_stats

def fuzzy_search_songs(query: str, search_index: Dict, song_stats: Dict, limit: int = None, min_score: int = None) -> List[Tuple[Tuple[str, str], float, str]]:
    """
    Perform weighted fuzzy search on songs with field-specific scoring.
    
    Args:
        query: Search query string
        search_index: Pre-built search index with individual fields
        song_stats: Song statistics for additional ranking
        limit: Maximum number of results to return
        min_score: Minimum score threshold for matches
    
    Returns:
        List of tuples: (song_key, weighted_score, match_type)
    """
    # Use default values from constants if not provided
    if limit is None:
        limit = SEARCH_CONSTANTS['DEFAULT_SEARCH_LIMIT']
    if min_score is None:
        min_score = SEARCH_CONSTANTS['DEFAULT_MIN_SCORE']
        
    if not query or not search_index:
        return []
    
    results = []
    query_lower = query.lower().strip()
    
    for song_key, fields in search_index.items():
        if not fields:
            continue
            
        try:
            # Calculate weighted scores for each field using constants
            track_score = fuzz.WRatio(query_lower, fields['track_name'].lower()) * SEARCH_CONSTANTS['TRACK_WEIGHT']
            artist_score = fuzz.WRatio(query_lower, fields['artist_name'].lower()) * SEARCH_CONSTANTS['ARTIST_WEIGHT']
            album_score = fuzz.WRatio(query_lower, fields['album_name'].lower()) * SEARCH_CONSTANTS['ALBUM_WEIGHT']
            
            # Also try combined search for cross-field matches
            combined_score = fuzz.WRatio(query_lower, fields['combined'].lower()) * SEARCH_CONSTANTS['COMBINED_WEIGHT']
            
            # Take the best score across all fields
            best_score = max(track_score, artist_score, album_score, combined_score)
            
            # Determine match type for better UX feedback (using epsilon for float comparison)
            epsilon = SEARCH_CONSTANTS['FLOAT_EPSILON']
            if abs(track_score - best_score) < epsilon:
                match_type = "track"
            elif abs(artist_score - best_score) < epsilon:
                match_type = "artist"
            elif abs(album_score - best_score) < epsilon:
                match_type = "album"
            else:
                match_type = "combined"
            
            # Only include results above threshold
            if best_score >= min_score:
                # Add slight bonus for high-play-count songs to break ties intelligently
                play_bonus = min(
                    song_stats.get(song_key, {}).get('total_minutes', 0) / SEARCH_CONSTANTS['PLAY_BONUS_DIVISOR'], 
                    SEARCH_CONSTANTS['MAX_PLAY_BONUS']
                )
                final_score = best_score + play_bonus
                
                results.append((song_key, final_score, match_type))
                
        except Exception as e:
            # Skip problematic entries but don't crash the search
            print(f"Warning: Error processing song {song_key}: {e}")
            continue
    
    # Sort by score (descending) and limit results
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]

def fuzzy_search_artists(query: str, artists: List[str], limit: int = 20, min_score: int = 70) -> List[Tuple[str, float]]:
    """Simple fuzzy search for artist names using RapidFuzz WRatio."""
    if not query:
        return []
    q = query.lower().strip()
    results: List[Tuple[str, float]] = []
    for artist in artists:
        try:
            score = fuzz.WRatio(q, (artist or '').lower())
            if score >= min_score:
                results.append((artist, score))
        except Exception:
            continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]

def assign_artist_colors(artists):
    """
    Assign colors to artists based on their musical mood/tone using a color wheel approach.
    Colors reflect both genre grouping AND emotional intensity/atmosphere of the music.
    """
    
    color_mapping = {
        # GIRL POP / FEMALE ARTISTS - Red-Pink-Purple spectrum (intensity reflects mood)
        'Ariana Grande': '#FF1493',  # Deep pink - powerful, confident pop diva
        'Dua Lipa': '#FF69B4',  # Hot pink - fun, danceable pop
        'Sabrina Carpenter': '#FF6B9D',  # Bright pink - fun, sexy, youthful pop
        'Ava Max': '#FF20B2',  # Electric pink - bold, high-energy dance-pop
        'Camila Cabello': '#FF91A4',  # Coral pink - warm, Latin-influenced
        'Zara Larsson': '#FF1493',  # Deep pink - confident Swedish pop
        'Tove Lo': '#C71585',  # Medium violet red - edgy Swedish pop
        'Rita Ora': '#FF69B4',  # Hot pink - bold British pop
        'Beyoncé': '#8B008B',  # Dark magenta - powerful, regal
        'Rihanna': '#DC143C',  # Crimson - bold, iconic, edgy
        'Adele': '#800080',  # Purple - deep, emotional, soulful
        'Ella Mai': '#DDA0DD',  # Plum - smooth, sensual R&B
        'Billie Eilish': '#32CD32',  # Lime green - unique, edgy, alternative
        
        # MALE HIP HOP / RAP - Navy-Blue-Green spectrum (intensity reflects rap style)
        'Drake': '#4169E1',  # Royal blue - smooth, melodic, mainstream
        'Travis Scott': '#191970',  # Midnight blue - dark, atmospheric trap
        'Kendrick Lamar': '#0000CD',  # Medium blue - intense, conscious, powerful
        'J. Cole': '#2E8B57',  # Sea green - thoughtful, introspective
        'Future': '#483D8B',  # Dark slate blue - moody trap/mumble rap
        'Eminem': '#8B0000',  # Dark red - intense, aggressive, fiery
        '21 Savage': '#2F4F4F',  # Dark slate gray - dark, menacing trap
        'DaBaby': '#1E90FF',  # Dodger blue - energetic, bold rap
        'Lil Tecca': '#87CEEB',  # Sky blue - young, melodic, lighter rap
        'Jack Harlow': '#20B2AA',  # Light sea green - smooth, confident
        'Logic': '#008B8B',  # Dark cyan - technical, fast rap
        'Joey Bada$$': '#708090',  # Slate gray - boom bap, old school
        'Aminé': '#48D1CC',  # Medium turquoise - unique, colorful rap
        'Cordae': '#5F9EA0',  # Cadet blue - young conscious rap
        'JID': '#4682B4',  # Steel blue - technical but not overly bright
        'Buddy': '#87CEEB',  # Sky blue - West Coast, laid back
        'Denzel Curry': '#B22222',  # Fire brick - aggressive, versatile
        'Ski Mask The Slump God': '#32CD32',  # Lime green - fun, playful SoundCloud rap
        'Migos': '#6495ED',  # Cornflower blue - trap trio
        'Rae Sremmurd': '#FF4500',  # Orange red - party rap, high energy
        'Offset': '#4169E1',  # Royal blue - Migos member
        'Huncho Jack': '#483D8B',  # Dark slate blue - Travis/Quavo collab
        'Dreamville': '#2E8B57',  # Sea green - J. Cole collective
        
        # CLASSIC HIP HOP LEGENDS - Deeper, more serious colors
        '2Pac': '#B8860B',  # Dark goldenrod - legendary, golden era
        'The Notorious B.I.G.': '#191970',  # Midnight blue - East Coast legend
        'Mac Miller': '#708090',  # Slate gray - introspective, melancholy
        
        # EDM / DANCE / ELECTRONIC - Yellow-Orange spectrum (bright, energetic)
        'The Chainsmokers': '#FFD700',  # Gold - mainstream electronic dance-pop
        'FISHER': '#FFA500',  # Orange - high-energy house music
        'Flume': '#FF8C00',  # Dark orange - future bass, energetic
        'ODESZA': '#DEB887',  # Burlywood - melodic electronic, more chill
        'Tiësto': '#FF4500',  # Orange red - trance/EDM legend, intense
        
        # CHILL R&B / ALTERNATIVE - Earthy, muted tones (atmospheric)
        'The Weeknd': '#8B4513',  # Saddle brown - dark, moody, atmospheric R&B
        'Post Malone': '#CD853F',  # Peru - laid back, melodic crossover
        'Anderson .Paak': '#D2691E',  # Chocolate - smooth, funky, warm R&B
        'Metro Boomin': '#A0522D',  # Sienna - dark, atmospheric production
        
        # AMBIENT / CHILL - Muted earth tones
        'Lofi Fruits Music': '#D2B48C',  # Tan - chill, study music
    }
    
    # Generate colors for any artists not in the mapping
    remaining_artists = [artist for artist in artists if artist not in color_mapping]
    
    # Use a diverse color palette for any remaining artists
    additional_colors = [
        '#FF1493', '#FF69B4', '#DA70D6', '#BA55D3', '#9370DB',
        '#4169E1', '#1E90FF', '#00CED1', '#20B2AA', '#008B8B',
        '#FFD700', '#FFA500', '#FF8C00', '#FFFF00', '#FF4500',
        '#CD853F', '#D2691E', '#A0522D', '#8B4513', '#DEB887',
        '#32CD32', '#228B22', '#6B8E23', '#9ACD32', '#ADFF2F'
    ]
    
    # Assign colors to remaining artists
    for i, artist in enumerate(remaining_artists):
        if i < len(additional_colors):
            color_mapping[artist] = additional_colors[i]
        else:
            color_mapping[artist] = additional_colors[i % len(additional_colors)]
    
    return color_mapping

def get_season(date, short_format=False):
    """Get season label for a date."""
    year = date.year
    month = date.month
    if short_format:
        if month <= 6:
            return f"S{year%100:02d}"  # Spring (first half)
        else:
            return f"F{year%100:02d}"  # Fall (second half)
    else:
        if month <= 6:
            return f"Spring<br>{year}"  # Spring (first half) with line break
        else:
            return f"Fall<br>{year}"  # Fall (second half) with line break

def generate_seasons(start_year=2016, end_year=2025, short_format=False):
    """Generate chronologically ordered seasons."""
    seasons = []
    
    if short_format:
        # Start with F16 (July-Dec 2016)
        seasons.append(f'F{start_year%100:02d}')
        
        # For years 2017-2024, add Spring first (Jan-June), then Fall (July-Dec)
        for year in range(start_year + 1, end_year):
            seasons.append(f"S{year%100:02d}")
            seasons.append(f"F{year%100:02d}")
        
        # For the final year, only add Spring
        seasons.append(f'S{end_year%100:02d}')
    else:
        # Start with Fall 2016 (July-Dec 2016)
        seasons.append(f'Fall<br>{start_year}')
        
        # For years 2017-2024, add Spring first (Jan-June), then Fall (July-Dec)
        for year in range(start_year + 1, end_year):
            seasons.append(f"Spring<br>{year}")
            seasons.append(f"Fall<br>{year}")
        
        # For the final year, only add Spring
        seasons.append(f'Spring<br>{end_year}')
    
    return seasons

def parse_season_for_sorting(season_str: str) -> tuple:
    """
    Parse season string and return tuple for chronological sorting.
    Returns (year, season_order) where season_order is 0 for Spring, 1 for Fall.
    """
    try:
        if '<br>' in season_str:
            # Long format: "Spring<br>2023" or "Fall<br>2023"
            season_type, year_str = season_str.split('<br>')
            year = int(year_str)
            season_order = 0 if season_type == 'Spring' else 1
        else:
            # Short format: "S23" or "F23"
            season_type = season_str[0]
            year_suffix = season_str[1:]
            year = 2000 + int(year_suffix)  # Convert "23" to 2023
            season_order = 0 if season_type == 'S' else 1
        
        return (year, season_order)
    except:
        # Fallback: return a high number to sort unknown formats last
        return (9999, 9999)

def generate_complete_season_range(first_date, last_date, short_format=False):
    """
    Generate all seasons between first and last date, inclusive.
    """
    first_season = get_season(first_date, short_format)
    last_season = get_season(last_date, short_format)
    
    # Get the year range
    start_year = first_date.year
    end_year = last_date.year
    
    # Generate all possible seasons in this range
    all_seasons = []
    
    for year in range(start_year, end_year + 1):
        if short_format:
            spring_season = f"S{year%100:02d}"
            fall_season = f"F{year%100:02d}"
        else:
            spring_season = f"Spring<br>{year}"
            fall_season = f"Fall<br>{year}"
        
        # Add spring season (Jan-June)
        if year > start_year or first_date.month <= 6:
            all_seasons.append(spring_season)
        
        # Add fall season (July-Dec)  
        if year < end_year or last_date.month > 6:
            all_seasons.append(fall_season)
    
    # Filter to only include seasons between first and last (inclusive)
    first_sort_key = parse_season_for_sorting(first_season)
    last_sort_key = parse_season_for_sorting(last_season)
    
    filtered_seasons = []
    for season in all_seasons:
        season_key = parse_season_for_sorting(season)
        if first_sort_key <= season_key <= last_sort_key:
            filtered_seasons.append(season)
    
    return filtered_seasons

# CSS styling is now handled by assets/custom.css which Dash loads automatically

# Navigation bar
navbar = dbc.NavbarSimple(
    brand="🎵 Spotify Analytics Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-4",
    style={"backgroundColor": "#1a252f"}
)

# Sidebar for navigation
sidebar = html.Div([
    html.Div([
        html.Img(src="/assets/spotify_logo.png", style={"width": "40px", "marginRight": "10px"}),
        html.H3("Analytics", className="text-light"),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),
    dbc.Nav([
        dbc.NavLink("🏠 Overview", href="/", active="exact", className="text-light"),
        dbc.NavLink("📈 Daily Streaming", href="/daily-streaming", active="exact", className="text-light"),
        dbc.NavLink("🎤 Top Artists", href="/top-artists", active="exact", className="text-light"),
        dbc.NavLink("🎵 Top Songs", href="/top-songs", active="exact", className="text-light"),
        dbc.NavLink("👑 Artists by Minutes", href="/artists-by-minutes", active="exact", className="text-light"),
        dbc.NavLink("⏱️ Artist Cumulative", href="/artist-cumulative", active="exact", className="text-light"),
        dbc.NavLink("🔍 Song Details", href="/song-details", active="exact", className="text-light"),
        dbc.NavLink("📊 Coming Soon...", href="/coming-soon", active="exact", className="text-light disabled"),
    ], vertical=True, pills=True)
], id="sidebar", style={
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "250px",
    "backgroundColor": "#1a252f",
    "padding": "20px",
    "zIndex": "1000"
})

# Main content area
content = html.Div(id="page-content", style={
    "marginLeft": "250px",
    "padding": "20px 40px",
    "backgroundColor": "#0e1117",
    "minHeight": "100vh"
})

# App layout
app.layout = html.Div([
    dcc.Location(id="url"),
    # navbar, # Navbar is removed, integrated into sidebar/content structure
    sidebar,
    content
], style={"fontFamily": "Verdana, sans-serif"})

# Overview Page
def overview_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    
    # Calculate some basic stats
    total_tracks = len(df_global)
    total_artists = df_global['artist_name'].nunique()
    total_hours = df_global['ms_played'].sum() / (1000 * 60 * 60)
    date_range = f"{df_global['timestamp'].min().strftime('%Y-%m-%d')} to {df_global['timestamp'].max().strftime('%Y-%m-%d')}"
    
    return html.Div([
        html.H2("🎵 Spotify Analytics Dashboard", className="mb-4"),
        html.P("Explore your music listening patterns through interactive visualizations.", 
               className="mb-4"),
        
        # Stats cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_tracks:,}", className="text-primary"),
                        html.P("Total Tracks Played", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_artists:,}", className="text-primary"),
                        html.P("Unique Artists", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_hours:,.0f}", className="text-primary"),
                        html.P("Hours Listened", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📅", className="text-primary"),
                        html.P(date_range, className="text-light small")
                    ])
                ], className="custom-card text-center")
            ], width=3),
        ], className="mb-4"),
        
        # Quick insights
        html.Div([
            html.H4("🚀 Quick Start", className="text-light mb-3"),
            html.P("Use the sidebar to navigate between different visualizations:"),
            html.Ul([
                html.Li("📈 Daily Streaming - View your daily listening patterns with EMA smoothing"),
                html.Li("🎤 Top Artists - See your favorite artists over time"),
                html.Li("🎵 Top Songs - Explore individual tracks by season"),
                html.Li("👑 Artists by Minutes - Rankings and charts of your top artists by listening time"),
                html.Li("🔍 Song Details - Search for any song and view detailed streaming statistics"),
                html.Li("📊 More visualizations coming soon!"),
            ])
        ])
    ])

# Top Artists Page
def top_artists_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    
    min_year = df_global['timestamp'].min().year
    max_year = df_global['timestamp'].max().year
    
    return html.Div([
        html.H2("🎤 Top Artists by Season", className="mb-4"),
        
        # Controls
        dbc.Card([
            dbc.CardBody([
                html.Label("Number of Top Artists:"),
                dcc.Slider(
                    id="top-n-artists-slider",
                    min=3,
                    max=15,
                    step=1,
                    value=6,
                    marks={i: str(i) for i in range(3, 16, 2)},
                    className="custom-slider mb-3"
                )
            ]),
            dbc.CardBody([
                html.Label("Date Range:"),
                dcc.RangeSlider(
                    id="date-range-slider-artists",
                    min=min_year,
                    max=max_year,
                    step=1,
                    value=[min_year, max_year],
                    marks={i: str(i) for i in range(min_year, max_year + 1, 2)},
                    className="custom-slider mb-3"
                )
            ]),
        ], className="mb-4"),
        
        # Chart
        dcc.Loading(
            id="loading-artists-chart",
            type="circle",
            children=dcc.Graph(id="top-artists-chart", style={"height": "800px"})
        ),
        
        # Stats
        html.Div(id="artists-stats", className="mt-4 stats-container")
    ])

# Top Songs Page
def top_songs_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    
    min_year = df_global['timestamp'].min().year
    max_year = df_global['timestamp'].max().year
    
    return html.Div([
        html.H2("🎵 Top Songs by Season", className="mb-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Number of Top Songs:"),
                dcc.Slider(
                    id="top-n-songs-slider",
                    min=3,
                    max=15,
                    step=1,
                    value=6,
                    marks={i: str(i) for i in range(3, 16, 2)},
                    className="custom-slider mb-3"
                )
            ], width=6),
            dbc.Col([
                html.Label("Date Range:"),
                dcc.RangeSlider(
                    id="date-range-slider-songs",
                    min=min_year,
                    max=max_year,
                    step=1,
                    value=[min_year, max_year],
                    marks={i: str(i) for i in range(min_year, max_year + 1, 2)},
                    className="custom-slider mb-3"
                )
            ], width=6),
        ], className="mb-4"),
        
        # Chart
        dcc.Loading(
            id="loading-songs-chart",
            type="circle",
            children=dcc.Graph(id="top-songs-chart", style={"height": "800px"})
        ),
        
        # Stats
        html.Div(id="songs-stats", className="mt-4 stats-container")
    ])

# Daily Streaming Page
def daily_streaming_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    
    min_year = df_global['timestamp'].min().year
    max_year = df_global['timestamp'].max().year
    
    return html.Div([
        html.H2("📈 Daily Streaming Minutes", className="mb-4"),
        html.P("Explore your daily listening patterns with exponential moving averages (EMA) to smooth out trends.", 
               className="mb-4"),
        
        # Controls
        dbc.Card([
            dbc.CardBody([
                html.Label("Exponential Moving Average (EMA) Duration:"),
                dcc.Slider(
                    id="ema-duration-slider",
                    min=0,
                    max=4,
                    step=1,
                    value=1,  # Default to 30 days
                    marks={
                        0: "No EMA",
                        1: "30 days",
                        2: "60 days", 
                        3: "90 days",
                        4: "120 days"
                    },
                    className="custom-slider mb-3"
                )
            ]),
            dbc.CardBody([
                html.Label("Date Range:"),
                dcc.RangeSlider(
                    id="date-range-slider-daily",
                    min=min_year,
                    max=max_year,
                    step=1,
                    value=[min_year, max_year],
                    marks={i: str(i) for i in range(min_year, max_year + 1, 2)},
                    className="custom-slider mb-3"
                )
            ]),
        ], className="mb-4"),
        
        # Chart
        dcc.Loading(
            id="loading-daily-chart",
            type="circle",
            children=dcc.Graph(id="daily-streaming-chart", style={"height": "600px"})
        ),
        
        # Stats
        html.Div(id="daily-stats", className="mt-4 stats-container")
    ])

# Song Details Page
def song_details_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    
    if song_stats_global is None:
        return html.Div([
            dbc.Alert("Song statistics not loaded. Please restart the application.", color="danger")
        ])
        
    return html.Div([
        html.H2("🔍 Song Details", className="mb-4"),
        html.P("Search for any song in your library to see detailed streaming statistics.", className="mb-4"),
        
        # Search section with improved fuzzy search
        dbc.Card([
            dbc.CardBody([
                html.H4("🔎 Search for a Song", className="mb-3"),
                html.P([
                    "Search by song title, artist, or album name.",
                    html.Br(),
                    html.Small("Icons indicate match type: 🎵 song title, 🎤 artist name, 💿 album name, 🔍 multiple fields", className="text-muted")
                ], className="mb-3"),
                html.Label("Search your music library:"),
                dcc.Dropdown(
                    id="song-search-dropdown",
                    placeholder="Try: 'drake hotline', 'bohemian rhap', 'taylor swift'...",
                    options=[],  # Will be populated by callback
                    value=None,
                    searchable=True,
                    className="mb-3"
                ),
            ])
        ], className="mb-4"),
        
        # Song details section
        html.Div(id="song-details-content")
    ])

# Artists by Minutes Page
def artists_by_minutes_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    
    if artist_rankings_global is None:
        return html.Div([
            dbc.Alert("Artist statistics not loaded. Please restart the application.", color="danger")
        ])
        
    return html.Div([
        html.H2("👑 Top Artists by Minutes Streamed", className="mb-4"),
        html.P("View your top artists ranked by total listening time across different periods.", className="mb-4"),
        
        # Time period selector
        dbc.Card([
            dbc.CardBody([
                html.Label("Select Time Period:", className="mb-3"),
                dbc.RadioItems(
                    id="time-period-selector",
                    options=[
                        {"label": "All Time", "value": "all_time"},
                        {"label": "Past 1 Year", "value": "past_1_year"},
                        {"label": "Past 6 Months", "value": "past_6_months"},
                        {"label": "Past 1 Month", "value": "past_1_month"}
                    ],
                    value="all_time",
                    inline=True,
                    className="mb-3"
                )
            ])
        ], className="mb-4"),
        
        # Chart and table will be populated by callback
        dcc.Loading(
            id="loading-artists-by-minutes",
            type="circle",
            children=[
                html.Div(id="artists-chart-container"),
                html.Div(id="artists-table-container", className="mt-4")
            ]
        )
    ])

# Artist Cumulative Page
def artist_cumulative_layout():
    if df_global is None:
        return html.Div([
            dbc.Alert("No data loaded. Please run the app with --data argument.", color="danger")
        ])
    # Date bounds for picker
    min_date = df_global['date'].min()
    max_date = df_global['date'].max()
    return html.Div([
        html.H2("⏱️ Cumulative Minutes by Artist", className="mb-4"),
        html.P("Add artists to compare cumulative minutes listened over time. Use the list to show/hide or remove artists.", className="mb-4"),
        dcc.Store(id="selected-artists-store", data={"selected": [], "visible": []}),
        dbc.Card([
            dbc.CardBody([
                html.Label("Search and add an artist:"),
                dcc.Dropdown(
                    id="artist-search-dropdown",
                    placeholder="Type an artist name (e.g., 'dua', 'cole', 'drake')",
                    options=[],
                    value=None,
                    searchable=True,
                    className="mb-3"
                )
            ])
        ], className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Label("Start date (x-axis):"),
                dcc.DatePickerSingle(
                    id="artist-cum-start-date",
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    date=min_date,
                    display_format='YYYY-MM-DD',
                    className="mb-2"
                ),
                html.Div(
                    html.Small(f"Data range: {min_date} to {max_date}", className="text-muted"),
                    className="text-muted"
                )
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Artists in Graph", className="mb-3"),
                        dcc.Checklist(id="artist-visibility-checklist", options=[], value=[], className="mb-3"),
                        dbc.ButtonGroup([
                            dbc.Button("Show All", id="btn-show-all-artists", size="sm", color="secondary"),
                            dbc.Button("Hide All", id="btn-hide-all-artists", size="sm", color="secondary"),
                            dbc.Button("Remove Hidden", id="btn-remove-hidden-artists", size="sm", color="danger")
                        ])
                    ])
                ])
            ], width=4),
            dbc.Col([
                dcc.Loading(
                    id="loading-artist-cum",
                    type="circle",
                    children=dcc.Graph(id="artist-cumulative-graph", style={"height": "650px"})
                )
            ], width=8)
        ])
    ])

# Callback for page routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/daily-streaming":
        return daily_streaming_layout()
    elif pathname == "/top-artists":
        return top_artists_layout()
    elif pathname == "/top-songs":
        return top_songs_layout()
    elif pathname == "/artists-by-minutes":
        return artists_by_minutes_layout()
    elif pathname == "/artist-cumulative":
        return artist_cumulative_layout()
    elif pathname == "/song-details":
        return song_details_layout()
    elif pathname == "/coming-soon":
        return html.Div([
            html.H2("🚧 Coming Soon!", className="text-light text-center"),
            html.P("More exciting visualizations are on the way!", className="text-light text-center")
        ])
    else:
        return overview_layout()

# Callback for top artists chart
@app.callback(
    [Output("top-artists-chart", "figure"),
     Output("artists-stats", "children")],
    [Input("top-n-artists-slider", "value"),
     Input("date-range-slider-artists", "value")]
)
def update_artists_chart(top_n, date_range):
    if df_global is None:
        return go.Figure(), html.Div()
    
    # Filter data by date range
    start_date = f"{date_range[0]}-01-01"
    end_date = f"{date_range[1]}-12-31"
    df_filtered = df_global[(df_global['timestamp'] >= start_date) & 
                           (df_global['timestamp'] <= end_date)].copy()
    
    if len(df_filtered) == 0:
        return go.Figure().update_layout(
            title="No data available for the selected date range.",
            template='plotly_dark',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
        ), html.Div([
            dbc.Alert("No data available for the selected date range.", color="warning")
        ])
    
    # Generate seasons for the date range to determine format
    seasons = generate_seasons(date_range[0], date_range[1], short_format=True)  # Get count first
    use_short_format = len(seasons) > 18
    
    # Add season column with appropriate format
    df_filtered['season'] = df_filtered['timestamp'].apply(lambda x: get_season(x, short_format=use_short_format))
    df_filtered['minutes_played'] = df_filtered['ms_played'] / (1000 * 60)
    
    # Group by season and artist
    season_artist_minutes = df_filtered.groupby(['season', 'artist_name'])['minutes_played'].sum().reset_index()
    
    # Generate seasons for the date range with the chosen format
    seasons = generate_seasons(date_range[0], date_range[1], short_format=use_short_format)
    available_seasons = [s for s in seasons if s in season_artist_minutes['season'].unique()]
    
    # Get top N artists per season
    season_data = {}
    all_artists = set()
    
    for season in available_seasons:
        season_artists = season_artist_minutes[season_artist_minutes['season'] == season]
        if len(season_artists) > 0:
            top_n_season = season_artists.nlargest(top_n, 'minutes_played')
            season_data[season] = top_n_season
            all_artists.update(top_n_season['artist_name'].tolist())
    
    # Get color mapping
    artist_colors = assign_artist_colors(list(all_artists))
    
    # Create plot data
    plot_data = []
    for season in available_seasons:
        if season in season_data:
            season_artists = season_data[season].sort_values('minutes_played', ascending=False)
            for idx, (_, row) in enumerate(season_artists.iterrows()):
                plot_data.append({
                    'season': season,
                    'artist': row['artist_name'],
                    'minutes': row['minutes_played'],
                    'color': artist_colors[row['artist_name']],
                    'rank': idx + 1,
                })
    
    if not plot_data:
        return go.Figure().update_layout(
            title="No data to display for the selected parameters.",
            template='plotly_dark',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
        ), html.Div([
            dbc.Alert("No data to display for the selected parameters.", color="warning")
        ])
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig = go.Figure()
    
    # Create one trace per rank position
    for rank in range(1, top_n + 1):
        rank_data = plot_df[plot_df['rank'] == rank]
        
        if len(rank_data) > 0:
            x_vals = []
            y_vals = []
            colors = []
            texts = []
            hovers = []
            
            for season in available_seasons:
                season_rank_data = rank_data[rank_data['season'] == season]
                if len(season_rank_data) > 0:
                    row = season_rank_data.iloc[0]
                    x_vals.append(season)
                    y_vals.append(row['minutes'])
                    colors.append(row['color'])
                    texts.append(f"{row['artist']}<br>{row['minutes']:.0f} min")
                    hovers.append(f"<b>{row['artist']}</b><br>Season: {season}<br>Rank: #{rank}<br>Minutes: {row['minutes']:.1f}<br><extra></extra>")
                else:
                    x_vals.append(season)
                    y_vals.append(0)
                    colors.append('#000000')
                    texts.append("")
                    hovers.append("")
            
            fig.add_trace(go.Bar(
                name=f"Rank {rank}",
                x=x_vals,
                y=y_vals,
                marker_color=colors,
                showlegend=False,
                text=texts,
                textposition="outside",
                textfont=dict(size=8),
                hovertemplate=hovers,
                visible=True
            ))
    
    # Remove zero-height bars
    for trace in fig.data:
        y_data = list(trace.y)
        text_data = list(trace.text)
        for i, val in enumerate(y_data):
            if val == 0:
                y_data[i] = None
                text_data[i] = ""
        trace.y = y_data
        trace.text = text_data
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Top {top_n} Artists by Streaming Minutes per Season",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Season",
        yaxis_title="Minutes Streamed",
        barmode='group',
        height=800,
        template='plotly_dark',
        xaxis=dict(
            tickangle=0,
            categoryorder='array',
            categoryarray=available_seasons,
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
    )
    
    # Create stats
    total_unique_artists = len(all_artists)
    total_seasons = len(available_seasons)
    stats = html.Div([
        html.H4("📊 Statistics"),
        html.P(f"Showing top {top_n} artists across {total_seasons} seasons"),
        html.P(f"Total unique artists: {total_unique_artists}"),
        html.P(f"Date range: {available_seasons[0] if available_seasons else 'N/A'} to {available_seasons[-1] if available_seasons else 'N/A'}"),
    ])
    
    return fig, stats

# Callback for top songs chart
@app.callback(
    [Output("top-songs-chart", "figure"),
     Output("songs-stats", "children")],
    [Input("top-n-songs-slider", "value"),
     Input("date-range-slider-songs", "value")]
)
def update_songs_chart(top_n, date_range):
    if df_global is None:
        return go.Figure(), html.Div()
    
    # Filter data by date range
    start_date = f"{date_range[0]}-01-01"
    end_date = f"{date_range[1]}-12-31"
    df_filtered = df_global[(df_global['timestamp'] >= start_date) & 
                           (df_global['timestamp'] <= end_date)].copy()
    
    if len(df_filtered) == 0:
        return go.Figure().update_layout(
            title="No data available for the selected date range.",
            template='plotly_dark',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
        ), html.Div([
            dbc.Alert("No data available for the selected date range.", color="warning")
        ])
    
    # Generate seasons for the date range to determine format
    seasons = generate_seasons(date_range[0], date_range[1], short_format=True)  # Get count first
    use_short_format = len(seasons) > 18
    
    # Add season column with appropriate format
    df_filtered['season'] = df_filtered['timestamp'].apply(lambda x: get_season(x, short_format=use_short_format))
    df_filtered['minutes_played'] = df_filtered['ms_played'] / (1000 * 60)
    
    # Group by season, song, and artist
    season_song_minutes = df_filtered.groupby(['season', 'track_name', 'artist_name'])['minutes_played'].sum().reset_index()
    
    # Generate seasons for the date range with the chosen format
    seasons = generate_seasons(date_range[0], date_range[1], short_format=use_short_format)
    available_seasons = [s for s in seasons if s in season_song_minutes['season'].unique()]
    
    # Get top N songs per season
    season_data = {}
    all_artists = set()
    
    for season in available_seasons:
        season_songs = season_song_minutes[season_song_minutes['season'] == season]
        if len(season_songs) > 0:
            top_n_season = season_songs.nlargest(top_n, 'minutes_played')
            season_data[season] = top_n_season
            all_artists.update(top_n_season['artist_name'].tolist())
    
    # Get color mapping based on artists
    artist_colors = assign_artist_colors(list(all_artists))
    
    # Create plot data
    plot_data = []
    for season in available_seasons:
        if season in season_data:
            season_songs = season_data[season].sort_values('minutes_played', ascending=False)
            for idx, (_, row) in enumerate(season_songs.iterrows()):
                plot_data.append({
                    'season': season,
                    'song': row['track_name'],
                    'artist': row['artist_name'],
                    'minutes': row['minutes_played'],
                    'color': artist_colors[row['artist_name']],
                    'rank': idx + 1,
                })
    
    if not plot_data:
        return go.Figure().update_layout(
            title="No data to display for the selected parameters.",
            template='plotly_dark',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
        ), html.Div([
            dbc.Alert("No data to display for the selected parameters.", color="warning")
        ])
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig = go.Figure()
    
    # Create one trace per rank position
    for rank in range(1, top_n + 1):
        rank_data = plot_df[plot_df['rank'] == rank]
        
        if len(rank_data) > 0:
            x_vals = []
            y_vals = []
            colors = []
            texts = []
            hovers = []
            
            for season in available_seasons:
                season_rank_data = rank_data[rank_data['season'] == season]
                if len(season_rank_data) > 0:
                    row = season_rank_data.iloc[0]
                    x_vals.append(season)
                    y_vals.append(row['minutes'])
                    colors.append(row['color'])
                    # Truncate long song names for display
                    song_display = row['song'][:25] + "..." if len(row['song']) > 25 else row['song']
                    texts.append(f"{song_display}<br>by {row['artist']}<br>{row['minutes']:.0f} min")
                    hovers.append(f"<b>{row['song']}</b><br>by {row['artist']}<br>Season: {season}<br>Rank: #{rank}<br>Minutes: {row['minutes']:.1f}<br><extra></extra>")
                else:
                    x_vals.append(season)
                    y_vals.append(0)
                    colors.append('#000000')
                    texts.append("")
                    hovers.append("")
            
            fig.add_trace(go.Bar(
                name=f"Rank {rank}",
                x=x_vals,
                y=y_vals,
                marker_color=colors,
                showlegend=False,
                text=texts,
                textposition="outside",
                textfont=dict(size=7),
                hovertemplate=hovers,
                visible=True
            ))
    
    # Remove zero-height bars
    for trace in fig.data:
        y_data = list(trace.y)
        text_data = list(trace.text)
        for i, val in enumerate(y_data):
            if val == 0:
                y_data[i] = None
                text_data[i] = ""
        trace.y = y_data
        trace.text = text_data
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Top {top_n} Songs by Streaming Minutes per Season (Color-coded by Artist)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Season",
        yaxis_title="Minutes Streamed",
        barmode='group',
        height=800,
        template='plotly_dark',
        xaxis=dict(
            tickangle=0,
            categoryorder='array',
            categoryarray=available_seasons,
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
    )
    
    # Create stats
    total_unique_songs = len(plot_df['song'].unique())
    total_unique_artists = len(all_artists)
    total_seasons = len(available_seasons)
    stats = html.Div([
        html.H4("📊 Statistics"),
        html.P(f"Showing top {top_n} songs across {total_seasons} seasons"),
        html.P(f"Total unique songs: {total_unique_songs}"),
        html.P(f"Total unique artists: {total_unique_artists}"),
        html.P(f"Date range: {available_seasons[0] if available_seasons else 'N/A'} to {available_seasons[-1] if available_seasons else 'N/A'}"),
    ])
    
    return fig, stats

# Callback for daily streaming chart
@app.callback(
    [Output("daily-streaming-chart", "figure"),
     Output("daily-stats", "children")],
    [Input("ema-duration-slider", "value"),
     Input("date-range-slider-daily", "value")]
)
def update_daily_chart(ema_duration, date_range):
    if df_global is None:
        return go.Figure(), html.Div()
    
    # Filter data by date range
    start_date = f"{date_range[0]}-01-01"
    end_date = f"{date_range[1]}-12-31"
    df_filtered = df_global[(df_global['timestamp'] >= start_date) & 
                           (df_global['timestamp'] <= end_date)].copy()
    
    if len(df_filtered) == 0:
        return go.Figure().update_layout(
            title="No data available for the selected date range.",
            template='plotly_dark',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
        ), html.Div([
            dbc.Alert("No data available for the selected date range.", color="warning")
        ])
    
    # Convert ms_played to minutes and group by date
    df_filtered['minutes_played'] = df_filtered['ms_played'] / (1000 * 60)
    daily_minutes = df_filtered.groupby('date')['minutes_played'].sum()
    
    # Create complete date range to fill gaps
    date_range_complete = pd.date_range(start=daily_minutes.index.min(), 
                                      end=daily_minutes.index.max(), 
                                      freq='D')
    daily_minutes_complete = daily_minutes.reindex(date_range_complete, fill_value=0)
    
    # Create plot dataframe
    plot_df = pd.DataFrame({
        'date': pd.to_datetime(daily_minutes_complete.index),
        'minutes': daily_minutes_complete.values
    })
    
    # Create figure
    fig = go.Figure()
    
    # Map slider values to EMA spans
    ema_spans = {0: None, 1: 30, 2: 60, 3: 90, 4: 120}
    ema_span = ema_spans[ema_duration]
    
    if ema_span is None:
        # No EMA - show raw daily data
        fig.add_trace(go.Scatter(
            x=plot_df['date'], 
            y=plot_df['minutes'],
            mode='lines',
            name='Daily Minutes',
            line=dict(color='#FF6B6B', width=1),
            hovertemplate='<b>%{x}</b><br>Minutes: %{y:.1f}<extra></extra>'
        ))
        title_text = "Daily Spotify Minutes Streamed"
    else:
        # Calculate EMA
        plot_df['ema'] = plot_df['minutes'].ewm(span=ema_span).mean()
        
        # Add raw data as a light background line
        fig.add_trace(go.Scatter(
            x=plot_df['date'], 
            y=plot_df['minutes'],
            mode='lines',
            name='Daily Minutes (Raw)',
            line=dict(color='#888888', width=1, dash='dot'),
            opacity=0.6,
            hovertemplate='<b>%{x}</b><br>Raw Minutes: %{y:.1f}<extra></extra>'
        ))
        
        # Add EMA line
        fig.add_trace(go.Scatter(
            x=plot_df['date'], 
            y=plot_df['ema'],
            mode='lines',
            name=f'{ema_span}-day EMA',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate=f'<b>%{{x}}</b><br>{ema_span}-day EMA: %{{y:.1f}}<extra></extra>'
        ))
        
        title_text = f"Daily Spotify Minutes Streamed ({ema_span}-day EMA)"
    
    # Calculate a reasonable Y-axis maximum based on 99th percentile to avoid outlier compression
    # This gives the main trend lines more vertical room to be visible
    percentile_99 = plot_df['minutes'].quantile(0.99)
    # Round up to the next 100-minute interval
    y_max = int((percentile_99 // 100 + 1) * 100)
    # Ensure minimum of 200 for very low-usage cases
    y_max = max(y_max, 200)
    
    # Update layout
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Date",
        yaxis_title="Minutes Streamed",
        height=600,
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(range=[0, y_max]),  # Set calculated Y-axis range to avoid outlier compression
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
    )
    
    # Calculate statistics
    total_days = len(plot_df)
    total_minutes = plot_df['minutes'].sum()
    avg_minutes = plot_df['minutes'].mean()
    max_minutes = plot_df['minutes'].max()
    days_with_listening = (plot_df['minutes'] > 0).sum()
    
    # Create stats
    stats = html.Div([
        html.H4("📊 Statistics"),
        html.P(f"Total days: {total_days:,}"),
        html.P(f"Days with listening: {days_with_listening:,} ({days_with_listening/total_days*100:.1f}%)"),
        html.P(f"Total minutes: {total_minutes:,.0f} ({total_minutes/60:.0f} hours)"),
        html.P(f"Average minutes per day: {avg_minutes:.1f}"),
        html.P(f"Peak listening day: {max_minutes:.0f} minutes"),
        html.P(f"EMA smoothing: {'None' if ema_span is None else f'{ema_span} days'}"),
        html.Hr(),
        html.P(f"Chart Y-axis max: {y_max} min (99th percentile: {percentile_99:.0f} min)", 
               className="text-muted small"),
        html.P("Y-axis optimized to show trend details by excluding extreme outliers", 
               className="text-muted small"),
    ])
    
    return fig, stats

# Callback to populate song search dropdown
@app.callback(
    Output("song-search-dropdown", "options"),
    [Input("song-search-dropdown", "search_value"),
     Input("song-search-dropdown", "value")]
)
def populate_song_dropdown(search_value, selected_value):
    if song_stats_global is None or song_search_index is None:
        return []
    
    all_songs = list(song_stats_global.keys())
    
    # If we have a selected value, we want to ensure it's in the options
    # This helps maintain the selection after the user clicks
    selected_song_key = None
    if selected_value:
        try:
            parts = selected_value.split("|||")
            if len(parts) == 2:
                selected_song_key = (parts[0].strip(), parts[1].strip())
        except:
            pass
    
    # If no search value, return top songs by minutes
    if not search_value:
        top_songs = sorted(all_songs, key=lambda x: song_stats_global[x]['total_minutes'], reverse=True)[:SEARCH_CONSTANTS['TOP_SONGS_LIMIT']]
        
        # Ensure selected song is included if it exists and isn't already in top songs
        if selected_song_key and selected_song_key not in top_songs:
            top_songs = [selected_song_key] + top_songs[:-1]
        
        return [{"label": f"{track} - {artist}", "value": f"{track}|||{artist}"} 
                for track, artist in top_songs]
    
    # Use improved weighted fuzzy search
    try:
        search_results = fuzzy_search_songs(
            query=search_value,
            search_index=song_search_index,
            song_stats=song_stats_global,
            limit=SEARCH_CONSTANTS['DEFAULT_SEARCH_LIMIT'],
            min_score=SEARCH_CONSTANTS['SEARCH_MIN_SCORE']  # Slightly lower threshold for better recall
        )
        
        # Convert results to dropdown options with match type indicators
        options = []
        result_keys = set()
        
        for song_key, score, match_type in search_results:
            track, artist = song_key
            result_keys.add(song_key)
            
            # Add subtle indicators for match type to help users understand why songs appear
            match_indicator = {
                'track': '🎵',
                'artist': '🎤', 
                'album': '💿',
                'combined': '🔍'
            }.get(match_type, '')
            
            label = f"{match_indicator} {track} - {artist}"
            options.append({
                "label": label,
                "value": f"{track}|||{artist}"
            })
        
        # Ensure selected song is included if it exists and isn't already in search results
        if selected_song_key and selected_song_key not in result_keys and selected_song_key in all_songs:
            track, artist = selected_song_key
            options.insert(0, {
                "label": f"✓ {track} - {artist}",  # Checkmark to show it's selected
                "value": f"{track}|||{artist}"
            })
        
        return options
        
    except Exception as e:
        # Fallback to basic search if advanced search fails
        print(f"Warning: Advanced search failed, using fallback: {e}")
        search_lower = search_value.lower()
        matching_songs = []
        
        for track, artist in all_songs:
            track_safe = track or ''
            artist_safe = artist or ''
            if (search_lower in track_safe.lower() or search_lower in artist_safe.lower()):
                matching_songs.append((track, artist))
        
        # Sort by total minutes and limit results
        matching_songs = sorted(matching_songs, 
                              key=lambda x: song_stats_global[x]['total_minutes'], 
                              reverse=True)[:SEARCH_CONSTANTS['DEFAULT_SEARCH_LIMIT']]
        
        # Ensure selected song is included if it exists and isn't already in results
        if selected_song_key and selected_song_key not in matching_songs and selected_song_key in all_songs:
            matching_songs.insert(0, selected_song_key)
            if len(matching_songs) > SEARCH_CONSTANTS['DEFAULT_SEARCH_LIMIT']:
                matching_songs = matching_songs[:SEARCH_CONSTANTS['DEFAULT_SEARCH_LIMIT']]
        
        return [{"label": f"{track} - {artist}", "value": f"{track}|||{artist}"} 
                for track, artist in matching_songs]

# Callback to clear search when a selection is made (improves UX)
@app.callback(
    Output("song-search-dropdown", "search_value"),
    Input("song-search-dropdown", "value"),
    prevent_initial_call=True
)
def clear_search_on_selection(selected_value):
    """Clear search input when user selects a song to improve UX."""
    if selected_value:
        return ""  # Clear the search value
    return dash.no_update

# Callback for song details display
@app.callback(
    Output("song-details-content", "children"),
    Input("song-search-dropdown", "value")
)
def display_song_details(selected_song):
    if not selected_song or song_stats_global is None or artist_rankings_global is None:
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Select a song to view detailed statistics", className="text-center text-muted"),
                    html.P("Use the dropdown above to search for and select a song from your library.", className="text-center text-muted")
                ])
            ])
        ])
    
    # Parse the selected song with proper validation
    try:
        # Validate input format
        if not isinstance(selected_song, str) or "|||" not in selected_song:
            return html.Div([dbc.Alert("Invalid song selection format.", color="danger")])
        
        parts = selected_song.split("|||")
        if len(parts) != 2:
            return html.Div([dbc.Alert("Invalid song selection format - expected track|||artist.", color="danger")])
        
        track_name, artist_name = parts
        
        # Validate that parts are not empty
        if not track_name.strip() or not artist_name.strip():
            return html.Div([dbc.Alert("Invalid song selection - empty track or artist name.", color="danger")])
        
        song_key = (track_name.strip(), artist_name.strip())
        
    except Exception as e:
        return html.Div([dbc.Alert(f"Error parsing selected song: {str(e)}", color="danger")])
    
    if song_key not in song_stats_global:
        return html.Div([dbc.Alert("Song not found in statistics. Please try searching again.", color="warning")])
    
    stats = song_stats_global[song_key]
    
    # Get artist rankings
    artist_rank_all_time = artist_rankings_global['all_time']['rankings'].get(artist_name, "N/A")
    artist_rank_1yr = artist_rankings_global['past_1_year']['rankings'].get(artist_name, "N/A")
    artist_rank_6mo = artist_rankings_global['past_6_months']['rankings'].get(artist_name, "N/A")
    artist_rank_1mo = artist_rankings_global['past_1_month']['rankings'].get(artist_name, "N/A")
    
    # Create seasonal histogram with proper chronological ordering and missing seasons
    seasonal_data = stats.get('seasonal_data', {})
    first_streamed = stats.get('first_streamed')
    last_streamed = stats.get('last_streamed')
    
    # Only create chart if we have valid streaming dates
    if first_streamed and last_streamed and seasonal_data:
        # Determine the season format used (consistent with how data was stored)
        sample_season = list(seasonal_data.keys())[0] if seasonal_data else None
        use_short_format = sample_season and '<br>' not in sample_season
        
        # Generate complete chronological season range between first and last stream
        complete_seasons = generate_complete_season_range(
            first_streamed, 
            last_streamed, 
            short_format=use_short_format
        )
        
        # Fill in missing seasons with 0 minutes
        complete_data = []
        for season in complete_seasons:
            minutes = seasonal_data.get(season, 0.0)  # Default to 0 if season not in data
            complete_data.append((season, minutes))
        
        # Sort chronologically using our parsing function
        complete_data.sort(key=lambda x: parse_season_for_sorting(x[0]))
        
        # Extract sorted seasons and minutes
        seasons = [item[0] for item in complete_data]
        minutes = [item[1] for item in complete_data]
        
        # Create the figure
        fig = go.Figure(data=[
            go.Bar(
                x=seasons,
                y=minutes,
                marker_color='#FF6B6B',
                hovertemplate='<b>%{x}</b><br>Minutes: %{y:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': f"Streaming History by Season",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Season",
            yaxis_title="Minutes Streamed",
            height=400,
            template='plotly_dark',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            xaxis=dict(
                tickangle=45 if len(seasons) > 6 else 0,  # Rotate labels if many seasons
                type='category'  # Ensure categorical ordering is preserved
            )
        )
        
        seasonal_chart = dcc.Graph(figure=fig)
    else:
        # More informative message based on data state
        if not seasonal_data:
            message = "No seasonal streaming data available for this song."
        elif not first_streamed or not last_streamed:
            message = "Invalid streaming date information for this song."
        else:
            message = "No streaming activity recorded for this song across any season."
        seasonal_chart = dbc.Alert(message, color="info")
    
    return html.Div([
        # Song header
        dbc.Card([
            dbc.CardBody([
                html.H3(f"🎵 {stats['track_name']}", className="mb-2"),
                html.H5(f"by {stats['artist_name']}", className="text-muted mb-2"),
                html.H6(f"Album: {stats['album_name']}", className="text-muted")
            ])
        ], className="mb-4"),
        
        # Basic stats cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(stats['first_streamed'].strftime('%Y-%m-%d'), className="text-primary"),
                        html.P("First Streamed", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(stats['last_streamed'].strftime('%Y-%m-%d'), className="text-primary"),
                        html.P("Last Streamed", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['total_streams']:,}", className="text-primary"),
                        html.P("Total Streams", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['streams_60s']:,}", className="text-primary"),
                        html.P("Streams (60s+)", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=3),
        ], className="mb-4"),
        
        # More stats
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['total_minutes']:,.1f}", className="text-primary"),
                        html.P("Total Minutes", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['percentile_rank']:.1f}%", className="text-primary"),
                        html.P("Percentile Rank", className="text-light"),
                    ])
                ], className="custom-card text-center")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['total_minutes']/stats['total_streams']:.1f}" if stats['total_streams'] > 0 else "0.0", className="text-primary"),
                        html.P("Avg Minutes/Stream", className="text-light")
                    ])
                ], className="custom-card text-center")
            ], width=4),
        ], className="mb-4"),
        
        # Artist rankings
        dbc.Card([
            dbc.CardBody([
                html.H5("🎤 Artist Rankings", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H5(f"#{artist_rank_all_time}" if isinstance(artist_rank_all_time, int) else str(artist_rank_all_time), 
                               className="text-primary text-center"),
                        html.P("All-Time Rank", className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.H5(f"#{artist_rank_1yr}" if isinstance(artist_rank_1yr, int) else str(artist_rank_1yr), 
                               className="text-primary text-center"),
                        html.P("Past 1 Year Rank", className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.H5(f"#{artist_rank_6mo}" if isinstance(artist_rank_6mo, int) else str(artist_rank_6mo), 
                               className="text-primary text-center"),
                        html.P("Past 6 Months Rank", className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.H5(f"#{artist_rank_1mo}" if isinstance(artist_rank_1mo, int) else str(artist_rank_1mo), 
                               className="text-primary text-center"),
                        html.P("Past 1 Month Rank", className="text-center")
                    ], width=3),
                ])
            ])
        ], className="mb-4"),
        
        # Seasonal chart
        dbc.Card([
            dbc.CardBody([
                html.H5("📊 Streaming History by Season", className="mb-3"),
                seasonal_chart
            ])
        ])
    ])

# Callback for artists by minutes page
@app.callback(
    [Output("artists-chart-container", "children"),
     Output("artists-table-container", "children")],
    Input("time-period-selector", "value")
)
def update_artists_by_minutes(time_period):
    if artist_rankings_global is None:
        return html.Div(), html.Div()
    
    # Get the data for the selected time period
    period_data = artist_rankings_global.get(time_period, {})
    minutes_data = period_data.get('minutes', {})
    
    if not minutes_data:
        no_data_message = dbc.Alert(f"No data available for {time_period.replace('_', ' ').title()}.", color="warning")
        return no_data_message, html.Div()
    
    # Sort artists by minutes (highest to lowest)
    sorted_artists = sorted(minutes_data.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top 25 artists for better visualization
    top_artists = sorted_artists[:25]
    
    if not top_artists:
        return html.Div(), html.Div()
    
    # Prepare data for the chart
    artist_names = [artist for artist, _ in top_artists]
    minutes_values = [minutes for _, minutes in top_artists]
    
    # Get artist colors for consistent styling (use all artists for consistent color assignment)
    all_artists_in_data = list(minutes_data.keys())
    artist_colors_map = assign_artist_colors(all_artists_in_data)
    bar_colors = [artist_colors_map.get(artist, '#FF6B6B') for artist in artist_names]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=minutes_values,
            y=artist_names,
            orientation='h',
            marker_color=bar_colors,
            hovertemplate='<b>%{y}</b><br>Minutes: %{x:,.1f}<br>Hours: %{customdata:.1f}<extra></extra>',
            customdata=[m/60 for m in minutes_values]  # Convert to hours for hover
        )
    ])
    
    # Update layout for horizontal bar chart
    fig.update_layout(
        title={
            'text': f"Top Artists by Minutes Streamed - {time_period.replace('_', ' ').title()}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Minutes Streamed",
        yaxis_title="Artists",
        height=max(600, len(top_artists) * 25 + 100),  # Dynamic height based on number of artists
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        yaxis=dict(autorange='reversed'),  # Show highest at top
        margin=dict(l=150)  # More left margin for artist names
    )
    
    chart = dcc.Graph(figure=fig)
    
    # Create table data
    table_data = []
    for i, (artist, minutes) in enumerate(top_artists, 1):
        hours = minutes / 60
        table_data.append({
            'rank': i,
            'artist': artist,
            'minutes': f"{minutes:,.1f}",
            'hours': f"{hours:.1f}"
        })
    
    # Create table
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Rank", style={"width": "10%"}),
                html.Th("Artist", style={"width": "50%"}),
                html.Th("Minutes", style={"width": "20%"}),
                html.Th("Hours", style={"width": "20%"})
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(row['rank'], className="text-center"),
                html.Td(row['artist']),
                html.Td(row['minutes'], className="text-end"),
                html.Td(row['hours'], className="text-end")
            ]) for row in table_data
        ])
    ], 
    striped=True, 
    bordered=True, 
    hover=True,
    responsive=True,
    className="mt-3",
    style={"backgroundColor": "#1a252f"})
    
    # Wrap table in a card
    table_card = dbc.Card([
        dbc.CardBody([
            html.H5(f"📊 Top Artists Data - {time_period.replace('_', ' ').title()}", className="mb-3"),
            table
        ])
    ])
    
    return chart, table_card

# ===== Artist Cumulative Page Callbacks =====
@app.callback(
    Output("artist-search-dropdown", "options"),
    [Input("artist-search-dropdown", "search_value"),
     Input("artist-search-dropdown", "value")]
)
def populate_artist_search(search_value, selected_value):
    if artist_list_global is None:
        return []
    # Default list: top by minutes
    if not search_value:
        top = artist_total_minutes_global.head(30).index.tolist() if artist_total_minutes_global is not None else artist_list_global[:30]
        if selected_value and selected_value not in top:
            top = [selected_value] + top[:-1]
        return [{"label": a, "value": a} for a in top]
    results = fuzzy_search_artists(search_value, artist_list_global, limit=20, min_score=SEARCH_CONSTANTS['SEARCH_MIN_SCORE'])
    options = [{"label": a, "value": a} for a, _ in results]
    if selected_value and selected_value not in [o['value'] for o in options] and selected_value in artist_list_global:
        options.insert(0, {"label": f"✓ {selected_value}", "value": selected_value})
    return options

@app.callback(
    Output("selected-artists-store", "data"),
    [Input("artist-search-dropdown", "value"),
     Input("btn-show-all-artists", "n_clicks"),
     Input("btn-hide-all-artists", "n_clicks"),
     Input("btn-remove-hidden-artists", "n_clicks"),
     Input("artist-visibility-checklist", "value")],
    [State("selected-artists-store", "data")],
    prevent_initial_call=True
)
def update_selected_artists(new_selection, show_all_clicks, hide_all_clicks, remove_hidden_clicks, visible_checked, store):
    store = store or {"selected": [], "visible": []}
    selected = list(store.get("selected", []))
    visible = list(store.get("visible", []))
    ctx = dash.callback_context
    if not ctx.triggered:
        return store
    trigger = ctx.triggered[0]['prop_id']
    if trigger.startswith("artist-search-dropdown") and new_selection:
        if new_selection not in selected:
            selected.append(new_selection)
            # Make newly added artist visible by default
            if new_selection not in visible:
                visible.append(new_selection)
        return {"selected": selected, "visible": visible}
    if trigger.startswith("artist-visibility-checklist"):
        return {"selected": selected, "visible": visible_checked or []}
    if trigger.startswith("btn-show-all-artists"):
        return {"selected": selected, "visible": selected}
    if trigger.startswith("btn-hide-all-artists"):
        return {"selected": selected, "visible": []}
    if trigger.startswith("btn-remove-hidden-artists"):
        remaining = [a for a in selected if a in visible]
        return {"selected": remaining, "visible": remaining}
    return store

@app.callback(
    Output("artist-visibility-checklist", "options"),
    Output("artist-visibility-checklist", "value"),
    Input("selected-artists-store", "data")
)
def render_artist_checklist(store):
    store = store or {"selected": [], "visible": []}
    options = [{"label": a, "value": a} for a in store.get("selected", [])]
    value = store.get("visible", []) or store.get("selected", [])
    return options, value

@app.callback(
    Output("artist-cumulative-graph", "figure"),
    [Input("selected-artists-store", "data"),
     Input("artist-cum-start-date", "date")]
)
def render_artist_cumulative_graph(store, start_date):
    if artist_daily_minutes_global is None:
        return go.Figure()
    store = store or {"selected": [], "visible": []}
    selected = store.get("selected", [])
    visible = set(store.get("visible", []))
    fig = go.Figure()
    if not selected:
        fig.update_layout(template='plotly_dark', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                          title={"text": "Add artists to view cumulative minutes over time", "x": 0.5, "xanchor": "center"})
        return fig
    # Build cumulative per artist
    dates = sorted(artist_daily_minutes_global['date'].unique())
    if start_date:
        # Filter dates to start from selected date
        dates = [d for d in dates if pd.to_datetime(d) >= pd.to_datetime(start_date)]
    # Color mapping based on existing scheme for consistency
    artist_colors = assign_artist_colors(selected)
    for artist in selected:
        df_a = artist_daily_minutes_global[artist_daily_minutes_global['artist_name'] == artist]
        if df_a.empty:
            continue
        # Apply start-date filter to the artist data as well to avoid index reintroduction
        if start_date:
            df_a = df_a[pd.to_datetime(df_a['date']) >= pd.to_datetime(start_date)]
        series = pd.Series(0.0, index=pd.to_datetime(dates))
        series.loc[pd.to_datetime(df_a['date'])] = df_a['minutes_played'].values
        cumulative = series.sort_index().cumsum()
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode='lines',
            name=artist,
            line=dict(color=artist_colors.get(artist, '#FF6B6B'), width=2),
            visible=True if artist in visible else 'legendonly',
        ))
    fig.update_layout(
        title={
            'text': 'Cumulative Minutes Listened by Artist',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Cumulative Minutes',
        template='plotly_dark',
        height=650,
        hovermode='x unified',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig

def main():
    global df_global, entries_global, song_stats_global, artist_rankings_global, song_search_index
    global artist_list_global, artist_total_minutes_global, artist_daily_minutes_global
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Spotify Analytics Dashboard")
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to Spotify Extended Streaming History directory")
    parser.add_argument("--port", type=int, default=8050, 
                        help="Port to run the dashboard on")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to run the dashboard on")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode")
    args = parser.parse_args()
    
    # Load data
    print("Loading Spotify data...")
    try:
        entries_global = load_and_clean_entries(args.data)
        df_global = pd.DataFrame(entries_global)
        df_global['timestamp'] = pd.to_datetime(df_global['timestamp'])
        df_global['date'] = df_global['timestamp'].dt.date
        print(f"✅ Data loaded successfully! {len(df_global)} records from {df_global['timestamp'].min()} to {df_global['timestamp'].max()}")
        
        # Precompute song statistics and artist rankings
        print("Precomputing analytics...")
        artist_rankings_global = precompute_artist_rankings(df_global.copy())
        song_stats_global = precompute_song_stats(df_global.copy())
        
        # Pre-build song search index with proper structure and null safety
        print("Building search index...")
        song_search_index = {}
        for key, stats in song_stats_global.items():
            # Handle potential None values and create structured search text
            track_name = stats.get('track_name') or ''
            artist_name = stats.get('artist_name') or ''
            album_name = stats.get('album_name') or ''
            
            # Store individual fields for weighted search
            song_search_index[key] = {
                'track_name': track_name,
                'artist_name': artist_name,
                'album_name': album_name,
                'combined': f"TRACK:{track_name} ARTIST:{artist_name} ALBUM:{album_name}"
            }
        print(f"✅ Search index built for {len(song_search_index)} songs!")
        
        # Precompute artist lists and daily/cumulative structures for the cumulative page
        print("Preparing artist cumulative data...")
        artist_list_global = sorted(df_global['artist_name'].dropna().unique().tolist())
        # Daily minutes per artist (for cumulative build)
        tmp = df_global.copy()
        tmp['minutes_played'] = tmp['ms_played'] / (1000 * 60)
        artist_daily_minutes_global = (
            tmp.groupby(['date', 'artist_name'])['minutes_played']
               .sum()
               .reset_index()
               .sort_values('date')
        )
        # Total minutes per artist (for default suggestions)
        artist_total_minutes_global = (
            tmp.groupby('artist_name')['minutes_played']
               .sum()
               .sort_values(ascending=False)
        )
        print("✅ Artist cumulative data prepared!")

        print("✅ Analytics precomputation complete!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Open browser automatically after a short delay
    def open_browser():
        webbrowser.open(f"http://{args.host}:{args.port}")
    
    Timer(1.5, open_browser).start()
    
    # Run the app
    print(f"🚀 Starting Spotify Analytics Dashboard at http://{args.host}:{args.port}")
    print("📊 Navigate between pages using the sidebar")
    print("🔧 Use the interactive controls to customize your visualizations")
    
    app.run(debug=args.debug, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 