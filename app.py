import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
from typing import List, Dict, Any
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import argparse
import webbrowser
from threading import Timer

# Set default plotly theme to dark
pio.templates.default = "plotly_dark"

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)

app.title = "Spotify Analytics Dashboard"

# Global variables for data
df_global = None
entries_global = None

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
        html.H1("🎵 Spotify Analytics Dashboard", className="mb-4"),
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
            html.H3("🚀 Quick Start", className="text-light mb-3"),
            html.P("Use the sidebar to navigate between different visualizations:"),
            html.Ul([
                html.Li("📈 Daily Streaming - View your daily listening patterns with EMA smoothing"),
                html.Li("🎤 Top Artists - See your favorite artists over time"),
                html.Li("🎵 Top Songs - Explore individual tracks by season"),
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
        html.H1("🎤 Top Artists by Season", className="mb-4"),
        
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
        html.H1("🎵 Top Songs by Season", className="mb-4"),
        
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
        html.H1("📈 Daily Streaming Minutes", className="mb-4"),
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

# Callback for page routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/daily-streaming":
        return daily_streaming_layout()
    elif pathname == "/top-artists":
        return top_artists_layout()
    elif pathname == "/top-songs":
        return top_songs_layout()
    elif pathname == "/coming-soon":
        return html.Div([
            html.H1("🚧 Coming Soon!", className="text-light text-center"),
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
    ])
    
    return fig, stats

def main():
    global df_global, entries_global
    
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