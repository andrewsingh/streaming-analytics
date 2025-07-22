# 🎵 Spotify Analytics Dashboard

A modern, interactive web application for analyzing your Spotify Extended Streaming History data. Built with Dash and Plotly.

## ✨ Features

- **📊 Interactive Visualizations**: Multiple chart types with real-time updates
- **🎛️ Configurable Parameters**: Adjust time ranges, number of items, and more
- **🎤 Multi-Page Layout**: Organized sections for different analysis types

## 📋 Current Visualizations

1. **Overview**: Summary statistics and quick insights
2. **Daily Minutes Streamed**: See your daily listening activity over time
3. **Top Artists by Season**: See your favorite artists over time with configurable parameters
4. **Top Songs by Season**: Explore individual tracks colored by artist
5. **More Coming Soon**: Additional visualizations in development

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Your Spotify Extended Streaming History data (downloaded from Spotify)

## 📁 Getting Your Spotify Data
1. Go to [Spotify Privacy Settings](https://www.spotify.com/account/privacy/) and make sure you are signed in
2. Request your "Extended Streaming History" (not the basic account data)
3. You’ll get an email from Spotify asking you to confirm the request. Open the email and click the link to confirm.
5. Some time after confirming the request (usually a few hours, but can sometimes take longer), Spotify will send you another email saying that your extended streaming history is ready to download. Open this email and click the download link. This should download a `.zip` file called something like `my_spotify_data.zip` to your downloads.
6. Once the download has completed, double-click the `.zip` file to uncompress it. You should now have a folder called `Spotify Extended Streaming History`.
7. Use the path to this `Spotify Extended Streaming History` folder as the value for the `--data` argument when launching the dashboard.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/andrewsingh/streaming-analytics.git
   cd streaming-analytics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard with the path to your Spotify history**:
   ```bash
   python app.py --data "/path/to/your/Spotify Extended Streaming History"
   ```

### Example Usage

```bash
# Basic usage
python app.py --data "/path/to/your/Spotify Extended Streaming History"

# Custom port and host
python app.py --data "/path/to/your/Spotify Extended Streaming History" --port 8080 --host 0.0.0.0

# Debug mode
python app.py --data "/path/to/your/Spotify Extended Streaming History" --debug
```



## 🎛️ Interactive Controls

### Top Artists by Season
- **Number of Artists Slider**: Choose how many top artists to display (3-15)
- **Date Range Slider**: Select the years to analyze
- **Real-time Updates**: Charts update automatically as you adjust controls

### Top Songs by Season
- **Number of Songs Slider**: Choose how many top songs to display (3-15)
- **Date Range Slider**: Select the years to analyze
- **Artist Color Coding**: Songs are colored by their artist using an intelligent color wheel system

## 🎨 Color System

The dashboard uses a sophisticated color assignment system that groups artists by:
- **Genre**: Similar genres get similar color families
- **Mood/Energy**: Color intensity reflects the emotional tone of the music
- **Visual Distinction**: Ensures different artists are easily distinguishable

## 🔧 Command Line Options

```bash
python spotify_analytics_dashboard.py [OPTIONS]

Options:
  --data TEXT    Path to Spotify Extended Streaming History directory [REQUIRED]
  --port INTEGER      Port to run the dashboard on (default: 8050)
  --host TEXT         Host to run the dashboard on (default: 127.0.0.1)
  --debug             Run in debug mode
  --help              Show this message and exit
```

## 🌐 Accessing the Dashboard

Once running, the dashboard will:
1. **Automatically open** in your default browser
2. Be accessible at `http://localhost:8050` (or your custom host/port)
3. Show a **sidebar navigation** for easy page switching
4. Display **loading progress** and **data statistics** in the terminal

## 📊 Understanding Your Data

### Seasons
- **Spring**: January - June
- **Fall**: July - December
- **Chronological Order**: F16 → S17 → F17 → S18 → F18 → etc.


## 🛠️ Troubleshooting

### Common Issues

1. **"No data loaded" error**:
   - Verify the path to your Spotify data directory
   - Check file permissions

2. **Empty charts**:
   - Adjust the date range sliders
   - Ensure you have data for the selected time period

3. **Port already in use**:
   - Use a different port: `--port 8051`
   - Or stop other applications using port 8050

4. **Browser doesn't open automatically**:
   - Manually navigate to `http://localhost:8050`
   - Check if your firewall is blocking the connection



---

**Enjoy exploring your music! 🎵** 