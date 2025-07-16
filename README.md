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

### Installation

1. **Clone or download the files**:
   ```bash
   git clone https://github.com/andrewsingh/streaming-analytics.git
   cd streaming-analytics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
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

## 📁 Getting Your Spotify Data

1. Go to [Spotify Privacy Settings](https://www.spotify.com/account/privacy/)
2. Request your "Extended Streaming History" (not the basic account data)
3. Wait for Spotify to email you the download link (can take up to 30 days)
4. Download and extract the zip file
6. Use the path to this `Spotify Extended Streaming History` folder as the value for the `--data` argument

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

### Metrics
- **Minutes Played**: Total listening time per artist/song
- **Streaming Sessions**: Individual play events
- **Date Range**: Covers your entire listening history

## 🛠️ Troubleshooting

### Common Issues

1. **"No data loaded" error**:
   - Verify the path to your Spotify data directory
   - Ensure JSON files are in the `audio` subfolder
   - Check file permissions

2. **Empty charts**:
   - Adjust the date range sliders
   - Ensure you have data for the selected time period
   - Try reducing the number of top items

3. **Port already in use**:
   - Use a different port: `--port 8051`
   - Or stop other applications using port 8050

4. **Browser doesn't open automatically**:
   - Manually navigate to `http://localhost:8050`
   - Check if your firewall is blocking the connection

## 🔮 Future Enhancements

- **📈 Listening Patterns**: Daily/weekly/monthly trends
- **🎵 Genre Analysis**: Music taste evolution over time
- **🌍 Discovery Analysis**: New vs. familiar music patterns
- **📱 Platform Comparison**: Mobile vs. desktop listening habits
- **🎯 Mood Tracking**: Energy levels and emotional patterns
- **📊 Advanced Statistics**: Detailed listening metrics

## 💡 Tips for Best Experience

1. **Full Screen**: Use full-screen mode for better chart visibility
2. **Hover Information**: Hover over chart elements for detailed information
3. **Responsive Design**: Try different screen sizes and orientations
4. **Data Range**: Experiment with different time ranges for insights
5. **Sharing**: Use `--host 0.0.0.0` to share with others on your network

## 📝 Technical Details

- **Framework**: Dash (Plotly)
- **Styling**: Bootstrap (Cyborg theme)
- **Charts**: Plotly.js with dark theme
- **Data Processing**: Pandas
- **Color System**: Custom artist-based color wheel
- **Architecture**: Multi-page application with modular design

## 🤝 Contributing

This dashboard is designed to be easily extensible. To add new visualizations:

1. Create a new layout function
2. Add navigation link in the sidebar
3. Implement the corresponding callback
4. Follow the existing dark theme patterns

## 📄 License

This project is open source and available under the MIT License.

---

**Enjoy exploring your music! 🎵** 