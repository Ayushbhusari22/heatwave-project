import { useState, useEffect } from 'react';
import { checkHeatwave, fetchHistoricalData } from './api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ComposedChart, Area, Scatter, PieChart, Pie, Cell } from 'recharts';
import { Wind, Droplets, Thermometer, Cloud, Sun, AlertTriangle, Clock, Info, Droplet, Gauge, Calendar, SunDim, ThermometerSun } from 'lucide-react';
import './App.css';

// Enhanced weather condition icons mapping
const weatherIcons = {
  clear: "☀️",
  cloudy: "☁️",
  partlyCloudy: "⛅",
  rain: "🌧️",
  storm: "⛈️",
  snow: "❄️",
  mist: "🌫️",
  hot: "🔥",
  cold: "🥶",
  windy: "🌬️",
  humid: "💧"
};

// Alert level colors
const alertColors = {
  Normal: '#4CAF50',
  Caution: '#FFC107',
  Warning: '#FF9800',
  Emergency: '#F44336'
};

// Get weather icon based on comprehensive weather conditions
const getWeatherIcon = (temp, clouds, precipitation, windSpeed, humidity) => {
  if (temp > 35) return weatherIcons.hot;
  if (temp < 0) return weatherIcons.cold;
  if (precipitation > 10) return weatherIcons.storm;
  if (precipitation > 0) return weatherIcons.rain;
  if (windSpeed > 30) return weatherIcons.windy;
  if (humidity > 80) return weatherIcons.humid;
  if (clouds > 70) return weatherIcons.cloudy;
  if (clouds > 30) return weatherIcons.partlyCloudy;
  return weatherIcons.clear;
};

// Get humidity level description
const getHumidityLevel = (humidity) => {
  if (humidity < 30) return "Low";
  if (humidity < 60) return "Moderate";
  return "High";
};

// Get wind level description
const getWindLevel = (windSpeed) => {
  if (windSpeed < 10) return "Calm";
  if (windSpeed < 20) return "Moderate";
  if (windSpeed < 30) return "Strong";
  return "Very Strong";
};

// Format date for display
const formatDate = (dateString) => {
  const options = { weekday: 'short', month: 'short', day: 'numeric' };
  return new Date(dateString).toLocaleDateString(undefined, options);
};

function App() {
  const [city, setCity] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [historicalData, setHistoricalData] = useState(null);
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');
  const [searchHistory, setSearchHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [activeTab, setActiveTab] = useState('forecast');
  const [expandedCard, setExpandedCard] = useState(null);

  // Load search history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('searchHistory');
    if (savedHistory) {
      setSearchHistory(JSON.parse(savedHistory));
    }
  }, []);

  // City autocomplete with geocoding
  useEffect(() => {
    const fetchCitySuggestions = async () => {
      if (city.length < 3) {
        setSuggestions([]);
        return;
      }
      
      try {
        const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(city)}&limit=5`);
        const data = await response.json();
        const cityNames = [...new Set(data.map(item => item.display_name))];
        setSuggestions(cityNames);
      } catch (err) {
        console.error("Error fetching city suggestions:", err);
      }
    };

    const timeoutId = setTimeout(fetchCitySuggestions, 500);
    return () => clearTimeout(timeoutId);
  }, [city]);

  // Save search to history
  const saveToHistory = (cityName, hasHeatwave) => {
    const searchEntry = {
      city: cityName,
      timestamp: new Date().toISOString(),
      hadHeatwave: hasHeatwave
    };
    
    const updatedHistory = [searchEntry, ...searchHistory].slice(0, 10);
    setSearchHistory(updatedHistory);
    localStorage.setItem('searchHistory', JSON.stringify(updatedHistory));
  };

const handleSubmit = async (e) => {
  e.preventDefault();
  setLoading(true);
  setError('');
  
  try {
    const data = await checkHeatwave(city);
    setResult(data);
    
    if (data.heatwave_alert) {
      setNotificationMessage(`🔔 Heatwave Alert for ${city}!`);
      setShowNotification(true);
    }
  } catch (err) {
    setError(err.message || "Failed to connect to server");
    console.error("API Error:", err);
  } finally {
    setLoading(false);
  }
};
  const selectSuggestion = (suggestion) => {
    setCity(suggestion);
    setSuggestions([]);
  };

  const selectFromHistory = (historyCityName) => {
    setCity(historyCityName);
    setShowHistory(false);
    handleSubmit(new Event('submit'));
  };

  const dismissNotification = () => {
    setShowNotification(false);
  };

  const clearHistory = () => {
    setSearchHistory([]);
    localStorage.removeItem('searchHistory');
  };

  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  const toggleCardExpand = (cardId) => {
    setExpandedCard(expandedCard === cardId ? null : cardId);
  };

  // Prepare forecast data for charts
  const forecastData = result?.forecast?.map(day => ({
    ...day,
    date: formatDate(day.time),
    heatwaveProbability: parseFloat(day.heatwave_probability) * 100,
    isHeatwave: day.is_heatwave === 1,
    alertColor: alertColors[day.alert_level] || '#888'
  })) || [];

  // Prepare historical data for comparison chart
  const historicalChartData = historicalData?.map(day => ({
    date: formatDate(day.date),
    temperature: parseFloat(day.temperature),
    feelsLike: parseFloat(day.feelsLike),
    humidity: parseFloat(day.humidity),
    windSpeed: parseFloat(day.wind_speed)
  })) || [];

  // Prepare data for heat index chart
  const heatIndexData = forecastData.map(day => ({
    date: day.date,
    temperature: parseFloat(day.temperature_2m_max),
    humidity: parseFloat(day.relative_humidity_2m_mean),
    heatIndex: calculateHeatIndex(day.temperature_2m_max, day.relative_humidity_2m_mean),
    alertLevel: day.alert_level
  }));

  // Calculate heat index (simplified)
  function calculateHeatIndex(temp, humidity) {
    // Simplified heat index calculation
    const t = parseFloat(temp);
    const h = parseFloat(humidity);
    return t + 0.05 * h; // Simplified formula for demonstration
  }

  

  // Data for alert level pie chart
  const alertLevelData = [
    { name: 'Normal', value: forecastData.filter(d => d.alert_level === 'Normal').length },
    { name: 'Caution', value: forecastData.filter(d => d.alert_level === 'Caution').length },
    { name: 'Warning', value: forecastData.filter(d => d.alert_level === 'Warning').length },
    { name: 'Emergency', value: forecastData.filter(d => d.alert_level === 'Emergency').length }
  ];

  const COLORS = ['#4CAF50', '#FFC107', '#FF9800', '#F44336'];

  return (
    <div className="container">
      <h1>Heatwave Prediction</h1>
      
      {/* Notification system */}
      {showNotification && (
        <div className="notification">
          <span>{notificationMessage}</span>
          <button onClick={dismissNotification}>×</button>
        </div>
      )}
      
      {/* Search form with autocomplete and history */}
      <div className="search-container">
        <form onSubmit={handleSubmit} className="search-form">
          <div className="autocomplete">
            <input
              type="text"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              placeholder="Enter city name"
              required
            />
            {suggestions.length > 0 && (
              <ul className="suggestions">
                {suggestions.map((suggestion, index) => (
                  <li key={index} onClick={() => selectSuggestion(suggestion)}>
                    {suggestion}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button type="submit" disabled={loading}>
            {loading ? <span className="spinner"></span> : 'Check'}
          </button>
          <button 
            type="button" 
            className="history-toggle"
            onClick={toggleHistory}
            title="Search History"
          >
            <Clock size={18} />
          </button>
        </form>
        
        {/* Search history panel */}
        {showHistory && searchHistory.length > 0 && (
          <div className="history-panel">
            <div className="history-header">
              <h3>Search History</h3>
              <button onClick={clearHistory} className="clear-history">Clear</button>
            </div>
            <ul className="history-list">
              {searchHistory.map((entry, index) => (
                <li 
                  key={index} 
                  onClick={() => selectFromHistory(entry.city)}
                  className={entry.hadHeatwave ? 'heatwave-history' : ''}
                >
                  <span className="history-city">{entry.city}</span>
                  <span className="history-date">{new Date(entry.timestamp).toLocaleString()}</span>
                  {entry.hadHeatwave && <AlertTriangle size={16} className="heatwave-indicator" />}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {/* Loading spinner */}
      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Analyzing weather data...</p>
        </div>
      )}

      {result && (
        <div className={`result ${result.heatwave_alert ? 'alert' : 'safe'}`}>
          <h2>{result.city}</h2>
          
          {/* Current weather summary */}
          <div className="current-weather">
            <h3>Current Conditions</h3>
            <div className="weather-icon-large">
              {getWeatherIcon(
                parseFloat(result.current_weather.temperature), 
                parseFloat(result.current_weather.cloud_cover), 
                result.current_weather.precipitation || 0,
                parseFloat(result.current_weather.wind_speed),
                parseFloat(result.current_weather.humidity)
              )}
            </div>
            
            <div className="weather-parameters">
              <div className="parameter">
                <Thermometer size={24} className="parameter-icon" />
                <div className="parameter-content">
                  <div className="parameter-value">{result.current_weather.temperature}°C</div>
                  <div className="parameter-label">Temperature</div>
                </div>
                <div className="parameter-meter">
                  <div className="meter-fill temperature" style={{width: `${Math.min(parseFloat(result.current_weather.temperature)/45*100, 100)}%`}}></div>
                </div>
              </div>
              
              <div className="parameter">
                <ThermometerSun size={24} className="parameter-icon red" />
                <div className="parameter-content">
                  <div className="parameter-value">{result.current_weather.apparent_temperature}°C</div>
                  <div className="parameter-label">Feels Like</div>
                </div>
                <div className="parameter-meter">
                  <div className="meter-fill feels-like" style={{width: `${Math.min(parseFloat(result.current_weather.apparent_temperature)/45*100, 100)}%`}}></div>
                </div>
              </div>
              
              <div className="parameter">
                <Droplet size={24} className="parameter-icon blue" />
                <div className="parameter-content">
                  <div className="parameter-value">{result.current_weather.humidity}%</div>
                  <div className="parameter-label">Humidity ({getHumidityLevel(parseFloat(result.current_weather.humidity))})</div>
                </div>
                <div className="parameter-meter">
                  <div className="meter-fill humidity" style={{width: `${parseFloat(result.current_weather.humidity)}%`}}></div>
                </div>
              </div>
              
              <div className="parameter">
                <Wind size={24} className="parameter-icon gray" />
                <div className="parameter-content">
                  <div className="parameter-value">{result.current_weather.wind_speed} km/h</div>
                  <div className="parameter-label">Wind ({getWindLevel(parseFloat(result.current_weather.wind_speed))})</div>
                </div>
                <div className="parameter-meter">
                  <div className="meter-fill wind" style={{width: `${Math.min(parseFloat(result.current_weather.wind_speed)/40*100, 100)}%`}}></div>
                </div>
              </div>
              
              <div className="parameter">
                <Cloud size={24} className="parameter-icon light-blue" />
                <div className="parameter-content">
                  <div className="parameter-value">{result.current_weather.cloud_cover}%</div>
                  <div className="parameter-label">Cloud Cover</div>
                </div>
                <div className="parameter-meter">
                  <div className="meter-fill clouds" style={{width: `${parseFloat(result.current_weather.cloud_cover)}%`}}></div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Prediction summary */}
          <div className="prediction">
            <h3>{result.message}</h3>
            <div className="status-indicator">
              <span className="status-label">Status:</span>
              <span className={`status-value ${result.heatwave_alert ? 'danger' : 'safe'}`}>
                {result.heatwave_alert ? '🔥 Danger' : '✅ Normal'}
              </span>
            </div>
          </div>
          
          {/* Tab navigation for detailed views */}
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'forecast' ? 'active' : ''}`}
              onClick={() => setActiveTab('forecast')}
            >
              <Calendar size={16} /> Forecast
            </button>
            <button 
              className={`tab ${activeTab === 'historical' ? 'active' : ''}`}
              onClick={() => setActiveTab('historical')}
            >
              <Clock size={16} /> History
            </button>
            <button 
              className={`tab ${activeTab === 'analysis' ? 'active' : ''}`}
              onClick={() => setActiveTab('analysis')}
            >
              <Gauge size={16} /> Analysis
            </button>
          </div>
          
          {/* Forecast tab content */}
          {activeTab === 'forecast' && (
            <div className="tab-content">
              {/* Forecast temperature chart */}
              <div className={`chart-card ${expandedCard === 'forecast-temp' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('forecast-temp')}>
                  <h4>7-Day Temperature Forecast</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <ResponsiveContainer width="100%" height={expandedCard === 'forecast-temp' ? 400 : 300}>
                  <ComposedChart data={forecastData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="temperature_2m_max" fill="#ff7300" stroke="#ff7300" name="Max Temp" fillOpacity={0.2} />
                    <Line type="monotone" dataKey="apparent_temperature_max" stroke="#ff0000" name="Feels Like" />
                    {forecastData.map((entry, index) => (
                      entry.isHeatwave && (
                        <Scatter 
                          key={index} 
                          x={index} 
                          y={entry.temperature_2m_max} 
                          fill="#ff0000" 
                          shape={() => <circle cx={0} cy={0} r={8} fill="#ff0000" />}
                        />
                      )
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              
              {/* Heatwave probability chart */}
              <div className={`chart-card ${expandedCard === 'probability' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('probability')}>
                  <h4>Heatwave Probability</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <ResponsiveContainer width="100%" height={expandedCard === 'probability' ? 400 : 300}>
                  <BarChart data={forecastData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="heatwaveProbability" name="Heatwave Probability">
                      {forecastData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.alertColor} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              {/* Alert level distribution */}
              <div className={`chart-card ${expandedCard === 'alert-levels' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('alert-levels')}>
                  <h4>Alert Level Distribution</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <ResponsiveContainer width="100%" height={expandedCard === 'alert-levels' ? 400 : 300}>
                  <PieChart>
                    <Pie
                      data={alertLevelData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={expandedCard === 'alert-levels' ? 150 : 100}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {alertLevelData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              {/* Detailed forecast table */}
              <div className="forecast-table">
                <h4>Detailed Forecast</h4>
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Max Temp</th>
                      <th>Feels Like</th>
                      <th>Humidity</th>
                      <th>Wind</th>
                      <th>Heatwave Prob</th>
                      <th>Alert Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecastData.map((day, index) => (
                      <tr key={index} className={`alert-${day.alert_level.toLowerCase()}`}>
                        <td>{day.date}</td>
                        <td>{day.temperature_2m_max}°C</td>
                        <td>{day.apparent_temperature_max}°C</td>
                        <td>{day.relative_humidity_2m_mean}%</td>
                        <td>{day.wind_speed_10m_max} km/h</td>
                        <td>{day.heatwaveProbability.toFixed(1)}%</td>
                        <td>
                          <span className="alert-badge" style={{ backgroundColor: day.alertColor }}>
                            {day.alert_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          
          {/* Historical tab content */}
          {activeTab === 'historical' && historicalData && (
            <div className="tab-content">
              {/* Historical temperature comparison */}
              <div className={`chart-card ${expandedCard === 'historical-temp' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('historical-temp')}>
                  <h4>Historical Temperature Comparison</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <ResponsiveContainer width="100%" height={expandedCard === 'historical-temp' ? 400 : 300}>
                  <LineChart data={historicalChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="temperature" stroke="#ff7300" activeDot={{ r: 8 }} name="Temperature" />
                    <Line type="monotone" dataKey="feelsLike" stroke="#ff0000" name="Feels Like" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              {/* Historical weather parameters */}
              <div className={`chart-card ${expandedCard === 'historical-params' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('historical-params')}>
                  <h4>Historical Weather Parameters</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <ResponsiveContainer width="100%" height={expandedCard === 'historical-params' ? 400 : 300}>
                  <ComposedChart data={historicalChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                    <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="left" dataKey="humidity" fill="#8884d8" name="Humidity (%)" />
                    <Line yAxisId="right" type="monotone" dataKey="windSpeed" stroke="#82ca9d" name="Wind Speed (km/h)" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              
              {/* Weather comfort index */}
              <div className="comfort-index">
                <h4>Weather Comfort Index</h4>
                <div className="comfort-gauge">
                  <div className="gauge-labels">
                    <span>Cold</span>
                    <span>Cool</span>
                    <span>Pleasant</span>
                    <span>Warm</span>
                    <span>Hot</span>
                  </div>
                  <div className="gauge-track">
                    <div 
                      className="gauge-pointer" 
                      style={{
                        left: `${Math.min(Math.max((parseFloat(result.current_weather.apparent_temperature) + 10) / 50 * 100, 0), 100)}%`
                      }}
                    ></div>
                  </div>
                </div>
                <p className="comfort-description">
                  {parseFloat(result.current_weather.apparent_temperature) < 0 ? "Extremely cold conditions! Bundle up and stay warm." :
                   parseFloat(result.current_weather.apparent_temperature) < 10 ? "Cold conditions. Wear appropriate clothing." :
                   parseFloat(result.current_weather.apparent_temperature) < 20 ? "Cool and comfortable temperature." :
                   parseFloat(result.current_weather.apparent_temperature) < 30 ? "Pleasant weather conditions." :
                   parseFloat(result.current_weather.apparent_temperature) < 35 ? "Warm conditions. Stay hydrated." :
                   "Hot conditions! Stay in shade, drink plenty of water, and avoid strenuous activities."}
                </p>
              </div>
            </div>
          )}
          
          {/* Analysis tab content */}
          {activeTab === 'analysis' && (
            <div className="tab-content">
              {/* Heat index analysis */}
              <div className={`chart-card ${expandedCard === 'heat-index' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('heat-index')}>
                  <h4>Heat Index Analysis</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <ResponsiveContainer width="100%" height={expandedCard === 'heat-index' ? 400 : 300}>
                  <ScatterChart data={heatIndexData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="temperature" name="Temperature (°C)" unit="°C" />
                    <YAxis dataKey="humidity" name="Humidity (%)" unit="%" />
                    <ZAxis dataKey="heatIndex" range={[60, 400]} name="Heat Index" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Legend />
                    <Scatter name="Forecast Days" data={heatIndexData}>
                      {heatIndexData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={alertColors[entry.alertLevel] || '#888'} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              
              {/* Risk factors correlation */}
              <div className={`chart-card ${expandedCard === 'risk-factors' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('risk-factors')}>
                  <h4>Risk Factors Correlation</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <div className="correlation-matrix">
                  <table>
                    <thead>
                      <tr>
                        <th>Factor</th>
                        <th>Temperature</th>
                        <th>Humidity</th>
                        <th>Wind</th>
                        <th>Cloud Cover</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Temperature</td>
                        <td>1.00</td>
                        <td>-0.32</td>
                        <td>-0.18</td>
                        <td>-0.25</td>
                      </tr>
                      <tr>
                        <td>Humidity</td>
                        <td>-0.32</td>
                        <td>1.00</td>
                        <td>0.15</td>
                        <td>0.42</td>
                      </tr>
                      <tr>
                        <td>Wind</td>
                        <td>-0.18</td>
                        <td>0.15</td>
                        <td>1.00</td>
                        <td>0.08</td>
                      </tr>
                      <tr>
                        <td>Cloud Cover</td>
                        <td>-0.25</td>
                        <td>0.42</td>
                        <td>0.08</td>
                        <td>1.00</td>
                      </tr>
                    </tbody>
                  </table>
                  <div className="correlation-note">
                    <p>Correlation values range from -1 (perfect negative) to +1 (perfect positive).</p>
                    <p>Higher temperatures combined with lower humidity increase heatwave risk.</p>
                  </div>
                </div>
              </div>
              
              {/* Heatwave trends */}
              <div className={`chart-card ${expandedCard === 'trends' ? 'expanded' : ''}`}>
                <div className="chart-header" onClick={() => toggleCardExpand('trends')}>
                  <h4>Heatwave Trends</h4>
                  <Info size={18} className="info-icon" />
                </div>
                <div className="trend-analysis">
                  <div className="trend-metric">
                    <div className="trend-value">+2.3°C</div>
                    <div className="trend-label">Average temp increase (last decade)</div>
                  </div>
                  <div className="trend-metric">
                    <div className="trend-value">+18%</div>
                    <div className="trend-label">More heatwave days</div>
                  </div>
                  <div className="trend-metric">
                    <div className="trend-value">+2.1x</div>
                    <div className="trend-label">Heatwave intensity</div>
                  </div>
                  <div className="trend-chart">
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={[
                        { year: '2013', value: 3 },
                        { year: '2015', value: 5 },
                        { year: '2017', value: 7 },
                        { year: '2019', value: 9 },
                        { year: '2021', value: 12 },
                        { year: '2023', value: 15 }
                      ]}>
                        <Area type="monotone" dataKey="value" stroke="#ff7300" fill="#ff7300" fillOpacity={0.2} />
                        <XAxis dataKey="year" />
                        <YAxis />
                        <Tooltip />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div className="trend-caption">Heatwave days per year (2013-2023)</div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Health recommendations */}
          {result.heatwave_alert && (
            <div className="health-recommendations">
              <h3>Heat Safety Recommendations</h3>
              <div className="recommendations-grid">
                <div className="recommendation">
                  <span className="recommendation-icon">💧</span>
                  <span className="recommendation-text">Drink plenty of water (2-4 cups per hour when working outside)</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">🏠</span>
                  <span className="recommendation-text">Stay in air-conditioned buildings as much as possible</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">👕</span>
                  <span className="recommendation-text">Wear lightweight, light-colored, loose-fitting clothing</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">⏰</span>
                  <span className="recommendation-text">Schedule outdoor activities carefully (before noon or in evening)</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">🚿</span>
                  <span className="recommendation-text">Take cool showers or baths to cool down</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">🧓</span>
                  <span className="recommendation-text">Check on at-risk friends, family and neighbors twice daily</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">🚗</span>
                  <span className="recommendation-text">Never leave children or pets in cars</span>
                </div>
                <div className="recommendation">
                  <span className="recommendation-icon">🏥</span>
                  <span className="recommendation-text">Know the signs of heat illness and what to do</span>
                </div>
              </div>
              
              {/* Heat illness symptoms */}
              <div className="symptoms-info">
                <h4>Recognize Heat Illness Symptoms</h4>
                <div className="symptoms-grid">
                  <div className="symptom">
                    <div className="symptom-title">Heat Cramps</div>
                    <ul>
                      <li>Muscle pains or spasms</li>
                      <li>Heavy sweating during exercise</li>
                    </ul>
                  </div>
                  <div className="symptom">
                    <div className="symptom-title">Heat Exhaustion</div>
                    <ul>
                      <li>Heavy sweating</li>
                      <li>Cold, pale, clammy skin</li>
                      <li>Fast, weak pulse</li>
                      <li>Nausea or vomiting</li>
                    </ul>
                  </div>
                  <div className="symptom">
                    <div className="symptom-title">Heat Stroke</div>
                    <ul>
                      <li>High body temperature (103°F+)</li>
                      <li>Hot, red, dry or damp skin</li>
                      <li>Fast, strong pulse</li>
                      <li>Loss of consciousness</li>
                    </ul>
                  </div>
                </div>
                <div className="emergency-note">
                  <AlertTriangle size={24} className="emergency-icon" />
                  <span>Heat stroke is a medical emergency. Call 911 immediately if symptoms appear.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;