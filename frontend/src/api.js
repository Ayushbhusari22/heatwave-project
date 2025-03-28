const API_BASE_URL='http://localhost:5000';
// API functions for heatwave prediction app
export const getCitySuggestions=async (query) => {
    const response=await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=5&language=en&format=json`
    );
    return response.json();
};

// Function to check heatwave status for a city

export const checkHeatwave=async (city) => {
    try {
        const response=await fetch(`${API_BASE_URL}/api/predict`,{
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({city})
        });

        if(!response.ok) {
            const errorData=await response.json();
            throw new Error(errorData.error||'Failed to get prediction');
        }

        const data=await response.json();

        // Transform response to match frontend expectations
        return {
            city: data.city,
            heatwave_alert: data.predictions?.some(p => p.is_heatwave===1)||false,
            message: data.message,
            forecast: data.predictions||[],
            current_weather: {
                temperature: data.predictions?.[0]?.temperature_2m_max||0,
                apparent_temperature: data.predictions?.[0]?.apparent_temperature_max||0,
                humidity: data.predictions?.[0]?.relative_humidity_2m_mean||0,
                wind_speed: data.predictions?.[0]?.wind_speed_10m_max||0,
                cloud_cover: data.predictions?.[0]?.cloud_cover_mean||0,
                precipitation: data.predictions?.[0]?.precipitation_sum||0
            }
        };
    } catch(error) {
        console.error('API Error:',error);
        throw new Error(error.message||'Network request failed');
    }
};

export const fetchHistoricalData=async (city) => {
    try {
        const response=await fetch(`${API_BASE_URL}/api/historical?city=${encodeURIComponent(city)}`);
        if(!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data=await response.json();
        return data.historical_data;
    } catch(error) {
        console.error('Error fetching historical data:',error);
        throw error;
    }
};

// Helper function to format date as YYYY-MM-DD
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

// Helper function to format date for display
function formatDisplayDate(dateStr) {
    const date=new Date(dateStr);
    return date.toLocaleDateString('en-US',{month: 'short',day: 'numeric'});
}
