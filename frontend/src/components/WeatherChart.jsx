// src/components/WeatherChart.jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const WeatherChart = ({ data }) => (
  <div className="chart-container">
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="day" />
        <YAxis unit="°C" />
        <Tooltip />
        <Line
          type="monotone"
          dataKey="temperature"
          stroke="#ff7300"
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);
