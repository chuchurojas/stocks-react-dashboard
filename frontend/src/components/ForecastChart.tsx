import React, { useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ForecastChartProps {
  historicalData: {
    dates: string[];
    close: number[];
  };
  forecastData: {
    forecast_data: {
      dates: string[];
      prices: number[];
    };
    summary: {
      current_price: number;
      forecast_price: number;
      price_change: number;
      percent_change: number;
    };
  };
  method: string;
  currency?: string;
}

const ForecastChart: React.FC<ForecastChartProps> = ({ 
  historicalData, 
  forecastData, 
  method,
  currency = 'USD' 
}) => {
  const chartRef = useRef<ChartJS<'line'>>(null);

  // Handle undefined data gracefully
  if (!historicalData || !forecastData || !historicalData.dates || !historicalData.close || !forecastData.forecast_data) {
    return (
      <div className="h-96 flex items-center justify-center bg-dark-700 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading forecast data...</p>
        </div>
      </div>
    );
  }

  // Format currency helper
  const formatCurrency = (value: number) => {
    if (currency === 'GBP') {
      return '£' + value.toFixed(2);
    } else if (currency === 'EUR') {
      return '€' + value.toFixed(2);
    } else if (currency === 'JPY') {
      return '¥' + value.toFixed(0);
    } else {
      return '$' + value.toFixed(2);
    }
  };

  // Combine historical and forecast data
  const allDates = [...historicalData.dates, ...forecastData.forecast_data.dates];

  const chartData = {
    labels: allDates,
    datasets: [
      {
        label: 'Historical Price',
        data: [...historicalData.close, ...Array(forecastData.forecast_data.prices.length).fill(null)],
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: `${method.charAt(0).toUpperCase() + method.slice(1)} Forecast`,
        data: [...Array(historicalData.close.length).fill(null), ...forecastData.forecast_data.prices],
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderDash: [5, 5],
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#9ca3af',
          usePointStyle: true,
        },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        backgroundColor: 'rgba(31, 41, 55, 0.95)',
        titleColor: '#f9fafb',
        bodyColor: '#f9fafb',
        borderColor: '#374151',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: '#9ca3af',
          maxTicksLimit: 15,
        },
      },
      y: {
        display: true,
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: '#9ca3af',
          callback: function(value: any) {
            return formatCurrency(value);
          },
        },
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
  };

  return (
    <div className="space-y-4">
      <div className="h-96">
        <Line ref={chartRef} data={chartData} options={options} />
      </div>
      
      {/* Forecast Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="text-sm text-gray-400">Current Price</div>
          <div className="text-xl font-semibold">
            {formatCurrency(forecastData.summary.current_price)}
          </div>
        </div>
        
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="text-sm text-gray-400">Forecast Price</div>
          <div className="text-xl font-semibold">
            {formatCurrency(forecastData.summary.forecast_price)}
          </div>
        </div>
        
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="text-sm text-gray-400">Expected Change</div>
          <div className={`text-xl font-semibold ${
            forecastData.summary.percent_change >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {forecastData.summary.percent_change >= 0 ? '+' : ''}
            {forecastData.summary.percent_change.toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastChart;

