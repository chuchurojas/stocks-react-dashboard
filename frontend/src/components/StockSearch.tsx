import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, X } from 'lucide-react';
import { searchStock } from '../services/api';

interface SearchResult {
  ticker: string;
  company: string;
}

interface StockSearchProps {
  onStockSelect: (ticker: string) => void;
}

const StockSearch: React.FC<StockSearchProps> = ({ onStockSelect }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [recentStocks, setRecentStocks] = useState<SearchResult[]>([]);

  // Load recent stocks from localStorage on component mount
  useEffect(() => {
    const saved = localStorage.getItem('recentStocks');
    if (saved) {
      try {
        setRecentStocks(JSON.parse(saved));
      } catch (error) {
        console.error('Error loading recent stocks:', error);
      }
    }
  }, []);

  useEffect(() => {
    if (query.length >= 2) {
      setLoading(true);
      searchStock(query)
        .then((result) => {
          setSuggestions([result]);
          setShowSuggestions(true);
        })
        .catch(() => {
          setSuggestions([]);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onStockSelect(query.trim().toUpperCase());
      setQuery('');
      setShowSuggestions(false);
    }
  };

  const saveRecentStock = (result: SearchResult) => {
    const updatedRecent = [
      result,
      ...recentStocks.filter(stock => stock.ticker !== result.ticker)
    ].slice(0, 8); // Keep only the 8 most recent
    
    setRecentStocks(updatedRecent);
    localStorage.setItem('recentStocks', JSON.stringify(updatedRecent));
  };

  const handleSuggestionClick = (result: SearchResult) => {
    saveRecentStock(result);
    onStockSelect(result.ticker);
    setQuery('');
    setShowSuggestions(false);
  };

  const handleRecentClick = (result: SearchResult) => {
    saveRecentStock(result);
    onStockSelect(result.ticker);
  };

  const removeRecentStock = (tickerToRemove: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent triggering the click handler
    if (window.confirm(`Remove ${tickerToRemove} from recent stocks?`)) {
      const updatedRecent = recentStocks.filter(stock => stock.ticker !== tickerToRemove);
      setRecentStocks(updatedRecent);
      localStorage.setItem('recentStocks', JSON.stringify(updatedRecent));
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by ticker or company name..."
            className="w-full pl-10 pr-4 py-2 bg-dark-700 border border-dark-600 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          {loading && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500"></div>
            </div>
          )}
        </div>
        
        {showSuggestions && suggestions.length > 0 && (
          <div className="absolute z-10 w-full mt-1 bg-dark-700 border border-dark-600 rounded-md shadow-lg">
            {suggestions.map((result, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(result)}
                className="w-full px-4 py-3 text-left hover:bg-dark-600 first:rounded-t-md last:rounded-b-md"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-medium text-white">{result.ticker}</div>
                    <div className="text-xs text-gray-400">{result.company}</div>
                  </div>
                   <div className="text-xs text-gray-500">
                     {result.ticker && result.ticker.includes('.') ? 'International' : 'US'}
                   </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </form>

      {recentStocks.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-gray-400">Recently Used</h4>
            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all recent stocks?')) {
                  setRecentStocks([]);
                  localStorage.removeItem('recentStocks');
                }
              }}
              className="text-xs text-gray-500 hover:text-red-400 transition-colors"
            >
              Clear All
            </button>
          </div>
          <div className="grid grid-cols-1 gap-2">
            {recentStocks.map((stock) => (
              <div
                key={stock.ticker}
                className="flex items-center justify-between px-3 py-2 bg-dark-700 hover:bg-dark-600 rounded-md text-sm transition-colors group"
              >
                <button
                  onClick={() => handleRecentClick(stock)}
                  className="flex items-center flex-1 text-left"
                >
                  <TrendingUp className="h-3 w-3 mr-2" />
                  <div className="flex-1">
                    <div className="font-medium">{stock.ticker}</div>
                    <div className="text-xs text-gray-400">{stock.company}</div>
                  </div>
                   <div className="text-xs text-gray-500 mr-2">
                     {stock.ticker && stock.ticker.includes('.') ? 'International' : 'US'}
                   </div>
                </button>
                <button
                  onClick={(e) => removeRecentStock(stock.ticker, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-900/30 rounded transition-all duration-200"
                  title="Remove from recent"
                >
                  <X className="h-3 w-3 text-gray-400 hover:text-red-400" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default StockSearch;

