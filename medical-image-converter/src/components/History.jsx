import React, { useState, useEffect } from 'react';
import { Download, Trash2, ArrowLeft, Image as ImageIcon, Zap, Clock, Star } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const History = () => {
  const [historyData, setHistoryData] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const savedHistory = JSON.parse(localStorage.getItem('conversionHistory') || '[]');
    setHistoryData(savedHistory);
  }, []);

  const handleDownload = (entry) => {
    const link = document.createElement('a');
    link.href = entry.outputImage;
    link.download = `${entry.conversionType || entry.mode}_${entry.timestamp}.png`;
    link.click();
  };

  const handleExportAll = () => {
    alert('Export all as ZIP is not yet implemented');
  };

  const handleClear = () => {
    if (window.confirm('Clear all history? This action cannot be undone.')) {
      localStorage.removeItem('conversionHistory');
      setHistoryData([]);
    }
  };

  const handleBookmark = (entry) => {
    const prev = JSON.parse(localStorage.getItem('bookmarkedResults') || '[]');
    const isDuplicate = prev.some(e => e.timestamp === entry.timestamp);
    if (isDuplicate) return alert("Already bookmarked!");

    localStorage.setItem('bookmarkedResults', JSON.stringify([entry, ...prev]));
    alert("Bookmarked successfully!");
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getModeColor = (mode) => {
    switch(mode) {
      case 'conversion': return 'bg-blue-100 text-blue-800';
      case 'segmentation': return 'bg-green-100 text-green-800';
      case 'overlay': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getModeLabel = (mode) => {
    switch(mode) {
      case 'conversion': return 'Conversion';
      case 'segmentation': return 'Tumor Segmentation';
      case 'overlay': return 'Tissue Overlay';
      default: return 'Result';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">History</h1>
            <p className="text-sm text-gray-500 dark:text-gray-300">View and manage your past conversions and segmentations</p>
          </div>
          <button
            onClick={() => navigate('/converter')}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <ArrowLeft size={16} className="mr-2" />
            Back to Converter
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {/* Controls */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
          <div className="flex items-center space-x-3">
            <button
              onClick={handleExportAll}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <Download size={16} className="mr-2" />
              Export All
            </button>
            <button
              onClick={handleClear}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              <Trash2 size={16} className="mr-2" />
              Clear History
            </button>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-300">
            Showing {historyData.length} {historyData.length === 1 ? 'entry' : 'entries'}
          </p>
        </div>

        {/* No Data */}
        {historyData.length === 0 ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-8 text-center">
            <div className="mx-auto h-24 w-24 text-gray-400 dark:text-gray-500">
              <Clock size={24} className="mx-auto" />
            </div>
            <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-gray-100">No history found</h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Perform a conversion or segmentation to see results here.</p>
            <div className="mt-6">
              <button
                onClick={() => navigate('/converter')}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Go to Converter
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {historyData.map((entry, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 overflow-hidden shadow-sm rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow duration-200"
              >
                {/* Header */}
                <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getModeColor(entry.mode)}`}>
                    {getModeLabel(entry.mode)}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {formatDate(entry.timestamp)}
                  </span>
                </div>

                {/* Image(s) */}
                <div className="p-4">
                  {entry.mode === 'overlay' ? (
                    entry.hardOverlay && (
                      <div className="mb-4 rounded-md overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                          src={entry.hardOverlay}
                          alt="Overlay result"
                          className="w-full h-48 object-contain bg-gray-50 dark:bg-gray-900"
                        />
                      </div>
                    )
                  ) : (
                    <div className="mb-4 rounded-md overflow-hidden border border-gray-200 dark:border-gray-700">
                      <img
                        src={entry.outputImage}
                        alt="Output result"
                        className="w-full h-48 object-contain bg-gray-50 dark:bg-gray-900"
                      />
                    </div>
                  )}

                  {entry.extraImage && (
                    <div className="mb-4 rounded-md overflow-hidden border border-gray-200 dark:border-gray-700">
                      <img
                        src={entry.extraImage}
                        alt="Additional result"
                        className="w-full h-48 object-contain bg-gray-50 dark:bg-gray-900"
                      />
                    </div>
                  )}

                  {/* Details */}
                  <div className="space-y-2">
                    {entry.conversionType && (
                      <p className="text-sm">
                        <span className="font-medium text-gray-700 dark:text-gray-200">Conversion:</span>{' '}
                        <span className="text-gray-600 dark:text-gray-400">
                          {entry.conversionType.replace(/-/g, ' ').toUpperCase()}
                        </span>
                      </p>
                    )}
                    <p className="text-sm">
                      <span className="font-medium text-gray-700 dark:text-gray-200">Model:</span>{' '}
                      <span className="text-gray-600 dark:text-gray-400">{entry.modelName || 'U-Net'}</span>
                    </p>
                  </div>

                  {/* Metrics */}
                  <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                    <div className="bg-gray-50 dark:bg-gray-900 p-2 rounded">
                      <Zap size={16} className="mx-auto text-blue-500" />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">SSIM</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {entry.metrics?.ssim?.toFixed(4) || '–'}
                      </p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900 p-2 rounded">
                      <ImageIcon size={16} className="mx-auto text-blue-500" />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">PSNR</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {entry.metrics?.psnr?.toFixed(2) || '–'} dB
                      </p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900 p-2 rounded">
                      <Clock size={16} className="mx-auto text-blue-500" />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">LPIPS</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {entry.metrics?.lpips?.toFixed(4) || '–'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="px-4 py-3 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 flex justify-between space-x-3">
                  <button
                    onClick={() => handleDownload(entry)}
                    className="flex-1 inline-flex justify-center items-center px-3 py-2 border border-gray-300 dark:border-gray-700 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  >
                    <Download size={16} className="mr-2" />
                    Download
                  </button>
                  <button
                    onClick={() => handleBookmark(entry)}
                    className="flex-1 inline-flex justify-center items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-yellow-500 hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
                  >
                    <Star size={16} className="mr-2" />
                    Bookmark
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default History;