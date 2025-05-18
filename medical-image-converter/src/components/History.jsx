import React, { useState, useEffect } from 'react';
import { Download, Trash2, ArrowLeft, Image as ImageIcon, Zap, Clock } from 'lucide-react';
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
    if (window.confirm('Clear all history?')) {
      localStorage.removeItem('conversionHistory');
      setHistoryData([]);
    }
  };

  const handleBookmark = (entry) => {
    const prev = JSON.parse(localStorage.getItem('bookmarkedResults') || '[]');
    const isDuplicate = prev.some(e => e.timestamp === entry.timestamp);
    if (isDuplicate) return alert("Already bookmarked!");

    localStorage.setItem('bookmarkedResults', JSON.stringify([entry, ...prev]));
    alert("★ Bookmarked successfully!");
  };

  return (
    <div className="p-6 text-white bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Conversion & Segmentation History</h1>
        <button
          onClick={() => navigate('/converter')}
          className="flex items-center gap-2 text-sm bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded"
        >
          <ArrowLeft size={16} /> Back
        </button>
      </div>

      {/* Controls */}
      <div className="flex justify-between items-center mb-8">
        <button
          onClick={handleExportAll}
          className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded flex items-center gap-2"
        >
          <Download size={16} /> Export All
        </button>
        <button
          onClick={handleClear}
          className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded flex items-center gap-2"
        >
          <Trash2 size={16} /> Clear History
        </button>
      </div>

      {/* No Data */}
      {historyData.length === 0 ? (
        <p className="text-gray-400 text-center">No history found. Perform a conversion or segmentation first.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {historyData.map((entry, idx) => (
            <div
              key={idx}
              className="bg-gray-800 hover:shadow-xl transition-shadow duration-200 p-4 rounded-lg border border-gray-700 flex flex-col"
            >
              {/* Tag */}
              <span className={`text-xs font-semibold px-2 py-1 mb-2 rounded-full w-max
                ${entry.mode === 'conversion'
                  ? 'bg-purple-900 text-purple-300'
                  : entry.mode === 'overlay'
                    ? 'bg-yellow-900 text-yellow-300'
                    : 'bg-blue-900 text-blue-300'
                }`}
              >
                {entry.mode === 'overlay' ? 'Overlay Segmentation' : entry.mode.charAt(0).toUpperCase() + entry.mode.slice(1)}
              </span>

              {/* Main Image(s) */}
              {entry.mode === 'overlay' ? (
                <>
                  {/* Show only final overlay on input */}
                  {entry.hardOverlay && (
                    <div className="overflow-hidden rounded-lg border border-gray-600 mb-3">
                      <img
                        src={entry.hardOverlay}
                        alt="Overlay on Input"
                        className="w-full object-contain transition-transform duration-200 hover:scale-105"
                      />
                    </div>
                  )}
                </>
              ) : (
                <div className="overflow-hidden rounded-lg border border-gray-600 mb-3">
                  <img
                    src={entry.outputImage}
                    alt="Output"
                    className="w-full object-contain transition-transform duration-200 hover:scale-105"
                  />
                </div>
              )}


              {/* Extra Output */}
              {entry.extraImage && (
                <div className="overflow-hidden rounded-lg border border-gray-600 mb-3">
                  <img
                    src={entry.extraImage}
                    alt="Extra"
                    className="w-full object-contain transition-transform duration-200 hover:scale-105"
                  />
                </div>
              )}

              {/* Info */}
              <div className="text-sm space-y-1 text-gray-300">
                {entry.conversionType && (
                  <p><strong>Conversion:</strong> {entry.conversionType.replace(/-/g, ' ').toUpperCase()}</p>
                )}
                <p><strong>Model:</strong> {entry.modelName || 'U-Net'}</p>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-2 text-center mt-3 text-xs text-gray-400">
                <div className="bg-gray-700 p-2 rounded">
                  <Zap className="mx-auto mb-1" size={14} />
                  SSIM<br />
                  <span className="text-white">{entry.metrics?.ssim?.toFixed(4) || '–'}</span>
                </div>
                <div className="bg-gray-700 p-2 rounded">
                  <ImageIcon className="mx-auto mb-1" size={14} />
                  PSNR<br />
                  <span className="text-white">{entry.metrics?.psnr?.toFixed(2) || '–'} dB</span>
                </div>
                <div className="bg-gray-700 p-2 rounded">
                  <Clock className="mx-auto mb-1" size={14} />
                  LPIPS<br />
                  <span className="text-white">{entry.metrics?.lpips?.toFixed(4) || '–'}</span>
                </div>
              </div>

              {/* Timestamp */}
              <p className="text-xs text-gray-500 mt-3 italic text-right">
                {new Date(entry.timestamp).toLocaleString()}
              </p>

              {/* Download */}
              <button
                onClick={() => handleDownload(entry)}
                className="mt-4 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm w-full"
              >
                Download Output
              </button>
              <button
                onClick={() => handleBookmark(entry)}
                className="mt-2 bg-yellow-600 hover:bg-yellow-700 px-4 py-1 rounded text-sm w-full"
              >
                ★ Bookmark
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default History;
