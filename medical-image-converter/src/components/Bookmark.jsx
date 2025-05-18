import React, { useEffect, useState } from 'react';
import { ArrowLeft, Trash2, Download } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Bookmark = () => {
  const [bookmarks, setBookmarks] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem('bookmarkedResults') || '[]');
    setBookmarks(saved);
  }, []);

  const handleClearBookmarks = () => {
    if (window.confirm("Clear all bookmarks?")) {
      localStorage.removeItem('bookmarkedResults');
      setBookmarks([]);
    }
  };

  const handleDownload = (entry) => {
    const link = document.createElement('a');
    link.href = entry.outputImage;
    link.download = `${entry.mode}_${entry.timestamp}.png`;
    link.click();
  };

  return (
    <div className="p-6 text-white bg-gray-900 min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">★ Bookmarked Outputs</h1>
        <button
          onClick={() => navigate('/converter')}
          className="flex items-center gap-2 text-sm bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded"
        >
          <ArrowLeft size={16} /> Back
        </button>
      </div>

    <div className="flex justify-between mb-6">
    <button
        onClick={() => alert('Export all as ZIP not implemented yet')}
        className="flex items-center gap-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
    >
        <Download size={16} /> Export All
    </button>

    <button
        onClick={handleClearBookmarks}
        className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded flex items-center gap-2"
    >
        <Trash2 size={16} /> Clear All
    </button>
    </div>

      {bookmarks.length === 0 ? (
        <p className="text-gray-400 text-center">No bookmarks yet. Use ★ in history to save results.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {bookmarks.map((entry, i) => (
            <div key={i} className="bg-gray-800 p-4 rounded-lg border border-gray-700">
              <span className="text-xs px-2 py-1 mb-2 inline-block rounded-full bg-yellow-900 text-yellow-300 font-semibold">
                ★ {entry.mode.charAt(0).toUpperCase() + entry.mode.slice(1)}
              </span>

              <div className="overflow-hidden border border-gray-600 mb-2">
                <img src={entry.outputImage} alt="Output" className="w-full object-contain" />
              </div>

              {entry.extraImage && (
                <div className="overflow-hidden border border-gray-600 mb-2">
                  <img src={entry.extraImage} alt="Extra" className="w-full object-contain" />
                </div>
              )}

              {entry.overlays && entry.overlays.length > 0 && (
                <div className="grid grid-cols-2 gap-1 mb-2">
                  {entry.overlays.map((img, idx) => (
                    <img key={idx} src={img} alt={`Overlay ${idx}`} className="w-full border border-gray-600" />
                  ))}
                </div>
              )}

              <div className="text-sm text-gray-400 mt-2">
                <p><strong>Model:</strong> {entry.modelName || '–'}</p>
                <p><strong>Timestamp:</strong> {new Date(entry.timestamp).toLocaleString()}</p>
              </div>

              <button
                onClick={() => handleDownload(entry)}
                className="mt-4 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm w-full"
              >
                Download Output
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Bookmark;
