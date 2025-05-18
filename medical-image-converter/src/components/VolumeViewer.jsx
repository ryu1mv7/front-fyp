import React, { useState, useEffect } from 'react';

const VolumeViewer = ({ slices }) => {
  const [index, setIndex] = useState(0);
  const [autoScroll, setAutoScroll] = useState(false);

  const next = () => setIndex((i) => Math.min(i + 1, slices.length - 1));
  const prev = () => setIndex((i) => Math.max(i - 1, 0));

  // Auto-scroll functionality
  useEffect(() => {
    let interval = null;
    if (autoScroll) {
      interval = setInterval(() => {
        setIndex((i) => (i + 1) % slices.length);
      }, 120); // speed of playback
    }
    return () => clearInterval(interval);
  }, [autoScroll, slices.length]);

  // Scroll wheel handler
  const handleWheel = (e) => {
    if (e.deltaY > 0) next();
    else prev();
  };

  return (
    <div className="bg-gray-900 p-4 rounded-lg text-white w-full">
      <h3 className="text-lg font-semibold mb-2">3D Slice Viewer</h3>

      <div onWheel={handleWheel} className="flex justify-center mb-2">
        <img
          src={slices[index]}
          alt={`Slice ${index}`}
          className="max-w-lg border border-gray-600 rounded"
        />
      </div>

      <input
        type="range"
        min="0"
        max={slices.length - 1}
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="w-full accent-blue-500 mb-4"
      />

      <div className="flex justify-between items-center">
        <button onClick={prev} className="bg-gray-700 px-4 py-2 rounded hover:bg-gray-600">
          ←
        </button>
        <span className="text-sm">{index + 1} / {slices.length}</span>
        <button onClick={next} className="bg-gray-700 px-4 py-2 rounded hover:bg-gray-600">
          →
        </button>
      </div>

      <div className="flex justify-center mt-4">
        <button
          onClick={() => setAutoScroll(!autoScroll)}
          className="bg-blue-600 px-4 py-1 rounded hover:bg-blue-700"
        >
          {autoScroll ? 'Pause' : 'Play'}
        </button>
      </div>
    </div>
  );
};

export default VolumeViewer;
