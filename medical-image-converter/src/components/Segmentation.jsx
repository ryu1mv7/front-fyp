import React, { useState } from 'react';

const Segmentation = () => {
  const [segmentationResult, setSegmentationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSegmentation = async (imageFile) => {
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);

    try {
      const response = await fetch('http://localhost:5000/api/segment/', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setSegmentationResult(data.result);

    } catch (err) {
      console.error(err);
      setError('Segmentation failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4 bg-gray-700 rounded">
      <button 
        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
        onClick={() => handleSegmentation(/* pass imageFile */)}
      >
        Run Segmentation
      </button>

      {isLoading && <p>Processing...</p>}
      {error && <p className="text-red-400">{error}</p>}

      {segmentationResult && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold">Segmentation Result</h3>
          <img src={segmentationResult} alt="Segmentation Output" className="w-full rounded border border-gray-600" />
        </div>
      )}
    </div>
  );
};

export default Segmentation;
