import React, { useState } from 'react';
import axios from 'axios';

const IXISegmentation = () => {
  const [inputFile, setInputFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [outputs, setOutputs] = useState({
    overlays: [],
    hardSeg: null,
    hardOverlay: null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const file = e.target.files[0];
    setInputFile(file);
    setPreview(URL.createObjectURL(file));
    setOutputs({ overlays: [], hardSeg: null, hardOverlay: null });
  };

  const handleSubmit = async () => {
    if (!inputFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('t1', inputFile);

    try {
    const res = await axios.post('http://localhost:5000/api/ixi-segment/', formData);
    setOutputs(res.data);

    // Log to History
    const historyEntry = {
        mode: 'overlay',
        modelName: 'IXI Simple U-Net',
        inputImage: preview,
        overlays: res.data.overlays,
        hardSeg: res.data.hardSeg,
        hardOverlay: res.data.hardOverlay,
        timestamp: Date.now()
    };

    const currentHistory = JSON.parse(localStorage.getItem('conversionHistory') || '[]');
    localStorage.setItem('conversionHistory', JSON.stringify([historyEntry, ...currentHistory]));
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">IXI Multiclass Segmentation</h2>

      <input type="file" name="t1" accept=".nii,.nii.gz" onChange={handleChange} />
      <button
        onClick={handleSubmit}
        disabled={!inputFile || isLoading}
        className="bg-blue-600 text-white px-4 py-2 mt-4 rounded"
      >
        {isLoading ? 'Segmenting...' : 'Run Segmentation'}
      </button>

      {error && <p className="text-red-600 mt-4">{error}</p>}

      {preview && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold">Input T1 Image</h3>
          <img src={preview} alt="Input preview" className="w-64 border" />
        </div>
      )}

      {outputs.overlays.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Overlay Channels</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {outputs.overlays.map((img, idx) => (
              <img key={idx} src={img} alt={`Overlay ${idx}`} className="w-64 border" />
            ))}
          </div>
        </div>
      )}

      {outputs.hardSeg && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Predicted Segmentation (Hard Labels)</h3>
          <img src={outputs.hardSeg} alt="Segmentation" className="w-64 border" />
        </div>
      )}

      {outputs.hardOverlay && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Overlay on Input</h3>
          <img src={outputs.hardOverlay} alt="Overlay" className="w-64 border" />
        </div>
      )}
    </div>
  );
};

export default IXISegmentation;
