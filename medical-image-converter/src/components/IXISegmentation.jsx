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
    <div className="p-6 bg-gray-800 text-white rounded-lg max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold mb-2 text-blue-400">Brain Tissue Segmentation (T1-weighted MRI)</h2>
      <p className="text-sm text-gray-300 mb-4">
        Upload a T1-weighted brain scan (in NIfTI format) to visualize segmented brain tissues:
        cerebrospinal fluid (CSF), gray matter (GM), white matter (WM), and background. This segmentation 
        aids in structural analysis and neuroimaging studies.
      </p>

      {/* Upload */}
      <input type="file" name="t1" accept=".nii,.nii.gz" onChange={handleChange} className="mb-4" />
      <button
        onClick={handleSubmit}
        disabled={!inputFile || isLoading}
        className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition"
      >
        {isLoading ? 'Segmenting...' : '▶ Run Segmentation'}
      </button>

      {error && <p className="text-red-400 mt-4">{error}</p>}

      {/* Input Preview */}
      {preview && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-cyan-300 mb-2">Uploaded T1 MRI Scan</h3>
          <img src={preview} alt="Input preview" className="w-64 border rounded shadow" />
        </div>
      )}

      {/* Soft Overlays */}
      {outputs.overlays.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-cyan-300 mb-2">Soft Tissue Probabilities</h3>
          <p className="text-sm text-gray-400 mb-4">
            These overlays represent the probability maps for each tissue type (CSF, GM, WM, BG). 
            Brighter pixels represent higher confidence in the detected region.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {outputs.overlays.map((img, idx) => (
              <img key={idx} src={img} alt={`Overlay ${idx}`} className="w-64 border rounded" />
            ))}
          </div>
        </div>
      )}

      {/* Hard Segmentation */}
      {outputs.hardSeg && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-cyan-300 mb-2">Final Segmentation Map</h3>
          <p className="text-sm text-gray-400 mb-4">
            This map shows the discrete classification of each pixel into one of the four tissue types.
            Useful for quantitative volume calculations and 3D modeling.
          </p>
          <img src={outputs.hardSeg} alt="Segmentation" className="w-64 border rounded shadow" />
        </div>
      )}

      {/* Overlay on Input */}
      {outputs.hardOverlay && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-cyan-300 mb-2">Overlay on T1 MRI</h3>
          <p className="text-sm text-gray-400 mb-4">
            The predicted segmentation mask is overlaid on the original input scan to help you visually 
            verify how well the model aligned tissue regions.
          </p>
          <img src={outputs.hardOverlay} alt="Overlay" className="w-64 border rounded shadow" />
        </div>
      )}
    </div>
  );
};

export default IXISegmentation;
