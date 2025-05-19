import React, { useState } from 'react';
import { Upload, Play, RefreshCw } from 'lucide-react';

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
      const res = await fetch('http://localhost:5000/api/ixi-segment/', {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) throw new Error('Segmentation failed');
      const data = await res.json();
      setOutputs(data);

      const historyEntry = {
        mode: 'overlay',
        modelName: 'IXI Simple U-Net',
        inputImage: preview,
        overlays: data.overlays,
        hardSeg: data.hardSeg,
        hardOverlay: data.hardOverlay,
        timestamp: Date.now()
      };

      const currentHistory = JSON.parse(localStorage.getItem('conversionHistory') || '[]');
      localStorage.setItem('conversionHistory', JSON.stringify([historyEntry, ...currentHistory]));
    } catch (err) {
      setError(err.message || 'An error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderImageCard = (title, imageUrl, description) => (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
      </div>
      <div className="p-4 h-64 flex items-center justify-center bg-gray-50">
        {imageUrl ? (
          <img 
            src={imageUrl} 
            alt={title} 
            className="max-w-full max-h-full object-contain" 
          />
        ) : (
          <div className="text-center text-gray-400">
            <p>{description || "Result will appear here"}</p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-xl shadow-sm overflow-hidden max-w-6xl mx-auto">
      {/* Header */}
      <div className="bg-blue-600 px-6 py-4">
        <h2 className="text-xl font-semibold text-white">Brain Tissue Segmentation</h2>
        <p className="text-indigo-100 text-sm mt-1">
          T1-weighted MRI analysis for CSF, GM, and WM visualization
        </p>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {/* File Upload Section */}
        <div className="mb-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upload T1 MRI Scan</h3>
          <div className="flex items-center">
            <label className="flex-1 flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 cursor-pointer">
              <Upload size={16} className="mr-2" />
              Select NIfTI File
              <input 
                type="file" 
                className="hidden" 
                accept=".nii,.nii.gz" 
                onChange={handleChange} 
              />
            </label>
            {inputFile && (
              <span className="ml-3 text-sm text-gray-500 truncate max-w-xs">
                {inputFile.name}
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-gray-500">
            T1-weighted MRI in NIfTI format (.nii or .nii.gz)
          </p>
        </div>

        {/* Action Button */}
        <div className="flex justify-center mb-8">
        <button
            className={`px-6 py-3 rounded-md shadow-sm text-white font-medium flex items-center ${
              isLoading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            onClick={handleSubmit}
            disabled={!inputFile || isLoading}
          >
            {isLoading ? (
              <>
                <RefreshCw size={18} className="mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play size={18} className="mr-2" />
                Run Segmentation
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-400">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Sections */}
        {preview && (
          <div className="mb-8">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Input Preview</h3>
            {renderImageCard("Uploaded T1 MRI Scan", preview)}
          </div>
        )}

        {outputs.overlays.length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Soft Tissue Probabilities</h3>
            <p className="text-sm text-gray-500 mb-4">
              Probability maps showing confidence levels for each tissue type (CSF, GM, WM, BG).
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {outputs.overlays.map((img, idx) => (
                renderImageCard(
                  `Tissue Map ${idx + 1}`,
                  img,
                  `Probability map for tissue type ${idx + 1}`
                )
              ))}
            </div>
          </div>
        )}

        {outputs.hardSeg && (
          <div className="mb-8">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Final Segmentation</h3>
            <p className="text-sm text-gray-500 mb-4">
              Discrete classification of each pixel into tissue types.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {renderImageCard("Segmentation Map", outputs.hardSeg)}
              {renderImageCard("Overlay on T1 MRI", outputs.hardOverlay)}
            </div>
          </div>
        )}
      </div>

      {/* Info Section */}
      <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
        <h3 className="text-sm font-medium text-gray-700 mb-2">About Tissue Segmentation</h3>
        <p className="text-xs text-gray-500">
          This model uses a U-Net architecture trained on the IXI dataset to segment T1-weighted MRI scans
          into cerebrospinal fluid (CSF), gray matter (GM), white matter (WM), and background. Results are
          suitable for research and educational purposes.
        </p>
      </div>
    </div>
  );
};

export default IXISegmentation;