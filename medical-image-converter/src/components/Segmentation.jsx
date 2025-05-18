import React, { useState } from 'react';
import { Upload, Play, RefreshCw } from 'lucide-react';

const Segmentation = () => {
  const [inputs, setInputs] = useState({ t1n: null, t1ce: null, t2: null });
  const [previews, setPreviews] = useState({ t1n: null });
  const [outputs, setOutputs] = useState({ t2fUrl: null, segUrl: null });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const { name, files } = e.target;
    if (files.length > 0) {
      const file = files[0];
      setInputs(prev => ({ ...prev, [name]: file }));
      if (name === 't1n') {
        setPreviews(prev => ({ ...prev, t1n: URL.createObjectURL(file) }));
      }
    }
  };

  const handleSubmit = async () => {
    const { t1n, t1ce, t2 } = inputs;
    if (!t1n || !t1ce || !t2) {
      setError("Please upload all three MRI scan types before continuing.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setOutputs({ t2fUrl: null, segUrl: null });

    const formData = new FormData();
    formData.append("t1n", t1n);
    formData.append("t1ce", t1ce);
    formData.append("t2", t2);

    try {
      const res = await fetch("http://localhost:5000/api/segment/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Segmentation failed");
      const data = await res.json();
      setOutputs({
        t2fUrl: data.t2f,
        segUrl: data.seg,
      });

      const historyEntry = {
        mode: 'segmentation',
        modelName: 'Multi-Input U-Net',
        outputImage: data.seg,
        metrics: {},
        timestamp: Date.now()
      };

      const currentHistory = JSON.parse(localStorage.getItem('conversionHistory') || '[]');
      localStorage.setItem('conversionHistory', JSON.stringify([historyEntry, ...currentHistory]));

    } catch (err) {
      setError("Segmentation failed. Please check the files or server status.");
      console.error("Segmentation error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderFileInput = (name, label, description) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>
      <div className="flex items-center">
        <label className="flex-1 flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 cursor-pointer">
          <Upload size={16} className="mr-2" />
          Select File
          <input 
            type="file" 
            name={name} 
            className="hidden" 
            accept=".nii,.nii.gz" 
            onChange={handleFileChange} 
          />
        </label>
        {inputs[name] && (
          <span className="ml-3 text-sm text-gray-500 truncate max-w-xs">
            {inputs[name].name}
          </span>
        )}
      </div>
      <p className="mt-1 text-xs text-gray-500">
        {description}
      </p>
    </div>
  );

  const renderImageCard = (title, imageUrl, altText) => (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
      </div>
      <div className="p-4 h-64 flex items-center justify-center bg-gray-50">
        {imageUrl ? (
          <img 
            src={imageUrl} 
            alt={altText} 
            className="max-w-full max-h-full object-contain" 
          />
        ) : (
          <div className="text-center text-gray-400">
            <p>{title === "Input Preview" ? "Select files to preview" : "Output will appear here"}</p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-xl shadow-sm overflow-hidden max-w-6xl mx-auto">
      {/* Header */}
      <div className="bg-blue-600 px-6 py-4">
        <h2 className="text-xl font-semibold text-white">Brain MRI Segmentation</h2>
        <p className="text-blue-100 text-sm mt-1">
          Upload three MRI scan types to generate T2-FLAIR and tissue segmentation
        </p>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {/* File Upload Section */}
        <div className="mb-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Scans</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {renderFileInput(
              "t1n",
              "T1-weighted (Native)",
              "Native anatomical MRI without contrast"
            )}
            {renderFileInput(
              "t1ce",
              "T1-weighted (Contrast)",
              "With gadolinium contrast for tumors"
            )}
            {renderFileInput(
              "t2",
              "T2-weighted Scan",
              "Shows fluid and inflammation"
            )}
          </div>
        </div>

        {/* Action Button */}
        <div className="flex justify-center mb-8">
          <button
            className={`px-6 py-3 rounded-md shadow-sm text-white font-medium flex items-center ${
              isLoading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            onClick={handleSubmit}
            disabled={isLoading}
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

        {/* Results Section */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {renderImageCard(
              "Input Preview (T1n)", 
              previews.t1n, 
              "T1-weighted native scan"
            )}
            {renderImageCard(
              "Synthesized T2-FLAIR", 
              outputs.t2fUrl, 
              "Generated T2-FLAIR image"
            )}
            {renderImageCard(
              "Segmentation Map", 
              outputs.segUrl, 
              "Brain tissue segmentation"
            )}
          </div>
        </div>
      </div>

      {/* Info Section */}
      <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
        <h3 className="text-sm font-medium text-gray-700 mb-2">About This Tool</h3>
        <p className="text-xs text-gray-500">
          This segmentation model uses a multi-input U-Net architecture trained on the BraTS dataset. 
          It can identify tumor regions (enhancing tumor, edema, necrosis) and predict T2-FLAIR images 
          from other MRI modalities. Results are for research purposes only.
        </p>
      </div>
    </div>
  );
};

export default Segmentation;