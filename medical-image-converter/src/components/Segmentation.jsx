import React, { useState } from 'react';

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

  const renderImageBox = (label, src, alt) => (
    <div className="text-center bg-gray-900 p-2 rounded border border-gray-600 min-h-[280px] flex flex-col justify-between">
      <p className="text-sm font-semibold text-gray-300 mb-2">{label}</p>
      {src ? (
        <img src={src} alt={alt} className="w-full rounded border" />
      ) : (
        <div className="flex items-center justify-center h-48 bg-gray-800 text-gray-500 border border-dashed rounded">
          Preview not available
        </div>
      )}
    </div>
  );

  return (
    <div className="p-6 bg-gray-800 rounded-lg text-white space-y-6 max-w-5xl mx-auto">
      <h2 className="text-xl font-bold text-green-400">MRI Scan Segmentation</h2>
      <p className="text-sm text-gray-300">
        This tool allows you to upload three types of MRI brain scans (T1n, T1ce, T2) to generate a predicted <strong>T2-FLAIR</strong> image and <strong>brain tissue segmentation</strong>. Useful for assisting diagnosis or cross-referencing regions of interest.
      </p>

      {/* Uploads */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Upload T1n (T1-weighted - native)
          </label>
          <input type="file" name="t1n" accept=".nii,.nii.gz" onChange={handleFileChange} />
          <p className="text-xs text-gray-400 mt-1">Native anatomical MRI without contrast (required for structure).</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Upload T1ce (T1-weighted with contrast)
          </label>
          <input type="file" name="t1ce" accept=".nii,.nii.gz" onChange={handleFileChange} />
          <p className="text-xs text-gray-400 mt-1">Highlights tumors and blood vessels with gadolinium contrast.</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Upload T2 (T2-weighted scan)
          </label>
          <input type="file" name="t2" accept=".nii,.nii.gz" onChange={handleFileChange} />
          <p className="text-xs text-gray-400 mt-1">Shows fluid and inflammation—important for edema or lesions.</p>
        </div>
      </div>

      {/* Submit */}
      <div>
        <button
          className="px-5 py-2 bg-green-600 hover:bg-green-700 font-semibold rounded shadow-sm"
          onClick={handleSubmit}
          disabled={isLoading}
        >
          {isLoading ? "Processing images..." : "▶ Run Segmentation"}
        </button>
        {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
      </div>

      {/* Output Section */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-cyan-300 mb-3">Output Visuals</h3>
        <p className="text-xs text-gray-400 mb-4">
          View your original scan (T1n), the AI-generated synthetic T2-FLAIR image, and the resulting brain tissue segmentation map.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {renderImageBox("Input Preview (T1n)", previews.t1n, "T1n")}
          {renderImageBox("Synthesized T2-FLAIR (T2f)", outputs.t2fUrl, "T2f Output")}
          {renderImageBox("Predicted Segmentation Map", outputs.segUrl, "Segmentation Output")}
        </div>
      </div>
    </div>
  );
};

export default Segmentation;
