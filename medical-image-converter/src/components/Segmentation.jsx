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
      setError("Please upload all three modalities.");
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
        outputImage: data.seg,  // or data.t2f if you want both
        metrics: {},            // optional: if you calculated SSIM, PSNR etc.
        timestamp: Date.now()
        };

        const currentHistory = JSON.parse(localStorage.getItem('conversionHistory') || '[]');
        localStorage.setItem('conversionHistory', JSON.stringify([historyEntry, ...currentHistory]));

    } catch (err) {
      setError("Segmentation failed.");
      console.error("Segmentation error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderImageBox = (label, src, alt) => (
    <div className="text-center bg-gray-900 p-2 rounded border border-gray-600 min-h-[280px] flex flex-col justify-between">
      <p className="text-sm mb-2">{label}</p>
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
    <div className="p-4 bg-gray-800 rounded text-white space-y-4">
      {/* Uploads */}
      <div className="space-y-2">
        <label>
          <span className="block text-sm mb-1">Upload T1n (.nii or .nii.gz)</span>
          <input type="file" name="t1n" accept=".nii,.nii.gz" onChange={handleFileChange} />
        </label>
        <label>
          <span className="block text-sm mb-1">Upload T1ce (.nii or .nii.gz)</span>
          <input type="file" name="t1ce" accept=".nii,.nii.gz" onChange={handleFileChange} />
        </label>
        <label>
          <span className="block text-sm mb-1">Upload T2 (.nii or .nii.gz)</span>
          <input type="file" name="t2" accept=".nii,.nii.gz" onChange={handleFileChange} />
        </label>
      </div>

      {/* Submit */}
      <button
        className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
        onClick={handleSubmit}
        disabled={isLoading}
      >
        {isLoading ? "Processing..." : "Run Segmentation"}
      </button>

      {/* Error */}
      {error && <p className="text-red-400">{error}</p>}

      {/* Output Previews */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
        {renderImageBox("Input Preview (T1n)", previews.t1n, "T1n")}
        {renderImageBox("Predicted T2f", outputs.t2fUrl, "T2f Output")}
        {renderImageBox("Predicted Segmentation", outputs.segUrl, "Segmentation Output")}
      </div>
    </div>
  );
};

export default Segmentation;
