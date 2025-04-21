import React, { useState } from 'react';
import { Upload, ArrowRight, RefreshCw } from 'lucide-react';

const MedicalImageConverter = () => {
  const [conversionType, setConversionType] = useState('mri-to-ct');
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('metrics');

  // Dummy metrics data - would come from backend in real implementation
  const metricsData = {
    ssim: outputImage ? 0.9663 : null,
    psnr: outputImage ? 37.14 : null,
    mse: outputImage ? 0.0031 : null,
    processingTime: outputImage ? "1.2 seconds" : null,
    modelType: "cGAN",
    nothing: 0
  };
  
  const conversionOptions = [
    { value: 'mri-to-ct', label: 'MRI → CT' },
    { value: 'ct-to-mri', label: 'CT → MRI' }
  ];
  
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setInputImage(file);
    setOutputImage(null);
    setError(null);
    
    // Create preview URL
    const fileReader = new FileReader();
    fileReader.onload = () => {
      setPreviewUrl(fileReader.result);
    };
    fileReader.readAsDataURL(file);
  };
  
  const handleDragOver = (event) => {
    event.preventDefault();
  };
  
  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      setInputImage(file);
      setOutputImage(null);
      setError(null);
      
      // Create preview URL
      const fileReader = new FileReader();
      fileReader.onload = () => {
        setPreviewUrl(fileReader.result);
      };
      fileReader.readAsDataURL(file);
    }
  };
  
  const handleConversion = async () => {
    if (!inputImage) {
      setError('Please upload an image first');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('image', inputImage);
      formData.append('conversionType', conversionType);
      
      // Send to backend API
      const response = await fetch('http://localhost:5000/api/convert', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Conversion process failed');
      }
      
      const data = await response.json();
      setOutputImage(data.result);
      
    } catch (err) {
      console.error('Error during conversion:', err);
      setError(err.message || 'Conversion process failed');
    } finally {
      setIsLoading(false);
    }
  };

  // For demo purposes, we simulate conversion without real API call
  const handleDemoConversion = () => {
    if (!inputImage) {
      setError('Please upload an image first');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Simulate API delay
    setTimeout(() => {
      setOutputImage(previewUrl); // For demo, just use the same image
      setIsLoading(false);
    }, 2000);
  };
  
  return (
    <div className="flex flex-col items-center p-6 bg-gray-900 text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-6">Multi-modal Medical Image Synthesis and Translation</h1>
      
      <div className="w-full max-w-4xl bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-2">Conversion Type</label>
            <select 
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:ring-2 focus:ring-blue-500"
              value={conversionType}
              onChange={(e) => setConversionType(e.target.value)}
            >
              {conversionOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          
          <div className="flex gap-2">
            <label className="flex items-center justify-center px-4 py-2 bg-blue-600 rounded cursor-pointer hover:bg-blue-700 transition">
              <Upload size={18} className="mr-2" />
              Upload
              <input 
                type="file" 
                className="hidden" 
                accept=".jpg,.jpeg,.png" 
                onChange={handleFileUpload} 
              />
            </label>
            
            <button 
              className="flex items-center justify-center px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleDemoConversion}
              disabled={!inputImage || isLoading}
            >
              {isLoading ? <RefreshCw size={18} className="mr-2 animate-spin" /> : <ArrowRight size={18} className="mr-2" />}
              Convert
            </button>
          </div>
        </div>
        
        {error && (
          <div className="mb-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 rounded text-red-300">
            {error}
          </div>
        )}
        
        <div className="flex flex-col md:flex-row gap-6">
          <div 
            className="flex-1 h-64 border-2 border-dashed border-gray-600 rounded flex items-center justify-center bg-gray-700"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {previewUrl ? (
              <img 
                src={previewUrl} 
                alt="Input" 
                className="max-w-full max-h-full object-contain" 
              />
            ) : (
              <div className="text-center p-4">
                <Upload size={32} className="mx-auto mb-2 text-gray-500" />
                <p className="text-gray-400">Drag and drop input image here</p>
              </div>
            )}
          </div>
          
          <div className="flex items-center justify-center">
            <ArrowRight size={24} className="text-blue-500" />
          </div>
          
          <div className="flex-1 h-64 border-2 border-gray-600 rounded flex items-center justify-center bg-gray-700">
            {isLoading ? (
              <div className="text-center">
                <RefreshCw size={32} className="mx-auto mb-2 animate-spin text-blue-500" />
                <p>Processing...</p>
              </div>
            ) : outputImage ? (
              <img 
                src={outputImage} 
                alt="Output" 
                className="max-w-full max-h-full object-contain" 
              />
            ) : (
              <div className="text-center p-4 text-gray-400">
                <p>Converted image will appear here</p>
              </div>
            )}
          </div>
        </div>
        
        {/* Tabbed Section */}
        <div className="mt-6">
          {/* Tab Headers */}
          <div className="flex border-b border-gray-700">
            <button
              className={`px-4 py-2 font-medium ${activeTab === 'metrics' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('metrics')}
            >
              Metrics
            </button>
            <button
              className={`px-4 py-2 font-medium ${activeTab === 'histogram' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('histogram')}
            >
              Visualization or Diagram?
            </button>
            <button
              className={`px-4 py-2 font-medium ${activeTab === 'details' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('details')}
            >
              Details
            </button>
            <button
              className={`px-4 py-2 font-medium ${activeTab === 'settings' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('settings')}
            >
              Settings
            </button>
          </div>
          
          {/* Tab Content */}
          <div className="p-4 bg-gray-700 rounded-b">
            {activeTab === 'metrics' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">SSIM</h3>
                  <p className="text-center text-2xl font-bold">{metricsData.ssim !== null ? metricsData.ssim.toFixed(4) : "0.88"}</p>
                  <p className="text-center text-xs text-gray-500">Structural Similarity</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">PSNR</h3>
                  <p className="text-center text-2xl font-bold">{metricsData.psnr !== null ? metricsData.psnr.toFixed(2) + " dB" : "28db"}</p>
                  <p className="text-center text-xs text-gray-500">Peak Signal-to-Noise Ratio</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">MSE</h3>
                  <p className="text-center text-2xl font-bold">{metricsData.mse !== null ? metricsData.mse.toFixed(4) : "0.005"}</p>
                  <p className="text-center text-xs text-gray-500">Mean Squared Error</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Model</h3>
                  <p className="text-center text-xl font-bold">{metricsData.modelType}</p>
                  <p className="text-center text-xs text-gray-500">Architecture</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">TBD</h3>
                  <p className="text-center text-xl font-bold">{metricsData.epochs}</p>
                  <p className="text-center text-xs text-gray-500">TBD</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Processing Time</h3>
                  <p className="text-center text-xl font-bold">{metricsData.processingTime || "2.36s"}</p>
                  <p className="text-center text-xs text-gray-500">Conversion Duration</p>
                </div>
              </div>
            )}
            
            {activeTab === 'histogram' && (
              <div className="flex justify-center items-center h-48 text-gray-400">
                <p>Some Diagram?.</p>
              </div>
            )}
            
            {activeTab === 'details' && (
              <div className="text-gray-300 space-y-2">
                <p><span className="font-medium">Input Format:</span> {inputImage ? inputImage.type : "-"}</p>
                <p><span className="font-medium">Output Format:</span> JPEG</p>
                <p><span className="font-medium">Model Version:</span> v1.2.0</p>
                <p><span className="font-medium">Conversion Type:</span> {conversionOptions.find(opt => opt.value === conversionType)?.label || conversionType}</p>
              </div>
            )}
            
            {activeTab === 'settings' && (
              <div className="text-gray-300 space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">some parametors?</label>
                  <input type="range" min="1" max="100" defaultValue="90" className="w-full" />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Inference Parameters</label>
                  <div className="flex gap-4">
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" defaultChecked /> 
                      Higher Accuracy
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" /> 
                      Post-processing
                    </label>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="mt-4 text-sm text-gray-400">
          <p>* This demo displays the uploaded image as a result for presentation purposes.</p>
          <p>* In the actual application, conversion would be performed via a backend API.</p>
        </div>
      </div>
    </div>
  );
};

export default MedicalImageConverter;