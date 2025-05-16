import React, { useState } from 'react';
import { Upload, ArrowRight, RefreshCw } from 'lucide-react';
//for firebase
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

const MedicalImageConverter = () => {
  const [conversionType, setConversionType] = useState('t1-to-t2');
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('metrics');
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);
  
  const [metrics, setMetrics] = useState(null);

  const [imageFormat, setImageFormat] = useState('png');

  const formatOptions = [
    { value: 'png', label: 'PNG / JPG' },
    { value: 'nii', label: '.nii' }
  ];

  
  const conversionOptions = [
    { value: 't1-to-t2',   label: 'T1 → T2'   },
    { value: 'pd-to-t2',   label: 'PD → T2'   },
    { value: 't2-to-t1',  label: 'T2 → T1'    }
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
      const formData = new FormData();
      formData.append('image', inputImage);
      formData.append('conversionType', conversionType);
      formData.append('imageFormat', imageFormat);

      const response = await fetch('http://localhost:5000/api/convert/', {
        method: 'POST',
        body: formData
      });
  
      // grab text so we can debug
      const text = await response.text();
      console.log('raw response:', response.status, text);
  
      if (!response.ok) {
        // try parsing JSON error, otherwise show raw text
        let msg;
        try {
          msg = JSON.parse(text).error;
        } catch {
          msg = text;
        }
        throw new Error(msg || 'Conversion process failed');
      }
  
      const data = JSON.parse(text);
      setOutputImage(data.result);

      setMetrics({
        ssim:  data.metrics.ssim,
        psnr:  data.metrics.psnr,
        lpips: data.metrics.lpips,
      });

  
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      setError('Failed to log out');
    }
  };
  
  return (
    <div className="flex flex-col items-center p-6 bg-gray-900 text-white min-h-screen">
      {/* Navigation Bar */}
      <div className="w-full bg-gray-800 p-4 flex justify-between items-center mb-6">
        <h1 className="text-xl font-bold">Medical Image Converter</h1>
        
        {/* User Info and Dropdown */}
        <div className="relative">
          <div 
            className="flex items-center cursor-pointer" 
            onMouseEnter={() => setShowDropdown(true)}
          >
            <span className="mr-2 text-gray-400">{currentUser?.displayName || currentUser?.email}</span>
            <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
              <span className="text-lg font-semibold">
                {currentUser?.displayName 
                  ? currentUser.displayName[0].toUpperCase() 
                  : currentUser?.email 
                    ? currentUser.email[0].toUpperCase() 
                    : 'U'
                }
              </span>
            </div>
          </div>
          
          {/* Dropdown Menu */}
          {showDropdown && (
            <div 
              className="absolute right-0 mt-2 w-64 bg-gray-800 rounded-md shadow-lg z-10"
              onMouseEnter={() => setShowDropdown(true)}
              onMouseLeave={() => setShowDropdown(false)}
            >
              <div className="p-4 flex flex-col items-center border-b border-gray-700">
                <div className="w-16 h-16 mb-2 bg-gray-700 rounded-full flex items-center justify-center">
                  <span className="text-2xl font-bold">
                    {currentUser?.displayName 
                      ? currentUser.displayName[0].toUpperCase() 
                      : currentUser?.email 
                        ? currentUser.email[0].toUpperCase() 
                        : 'U'
                    }
                  </span>
                </div>
                <span className="font-medium">{currentUser?.displayName || "User"}</span>
                <span className="text-sm text-gray-400 text-center">{currentUser?.email}</span>
              </div>
              
              <div className="py-1">
                <button className="block w-full text-left px-4 py-2 hover:bg-gray-700">
                  Dashboard
                </button>
                <button className="block w-full text-left px-4 py-2 hover:bg-gray-700">
                  History
                </button>
                <button 
                  className="block w-full text-left px-4 py-2 hover:bg-gray-700"
                  onClick={() => navigate('/settings')}
                >
                  Settings
                </button>
              </div>
              
              <div className="border-t border-gray-700 py-1">
                <button 
                  className="block w-full text-left px-4 py-2 text-red-400 hover:bg-gray-700"
                  onClick={handleLogout}
                >
                  Logout
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="w-full max-w-4xl bg-gray-800 rounded-lg p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-6">Multi-modal Medical Image Synthesis and Translation</h2>
        
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

           <div className="flex-1">
            <label className="block text-sm font-medium mb-2">Image Format</label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:ring-2 focus:ring-blue-500"
              value={imageFormat}
              onChange={e => setImageFormat(e.target.value)}
            >
              {formatOptions.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
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
              onClick={handleConversion}
              disabled={!inputImage || isLoading}
            >
              {isLoading 
                ? <RefreshCw size={18} className="mr-2 animate-spin" /> 
                : <ArrowRight size={18} className="mr-2" />
              }
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
                  <p className="text-center text-2xl font-bold">
                    {metrics ? metrics.ssim.toFixed(4) : '–'}
                  </p>
                  <p className="text-center text-xs text-gray-500">Structural Similarity</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">PSNR</h3>
                  <p className="text-center text-2xl font-bold">
                    {metrics ? `${metrics.psnr.toFixed(2)} dB` : '–'}
                  </p>
                  <p className="text-center text-xs text-gray-500">Peak Signal-to-Noise Ratio</p>
                </div>

                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">MSE</h3>
                  {/* <p className="text-center text-2xl font-bold">{metricsData.mse !== null ? metricsData.mse.toFixed(4) : "0.005"}</p> */}
                  <p className="text-center text-xs text-gray-500">Mean Squared Error</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Model</h3>
                  {/* <p className="text-center text-xl font-bold">{metricsData.modelType}</p> */}
                  <p className="text-center text-xs text-gray-500">Architecture</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">LPIPS</h3>
                  <p className="text-center text-2xl font-bold">
                    {metrics?.lpips != null
                      ? metrics.lpips.toFixed(4)
                      : '–'
                    }
                  </p>
                  <p className="text-center text-xs text-gray-500">
                    Learned Perceptual Image Patch Similarity
                  </p>
                </div>

                                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Processing Time</h3>
                  {/* <p className="text-center text-xl font-bold">{metricsData.processingTime || "2.36s"}</p> */}
                  <p className="text-center text-xs text-gray-500">Conversion Duration</p>
                </div>
              </div>
            )}
    
          {activeTab === 'histogram' && (
            <div className="text-gray-300 space-y-8">
              {/* Performance Chart */}
              <div>
                <h3 className="text-lg font-semibold text-yellow-300 mb-2">
                  Performance Comparison
                </h3>
                <img
                  src="/assets/Model_Comparison_Chart.png"
                  alt="Model Performance Comparison"
                  className="w-full rounded border border-gray-600"
                />
                <p className="text-sm text-gray-400 mt-1">
                  Scaled SSIM (×100) and PSNR scores for all tested models. Multi-Input U-Net shows superior performance and stability across both metrics, while other models like cGAN were discarded due to training instability or dataset mismatch.
                </p>
              </div>

              {/* Model Architecture Diagram */}
              <div>
                <h3 className="text-lg font-semibold text-blue-300 mb-2">
                  Model Architecture Diagram
                </h3>
                <img
                  src="/assets/Model_Architecture_Diagram.png"
                  alt="Architecture Overview"
                  className="w-full rounded border border-gray-600"
                />
                <p className="text-sm text-gray-400 mt-1">
                  The architecture features a multi-modal U-Net generator taking in T1N, T1C and T2W, followed by a segmentation head to jointly output T2-FLAIR and tumour map. This dual-path strategy enforces structural alignment.
                </p>
              </div>

              {/* Workflow Pipeline */}
              <div>
                <h3 className="text-lg font-semibold text-green-300 mb-2">
                  Workflow Pipeline
                </h3>
                <img
                  src="/assets/Workflow_Diagram.png"
                  alt="Workflow Pipeline"
                  className="w-full rounded border border-gray-600"
                />
                <p className="text-sm text-gray-400 mt-1">
                  Visual summary of the full conversion process from NIfTI input → preprocessing → model inference → segmentation and final visualisation.
                </p>
              </div>
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