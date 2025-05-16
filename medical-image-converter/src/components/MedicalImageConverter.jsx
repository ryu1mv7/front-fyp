import React, { useState } from 'react';
import { Upload, ArrowRight, RefreshCw, Folder } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

const MedicalImageConverter = () => {
  // Auth and navigation
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);

  // Conversion state
  const [conversionType, setConversionType] = useState('t1-to-t2');
  const [inputFormat, setInputFormat] = useState('image');
  const [inputImage, setInputImage] = useState(null);
  const [inputBatch, setInputBatch] = useState([]);
  const [niiFiles, setNiiFiles] = useState([]);
  const [modalities, setModalities] = useState({});
  const [outputImage, setOutputImage] = useState({ t2f: null, seg: null });
  const [outputBatch, setOutputBatch] = useState([]);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('metrics');
  const [brainSegOutput, setBrainSegOutput] = useState(null);
  const [batchProgress, setBatchProgress] = useState(0);
  const [processingMode, setProcessingMode] = useState('single');

  // Model data
  const modalityList = ['T1N', 'T1C', 'T2W', 'T2F', 'SEG'];
  
  const modelMetricsMap = {
    't1-to-t2': {
      ssim: 0.9152,
      psnr: 32.65,
      mse: 0.0045,
      lpips: 0.12,
      processingTime: "1.1 seconds",
      modelType: "U-Net",
      epochs: "100"
    },
    'pd-to-t2': {
      ssim: 0.9084,
      psnr: 31.90,
      mse: 0.0052,
      lpips: 0.15,
      processingTime: "1.0 seconds",
      modelType: "U-Net",
      epochs: "85"
    },
    'brats-t2f-seg': {
      ssim: 0.9663,
      psnr: 37.14,
      mse: 0.0031,
      lpips: 0.08,
      processingTime: "1.2 seconds",
      modelType: "cGAN",
      epochs: "50"
    },
    'mri-to-ct': {
      ssim: 0.9377,
      psnr: 34.35,
      mse: 0.0042,
      lpips: 0.10,
      processingTime: "1.3 seconds",
      modelType: "Pix2Pix",
      epochs: "60"
    },
    'ixi-brain-seg': {
      ssim: 0,
      psnr: 0,
      mse: 0,
      lpips: 0,
      processingTime: 'N/A',
      modelType: 'U-Net',
      epochs: 'N/A'
    }
  };

  const metricsData = modelMetricsMap[conversionType];

  const conversionOptions = [
    { value: 't1-to-t2', label: 'T1 → T2: Single-Modality MRI Synthesis' },
    { value: 'pd-to-t2', label: 'PD → T2: Proton Density to T2 Conversion' },
    { value: 'brats-t2f-seg', label: 'T1N/T1C/T2W → T2-FLAIR/Seg: Multi-Modal Brain Synthesis' },
    { value: 'mri-to-ct', label: 'MRI → CT: Cross-Modality Translation (Pix2Pix)' },
    { value: 'ixi-brain-seg', label: 'T1 → Segmentation: IXI Brain Tissue Map' }
  ];

  const modelDescriptions = {
    't1-to-t2': "Generates a synthetic T2-weighted MRI from a T1-normal input using a U-Net-based GAN.",
    'pd-to-t2': "Synthesizes a T2-weighted MRI from Proton Density input using a modified U-Net architecture.",
    'brats-t2f-seg': "Produces a synthetic T2-FLAIR and a brain anomaly segmentation map from T1N, T1C, and T2W inputs.",
    'mri-to-ct': "Translates MRI scans into synthetic CT using a Pix2Pix CGAN.",
    'ixi-brain-seg': "Segments brain tissue into CSF, Gray Matter, and White Matter using the IXI dataset model.",
  };

  // File handling
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setInputImage(file);
    setOutputImage({ t2f: null, seg: null });
    setError(null);
    
    const fileReader = new FileReader();
    fileReader.onload = () => setPreviewUrl(fileReader.result);
    fileReader.readAsDataURL(file);
  };

  const handleFolderUpload = (event) => {
    const files = Array.from(event.target.files);
    const imageFiles = files.filter(file => 
      file.type.startsWith('image/') || 
      file.name.endsWith('.jpg') || 
      file.name.endsWith('.jpeg') || 
      file.name.endsWith('.png')
    );

    if (imageFiles.length === 0) {
      setError('No valid image files found in the selected folder');
      return;
    }

    setInputBatch(imageFiles);
    setOutputBatch([]);
    setError(null);

    if (imageFiles.length > 0) {
      const fileReader = new FileReader();
      fileReader.onload = () => setPreviewUrl(fileReader.result);
      fileReader.readAsDataURL(imageFiles[0]);
    }
  };

  const handleNiftiUpload = (event, modality) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setModalities(prev => ({ ...prev, [modality]: file }));
    setOutputImage({ t2f: null, seg: null });
    setError(null);
  };

  const handleConversion = async () => {
    if (processingMode === 'single') {
      await handleSingleConversion();
    } else {
      await handleBatchConversion();
    }
  };

  const handleSingleConversion = async () => {
    if ((inputFormat === 'image' && !inputImage) || 
        (inputFormat === 'nii' && Object.keys(modalities).length === 0)) {
      setError('Please upload valid input files');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      const endpoint = inputFormat === 'nii' ? '/api/convert_nii/' : '/api/convert/';

      if (inputFormat === 'nii') {
        const niiFilesToUpload = conversionType === 'ixi-brain-seg'
          ? [modalities['t1']]
          : Object.values(modalities);

        niiFilesToUpload.forEach(file => file && formData.append('image', file));
      } else {
        formData.append('image', inputImage);
      }

      formData.append('conversionType', conversionType);

      const response = await fetch(`http://localhost:5000${endpoint}`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Conversion failed');

      setOutputImage({
        t2f: data.result.t2f || data.result,
        seg: data.result.seg || null
      });

    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchConversion = async () => {
    if (!inputBatch.length) {
      setError('Please select a folder with valid images');
      return;
    }

    setIsLoading(true);
    setError(null);
    setOutputBatch([]);
    setBatchProgress(0);

    try {
      const results = [];
      const endpoint = '/api/convert/';

      for (let i = 0; i < inputBatch.length; i++) {
        const formData = new FormData();
        formData.append('image', inputBatch[i]);
        formData.append('conversionType', conversionType);

        const response = await fetch(`http://localhost:5000${endpoint}`, {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(`Failed to convert ${inputBatch[i].name}: ${data.error}`);

        results.push({
          original: inputBatch[i].name,
          t2f: data.result.t2f || data.result,
          seg: data.result.seg || null
        });

        setBatchProgress(((i + 1) / inputBatch.length) * 100);
      }

      setOutputBatch(results);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsLoading(false);
      setBatchProgress(0);
    }
  };

  const handleBrainSegmentation = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('image', file);
      
      const res = await fetch('http://localhost:5000/api/brain_seg/', {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Segmentation failed');
      
      setBrainSegOutput(data.result);
    } catch (err) {
      setError(err.message);
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
        
        {/* User Dropdown */}
        <div className="relative">
          <div 
            className="flex items-center cursor-pointer" 
            onMouseEnter={() => setShowDropdown(true)}
          >
            <span className="mr-2 text-gray-400">{currentUser?.email}</span>
            <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
              <span className="text-lg font-semibold">
                {currentUser?.email?.[0].toUpperCase() || 'U'}
              </span>
            </div>
          </div>
          
          {showDropdown && (
            <div 
              className="absolute right-0 mt-2 w-64 bg-gray-800 rounded-md shadow-lg z-10"
              onMouseLeave={() => setShowDropdown(false)}
            >
              <div className="p-4 border-b border-gray-700">
                <div className="w-16 h-16 mb-2 bg-gray-700 rounded-full flex items-center justify-center mx-auto">
                  <span className="text-2xl font-bold">
                    {currentUser?.email?.[0].toUpperCase() || 'U'}
                  </span>
                </div>
                <span className="block text-center font-medium">{currentUser?.email}</span>
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

      {/* Main Content */}
      <div className="w-full max-w-4xl bg-gray-800 rounded-lg p-6 shadow-lg">
        <h2 className="text-xl font-bold mb-6">Multi-modal Medical Image Synthesis and Translation</h2>
        
        {/* Conversion Controls */}
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
            <p className="text-sm text-gray-400 italic mt-1">
              {modelDescriptions[conversionType]}
            </p>
          </div>
          
          <div className="flex-1">
            <label className="block text-sm font-medium mb-2">Processing Mode</label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600"
              value={processingMode}
              onChange={(e) => setProcessingMode(e.target.value)}
            >
              <option value="single">Single Image</option>
              <option value="batch">Batch Processing (Folder)</option>
            </select>
          </div>

          <div className="flex-1">
            <label className="block text-sm font-medium mb-2">Input Format</label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600"
              value={inputFormat}
              onChange={(e) => setInputFormat(e.target.value)}
            >
              <option value="image">JPG / PNG</option>
              <option value="nii">NIfTI (.nii, .nii.gz)</option>
            </select>
          </div>
        </div>

        {/* File Upload Section */}
        {inputFormat === 'image' && processingMode === 'single' && (
          <input 
            type="file" 
            accept=".jpg,.jpeg,.png" 
            onChange={handleFileUpload} 
            className="mb-4" 
          />
        )}

        {inputFormat === 'image' && processingMode === 'batch' && (
          <div className="mb-4">
            <input
              type="file"
              webkitdirectory="true"
              directory="true"
              multiple
              onChange={handleFolderUpload}
              className="mb-2"
            />
            {inputBatch.length > 0 && (
              <p className="text-sm text-gray-400">
                {inputBatch.length} images selected from folder
              </p>
            )}
          </div>
        )}

        {inputFormat === 'nii' && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            {conversionType === 'ixi-brain-seg' ? (
              <div className="bg-gray-800 p-4 rounded text-center">
                <p className="mb-2 font-semibold">T1</p>
                <input
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={(e) => {
                    const file = e.target.files[0];
                    if (file) {
                      setModalities({ t1: file });  // override all others
                      setOutputImage({ t2f: null, seg: null });
                    }
                  }}
                  className="text-sm text-gray-300"
                />
                <p className="text-sm text-gray-400 mt-2">
                  {modalities['t1']?.name || 'Not assigned'}
                </p>
              </div>
            ) : (
              modalityList.map(modality => (
                <div key={modality} className="bg-gray-800 p-4 rounded text-center">
                  <p className="mb-2 font-semibold">{modality}</p>
                  <input
                    type="file"
                    accept=".nii,.nii.gz"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        setModalities(prev => ({ ...prev, [modality]: file }));
                        setOutputImage({ t2f: null, seg: null });
                      }
                    }}
                    className="text-sm text-gray-300"
                  />
                  <p className="text-sm text-gray-400 mt-2">
                    {modalities[modality]?.name || 'Not assigned'}
                  </p>
                </div>
              ))
            )}
          </div>
        )}

        <button 
          onClick={handleConversion} 
          disabled={isLoading} 
          className="bg-green-600 px-4 py-2 rounded"
        >
          {isLoading ? 'Converting...' : 'Convert'}
        </button>

        {error && (
          <div className="mt-4 p-2 bg-red-500 bg-opacity-20 border border-red-500 text-red-300 rounded">
            {error}
          </div>
        )}

        {/* Batch Progress Bar */}
        {processingMode === 'batch' && isLoading && (
          <div className="mt-4">
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div 
                className="bg-green-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${batchProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-400 mt-1">
              Processing: {Math.round(batchProgress)}%
            </p>
          </div>
        )}

        {/* Output Preview Section */}
        {processingMode === 'single' ? (
          <div className="grid grid-cols-3 gap-4 mt-6 text-center">
            <div>
              <div className="bg-gray-700 h-64 flex items-center justify-center border border-gray-600">
                {inputFormat === 'image' && previewUrl ? <img src={previewUrl} alt="Input Preview" className="max-h-full" /> : <p>No preview available</p>}
              </div>
              <p className="text-sm mt-2 text-gray-400">Input Image</p>
            </div>
            <div>
              <div className="bg-gray-700 h-64 flex items-center justify-center border border-gray-600">
                {outputImage.t2f ? <img src={outputImage.t2f} alt="Predicted Output" className="max-h-full" /> : <p>Predicted (T2F) output</p>}
              </div>
              <p className="text-sm mt-2 text-gray-400">Synthesized T2-FLAIR</p>
            </div>
            <div>
              <div className="bg-gray-700 h-64 flex items-center justify-center border border-gray-600">
                {outputImage.seg ? <img src={outputImage.seg} alt="Segmentation Output" className="max-h-full" /> : <p>Segmentation output</p>}
              </div>
              <p className="text-sm mt-2 text-gray-400">Segmentation Map</p>
            </div>
          </div>
        ) : (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-4">Batch Processing Results</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {outputBatch.map((result, index) => (
                <div key={index} className="bg-gray-700 p-2 rounded">
                  <img 
                    src={result.t2f} 
                    alt={`Output ${index + 1}`} 
                    className="w-full h-32 object-cover rounded"
                  />
                  <p className="text-sm text-gray-400 mt-2 truncate">
                    {result.original}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        <button className={`px-4 py-2 font-medium ${activeTab === 'brain-seg' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`} onClick={() => setActiveTab('brain-seg')}>Brain Tissue Segmentation</button>

        {/* Tabbed Section */}
        <div className="mt-6">
          {/* Tab Headers */}
          <div className="flex border-b border-gray-700 overflow-x-auto">
            <button
              className={`px-4 py-2 font-medium whitespace-nowrap ${activeTab === 'metrics' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('metrics')}
            >
              Metrics
            </button>
            <button
              className={`px-4 py-2 font-medium whitespace-nowrap ${activeTab === 'histogram' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('histogram')}
            >
              Visualizations
            </button>
            <button
              className={`px-4 py-2 font-medium whitespace-nowrap ${activeTab === 'details' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setActiveTab('details')}
            >
              Details
            </button>
          </div>
          
          {/* Tab Content */}
          <div className="p-4 bg-gray-700 rounded-b">
            {activeTab === 'metrics' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">SSIM</h3>
                  <p className="text-center text-2xl font-bold">
                    {metricsData.ssim.toFixed(4)}
                  </p>
                  <p className="text-center text-xs text-gray-500">Structural Similarity</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">PSNR</h3>
                  <p className="text-center text-2xl font-bold">
                    {metricsData.psnr.toFixed(2)} dB
                  </p>
                  <p className="text-center text-xs text-gray-500">Peak Signal-to-Noise Ratio</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">LPIPS</h3>
                  <p className="text-center text-2xl font-bold">
                    {metricsData.lpips.toFixed(4)}
                  </p>
                  <p className="text-center text-xs text-gray-500">Perceptual Similarity</p>
                </div>

                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">MSE</h3>
                  <p className="text-center text-2xl font-bold">
                    {metricsData.mse.toFixed(4)}
                  </p>
                  <p className="text-center text-xs text-gray-500">Mean Squared Error</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Model</h3>
                  <p className="text-center text-xl font-bold">{metricsData.modelType}</p>
                  <p className="text-center text-xs text-gray-500">Architecture</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Processing Time</h3>
                  <p className="text-center text-xl font-bold">{metricsData.processingTime}</p>
                  <p className="text-center text-xs text-gray-500">Conversion Duration</p>
                </div>
              </div>
            )}
            
            {activeTab === 'brain-seg' && (
              <div className="text-gray-300 space-y-4">
                <h3 className="text-xl font-semibold text-green-400">IXI Brain Tissue Segmentation</h3>
                <input
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    const formData = new FormData();
                    formData.append('image', file);
                    try {
                      const res = await fetch('http://localhost:5000/api/brain_seg/', {
                        method: 'POST',
                        body: formData
                      });
                      const data = await res.json();
                      if (!res.ok) throw new Error(data.error || 'Segmentation failed');
                      setBrainSegOutput(data.result);
                    } catch (err) {
                      setBrainSegOutput(null);
                      alert('Error: ' + err.message);
                    }
                  }}
                  className="text-sm"
                />
                {brainSegOutput && (
                  <div className="mt-4 text-center">
                    <img src={brainSegOutput} alt="Tissue Segmentation Output" className="mx-auto border border-gray-500 rounded" />
                    <p className="text-sm text-gray-400 mt-2">CSF / Gray Matter / White Matter</p>
                  </div>
                )}
              </div>
            )}
            {activeTab === 'histogram' && (
            <div className="text-gray-300 space-y-8">
              
              {/* Performance Chart */}
              <div>
                <h3 className="text-lg font-semibold text-yellow-300 mb-2">Performance Comparison</h3>
                <img src="/assets/Model_Comparison_Chart.png" alt="Model Performance Comparison" className="w-full rounded border border-gray-600" />
                <p className="text-sm text-gray-400 mt-1">
                  Scaled SSIM (×100) and PSNR scores for all tested models. Multi-Input U-Net shows superior performance and stability across both metrics, while other models like CGAN were discarded due to training instability or dataset mismatch.
                </p>
              </div>

              {/* Model Architecture */}
              <div>
                <h3 className="text-lg font-semibold text-blue-300 mb-2">Model Architecture Diagram</h3>
                <img src="/assets/Model Architecture Diagram.png" alt="Architecture Overview" className="w-full rounded border border-gray-600" />
                <p className="text-sm text-gray-400 mt-1">
                  The architecture features a multi-modal U-Net generator taking in T1N, T1C, and T2W, followed by a segmentation head to jointly output T2-FLAIR and tumor map. This dual-path strategy enforces structural alignment.
                </p>
              </div>

              {/* Workflow */}
              <div>
                <h3 className="text-lg font-semibold text-green-300 mb-2">Workflow Pipeline</h3>
                <img src="/assets/Workflow Diagram.png" alt="Workflow Pipeline" className="w-full rounded border border-gray-600" />
                <p className="text-sm text-gray-400 mt-1">
                  Visual summary of the full conversion process from NIfTI input → preprocessing → model inference → segmentation and final visualization.
                </p>
              </div>
            </div>
          )}
            {activeTab === 'details' && (
              <div className="text-gray-300 space-y-2">
                <p><span className="font-medium">Input Format:</span> {inputFormat}</p>
                <p><span className="font-medium">Output Format:</span> JPEG</p>
                <p><span className="font-medium">Model Version:</span> v1.2.0</p>
                <p><span className="font-medium">Conversion Type:</span> {conversionOptions.find(opt => opt.value === conversionType)?.label || conversionType}</p>
              </div>
            )}
          {activeTab === 'settings' && (
            <div className="text-gray-300 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Some Parameters?</label>
                <input type="range" min="1" max="100" defaultValue="90" className="w-full" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Inference Options</label>
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
      </div>
    </div>
  );
};

export default MedicalImageConverter;