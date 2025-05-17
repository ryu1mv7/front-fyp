import React, { useState, useEffect, useRef, useCallback, useEffect, useRef } from 'react';
import { Upload, ArrowRight, RefreshCw, Folder, ChevronLeft, ChevronRight } from 'lucide-react';
//for firebase
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import Segmentation from './Segmentation';

const PixelHistogram = ({ imageUrl, label, color }) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const drawGrid = (ctx, margin, plotWidth, plotHeight) => {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;

    [
      { count: 10, start: margin.left, length: plotWidth, isVertical: true },
      { count: 5, start: margin.top, length: plotHeight, isVertical: false }
    ].forEach(({ count, start, length, isVertical }) => {
      for (let i = 0; i <= count; i++) {
        const pos = start + (i / count) * length;
        ctx.beginPath();
        ctx.moveTo(isVertical ? pos : margin.left, isVertical ? margin.top : pos);
        ctx.lineTo(
          isVertical ? pos : margin.left + plotWidth,
          isVertical ? margin.top + plotHeight : pos
        );
        ctx.stroke();
      }
    });
  };

  const calculateHistogram = (imageData) => {
    const intensityValues = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      const intensity = imageData.data[i];
      if (intensity > 0) intensityValues.push(intensity);
    }

    const min = Math.min(...intensityValues);
    const max = Math.max(...intensityValues);
    const range = max - min;
    const numBins = 100;
    const binSize = range / numBins;
    const histogram = new Array(numBins).fill(0);

    intensityValues.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / binSize), numBins - 1);
      histogram[binIndex]++;
    });

    const sortedCounts = [...histogram].sort((a, b) => b - a);
    const maxCount = Math.max(
      sortedCounts[Math.floor(sortedCounts.length * 0.02)],
      Math.max(...histogram) * 0.8
    );

    return { histogram, min, max, maxCount };
  };

  const drawHistogram = useCallback((ctx, data, margin, plotWidth, plotHeight) => {
    const { histogram } = data;
    const maxY = 1500;
    
    const points = histogram.map((value, i) => ({
      x: margin.left + (i / histogram.length) * plotWidth,
      y: margin.top + plotHeight - (Math.min(value, maxY) / maxY) * plotHeight
    }));

    // Draw filled area
    ctx.beginPath();
    ctx.moveTo(points[0].x, margin.top + plotHeight);
    points.forEach(point => ctx.lineTo(point.x, point.y));
    ctx.lineTo(points[points.length - 1].x, margin.top + plotHeight);
    ctx.fillStyle = `${color}40`;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    points.forEach((point, i) => {
      if (i === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [color]);

  const drawLabels = (ctx, data, margin, plotWidth, plotHeight) => {
    ctx.fillStyle = '#999';
    
    // Y-axis label
    ctx.save();
    ctx.translate(margin.left * 0.3, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.font = `${Math.max(10, ctx.canvas.width * 0.025)}px Arial`;
    ctx.fillText('Frequency', 0, 0);
    ctx.restore();

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillText('Intensity', margin.left + plotWidth / 2, margin.top + plotHeight + margin.bottom * 0.8);

    // Y-axis values
    ctx.textAlign = 'right';
    ctx.font = `${Math.max(8, ctx.canvas.width * 0.02)}px Arial`;
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (i / 5) * plotHeight;
      const value = Math.round(((5 - i) / 5) * 1500);
      ctx.fillText(value.toString(), margin.left - 5, y + 4);
    }

    // X-axis values
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const x = margin.left + (i / 5) * plotWidth;
      const value = Math.round((i / 5) * 255);
      ctx.fillText(value.toString(), x, margin.top + plotHeight + margin.bottom * 0.5);
    }
  };

  useEffect(() => {
    if (!imageUrl) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    
    const updateCanvasSize = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.width * 0.6;
    };

    updateCanvasSize();
    const resizeObserver = new ResizeObserver(updateCanvasSize);
    resizeObserver.observe(container);

    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = img.width;
      tempCanvas.height = img.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(img, 0, 0);
      
      const margin = {
        top: canvas.height * 0.15,
        right: canvas.width * 0.1,
        bottom: canvas.height * 0.15,
        left: canvas.width * 0.15
      };

      const plotWidth = canvas.width - margin.left - margin.right;
      const plotHeight = canvas.height - margin.top - margin.bottom;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const histogramData = calculateHistogram(tempCtx.getImageData(0, 0, img.width, img.height));
      
      drawGrid(ctx, margin, plotWidth, plotHeight);
      drawHistogram(ctx, histogramData, margin, plotWidth, plotHeight);
      drawLabels(ctx, histogramData, margin, plotWidth, plotHeight);
    };
    
    img.src = imageUrl;
    return () => resizeObserver.disconnect();
  }, [imageUrl, color]);

  return (
    <div className="flex flex-col items-center w-full">
      <h3 className="text-sm font-medium mb-2">{label}</h3>
      <div ref={containerRef} className="w-full bg-gray-800 rounded border border-gray-700 p-4">
        <canvas ref={canvasRef} className="w-full" style={{ display: 'block' }} />
      </div>
      <div className="text-xs text-gray-400 mt-2">
        MRI signal intensity distribution
      </div>
    </div>
  );
};

const MedicalImageConverter = () => {
  const [conversionType, setConversionType] = useState('t1-to-t2');
  const [inputImages, setinputImages] = useState([]);
  const [outputImages, setOutputImages] = useState([]);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('conversion'); // Main tabs: conversion or segmentation
  const [infoTab, setInfoTab] = useState('metrics'); // Bottom info tabs
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);
  const [targetImage, settargetImage] = useState(null);
  const [targetImageUrl, settargetImageUrl] = useState(null);
  
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
    const files = Array.from(event.target.files).filter(file => 
      file.type.startsWith('image/') || file.name.endsWith('.nii')
    );
    
    if (!files.length) return;
    
    setinputImages(files);
    setOutputImages([]);
    setPreviewUrls([]);
    setError(null);
    setCurrentImageIndex(0);
    
    // Create preview URLs for all files
    files.forEach(file => {
      const fileReader = new FileReader();
      fileReader.onload = () => {
        setPreviewUrls(prevUrls => [...prevUrls, fileReader.result]);
      };
      fileReader.readAsDataURL(file);
    });
  };
  
  const handleDragOver = (event) => {
    event.preventDefault();
  };
  
  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files).filter(file => 
      file.type.startsWith('image/') || file.name.endsWith('.nii')
    );
    
    if (files.length) {
      setinputImages(files);
      setOutputImages([]);
      setPreviewUrls([]);
      setError(null);
      setCurrentImageIndex(0);
      
      // Create preview URLs for all files
      files.forEach(file => {
        const fileReader = new FileReader();
        fileReader.onload = () => {
          setPreviewUrls(prevUrls => [...prevUrls, fileReader.result]);
        };
        fileReader.readAsDataURL(file);
      });
    }
  };

  const handlePrevImage = () => {
    setCurrentImageIndex(prev => (prev > 0 ? prev - 1 : prev));
  };

  const handleNextImage = () => {
    setCurrentImageIndex(prev => (prev < inputImages.length - 1 ? prev + 1 : prev));
  };
  
  const handleConversion = async () => {
    if (!inputImages.length) {
      setError('Please upload at least one image');
      return;
    }
  
    setIsLoading(true);
    setError(null);
    setOutputImages([]);
  
    try {
      const results = [];
      const allMetrics = [];

      // Process each file
      for (const file of inputImages) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('conversionType', conversionType);
        formData.append('imageFormat', imageFormat);

        const response = await fetch('http://localhost:5000/api/convert/', {
          method: 'POST',
          body: formData
        });
    
        const text = await response.text();
        
        if (!response.ok) {
          let msg;
          try {
            msg = JSON.parse(text).error;
          } catch {
            msg = text;
          }
          throw new Error(`Failed to convert ${file.name}: ${msg}`);
        }
    
        const data = JSON.parse(text);
        results.push(data.result);
        allMetrics.push(data.metrics);
      }

      setOutputImages(results);

      // Average the metrics across all processed images
      if (allMetrics.length > 0) {
        const avgMetrics = {
          ssim: allMetrics.reduce((acc, m) => acc + m.ssim, 0) / allMetrics.length,
          psnr: allMetrics.reduce((acc, m) => acc + m.psnr, 0) / allMetrics.length,
          lpips: allMetrics.reduce((acc, m) => acc + m.lpips, 0) / allMetrics.length,
        };
        setMetrics(avgMetrics);
      }
  
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

  const handletargetImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/') && !file.name.endsWith('.nii')) {
      setError('Please select a valid image file');
      return;
    }
    
    const fileReader = new FileReader();
    fileReader.onload = () => {
      settargetImageUrl(fileReader.result);
    };
    fileReader.readAsDataURL(file);
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
        
        {/* Main Tabs: Conversion vs Segmentation */}
        <div className="flex border-b border-gray-700 mb-6">
          <button
            className={`px-6 py-3 font-medium ${activeTab === 'conversion' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
            onClick={() => setActiveTab('conversion')}
          >
            Conversion
          </button>
          <button
            className={`px-6 py-3 font-medium ${activeTab === 'segmentation' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
            onClick={() => setActiveTab('segmentation')}
          >
            Segmentation
          </button>
        </div>

        {/* Conversion Panel */}
        {activeTab === 'conversion' && (
          <>
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
                  Select Files
                  <input 
                    type="file" 
                    className="hidden" 
                    accept=".jpg,.jpeg,.png,.nii"
                    multiple
                    onChange={handleFileUpload} 
                  />
                </label>
                
                <button 
                  className="flex items-center justify-center px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={handleConversion}
                  disabled={!inputImages.length || isLoading}
                >
                  {isLoading 
                    ? <RefreshCw size={18} className="mr-2 animate-spin" /> 
                    : <ArrowRight size={18} className="mr-2" />
                  }
                  Convert {inputImages.length > 0 ? `(${inputImages.length})` : ''}
                </button>
              </div>
            </div>
            
            {error && (
              <div className="mb-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 rounded text-red-300">
                {error}
              </div>
            )}
            
            <div className="flex flex-col gap-6">
              <div className="flex flex-col md:flex-row gap-6">
                <div 
                  className="flex-1 h-64 border-2 border-dashed border-gray-600 rounded flex items-center justify-center bg-gray-700"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  {previewUrls[currentImageIndex] ? (
                    <img 
                      src={previewUrls[currentImageIndex]} 
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
                  ) : outputImages[currentImageIndex] ? (
                    <img 
                      src={outputImages[currentImageIndex]} 
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

              {/* Image Navigation */}
              {inputImages.length > 0 && (
                <div className="flex items-center justify-center gap-4">
                  <button
                    className="p-2 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handlePrevImage}
                    disabled={currentImageIndex === 0}
                  >
                    <ChevronLeft size={24} />
                  </button>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-400">
                      Image {currentImageIndex + 1} of {inputImages.length}
                    </p>
                    <p className="text-sm font-medium truncate max-w-xs">
                      {inputImages[currentImageIndex]?.name}
                    </p>
                  </div>

                  <button
                    className="p-2 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handleNextImage}
                    disabled={currentImageIndex === inputImages.length - 1}
                  >
                    <ChevronRight size={24} />
                  </button>
                </div>
              )}
            </div>
          </>
        )}

        {/* Segmentation Panel */}
        {activeTab === 'segmentation' && (
          <div className="w-full max-w-4xl bg-gray-800 rounded-lg p-6 shadow-lg">
            <Segmentation />
          </div>
        )}
        
        {/* Always visible info tabs at the bottom */}
        <div className="mt-8">
          {/* Tab Headers */}
          <div className="flex border-b border-gray-700">
            <button
              className={`px-4 py-2 font-medium ${infoTab === 'metrics' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setInfoTab('metrics')}
            >
              Metrics
            </button>
            <button
              className={`px-4 py-2 font-medium ${infoTab === 'performance' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setInfoTab('performance')}
            >
              Performance
            </button>
            <button
              className={`px-4 py-2 font-medium ${infoTab === 'architecture' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setInfoTab('architecture')}
            >
              Architecture
            </button>
            <button
              className={`px-4 py-2 font-medium ${infoTab === 'histogram' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setInfoTab('histogram')}
            >
              Compare
            </button>
            <button
              className={`px-4 py-2 font-medium ${infoTab === 'details' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setInfoTab('details')}
            >
              Details
            </button>
            <button
              className={`px-4 py-2 font-medium ${infoTab === 'settings' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-300'}`}
              onClick={() => setInfoTab('settings')}
            >
              Settings
            </button>
          </div>
          
          {/* Tab Content */}
          <div className="p-4 bg-gray-700 rounded-b">
            {infoTab === 'metrics' && (
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
                  <p className="text-center text-2xl font-bold">
                    {metrics?.mse != null ? metrics.mse.toFixed(2) : '–'}
                  </p>
                  <p className="text-center text-xs text-gray-500">Mean Squared Error</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded">
                  <h3 className="text-center text-gray-400 mb-1">Model</h3>
                  <p className="text-center text-sm">Multi-Input U-Net</p>
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
                  <p className="text-center text-2xl font-bold">
                    {metrics?.time ? `${metrics.time.toFixed(2)}s` : '–'}
                  </p>
                  <p className="text-center text-xs text-gray-500">Conversion Duration</p>
                </div>
              </div>
            )}

            {infoTab === 'performance' && (
              <div className="text-gray-300 space-y-8">
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
              </div>
            )}

            {infoTab === 'architecture' && (
              <div className="text-gray-300 space-y-8">
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

            {infoTab === 'histogram' && (
              <div className="text-gray-300 space-y-12">
                <div>
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-lg font-semibold text-blue-300">
                      Compare Pixel Intensity
                    </h3>
                    <label className="flex items-center px-3 py-1.5 bg-blue-600 text-sm rounded cursor-pointer hover:bg-blue-700 transition">
                      <Upload size={14} className="mr-1.5" />
                      Upload Target
                      <input 
                        type="file" 
                        className="hidden" 
                        accept=".jpg,.jpeg,.png,.nii"
                        onChange={handletargetImageUpload} 
                      />
                    </label>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Current Output */}
                    <div className="flex flex-col items-center">
                      {outputImages[currentImageIndex] ? (
                        <PixelHistogram 
                          imageUrl={outputImages[currentImageIndex]}
                          label="Generated Output"
                          color="#10B981"
                        />
                      ) : (
                        <div className="w-full h-[300px] bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400">
                          <p>Convert an image to see distribution</p>
                        </div>
                      )}
                    </div>

                    {/* Reference Image */}
                    <div className="flex flex-col items-center">
                      {targetImageUrl ? (
                        <PixelHistogram 
                          imageUrl={targetImageUrl}
                          label="Target Image"
                          color="#3B82F6"
                        />
                      ) : (
                        <div className="w-full h-[300px] bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400">
                          <p>Upload a target image</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
                      
            {infoTab === 'details' && (
              <div className="text-gray-300 space-y-2">
                <p><span className="font-medium">Input Format:</span> {inputImages.length ? inputImages[0].type : "-"}</p>
                <p><span className="font-medium">Output Format:</span> JPEG</p>
                <p><span className="font-medium">Model Version:</span> v1.2.0</p>
                <p><span className="font-medium">Conversion Type:</span> {conversionOptions.find(opt => opt.value === conversionType)?.label || conversionType}</p>
              </div>
            )}
            
            {infoTab === 'settings' && (
              <div className="text-gray-300 space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Quality Level</label>
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