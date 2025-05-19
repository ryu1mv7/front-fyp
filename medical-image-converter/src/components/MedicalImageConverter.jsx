import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, ArrowRight, RefreshCw, ChevronLeft, ChevronRight, User, Settings, Bookmark, History, LogOut } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import Segmentation from './Segmentation';
import IXISegmentation from './IXISegmentation';
import VolumeViewer from './VolumeViewer';
import { Sun, Moon } from 'lucide-react';

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
  }, [imageUrl, color, drawHistogram]);

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
  // const [targetImage, settargetImage] = useState(null);
  const [targetImageUrl, settargetImageUrl] = useState(null);
  
  const [metrics, setMetrics] = useState(null);
  const [imageFormat, setImageFormat] = useState('png');
  const [volumeSlices, setVolumeSlices] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  const toggleTheme = () => {
    const newTheme = isDarkMode ? 'light' : 'dark';
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
    document.body.classList.toggle('bg-gray-50', newTheme === 'light');
    document.body.classList.toggle('bg-gray-900', newTheme === 'dark');
    document.body.classList.toggle('text-gray-800', newTheme === 'light');
    document.body.classList.toggle('text-gray-100', newTheme === 'dark');
    localStorage.setItem('theme', newTheme);
    setIsDarkMode(!isDarkMode);
  };

  const truncateText = (text, maxLength) => {
    if (isDarkMode && text.length > maxLength) {
      return text.substring(0, maxLength) + '...';
    }
    return text;
  };

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
    setVolumeSlices([]); 
  
    try {
      const results = [];
      const allMetrics = [];
      let lastSlices = [];

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

        // Inject mid-slice preview if provided (for NIfTI)
        if (data.preview) {
          setPreviewUrls(prev => {
            const newPreviews = [...prev];
            newPreviews.push(data.preview);
            return newPreviews;
          });
// store preview in the correct index
        } else {
          const reader = new FileReader();
          const base64 = await new Promise(resolve => {
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(file);
          });
          setPreviewUrls(prev => {
            const newPreviews = [...prev];
            newPreviews.push(base64);
            return newPreviews;
          });
        }

        if (data.sliceUrls?.length) {
          lastSlices = data.sliceUrls;
        }
      }

      setOutputImages(results);

      setPreviewUrls(previewUrls);

      const fullUrls = lastSlices.map(url => `http://localhost:5000${url}`);
      
      setVolumeSlices(fullUrls);


      // Average the metrics across all processed images
      if (allMetrics.length > 0) {
        const avgMetrics = {
          ssim: allMetrics.reduce((acc, m) => acc + m.ssim, 0) / allMetrics.length,
          psnr: allMetrics.reduce((acc, m) => acc + m.psnr, 0) / allMetrics.length,
          lpips: allMetrics.reduce((acc, m) => acc + m.lpips, 0) / allMetrics.length,
        };
        setMetrics(avgMetrics);
        const historyEntry = {
          mode: 'conversion',
          conversionType,
          modelName: 'Multi-Input U-Net',
          outputImage: results[currentImageIndex], // or just data.result inside the loop
          metrics: allMetrics[currentImageIndex],
          timestamp: Date.now()
        };

        const currentHistory = JSON.parse(localStorage.getItem('conversionHistory') || '[]');
        localStorage.setItem('conversionHistory', JSON.stringify([historyEntry, ...currentHistory]));

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
    <div className="flex flex-col min-h-screen bg-gray-50 text-gray-800">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6 dark:bg-gray-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-semibold text-blue-600 dark:text-blue-400">
            {truncateText('Mirage. Your Multi-Modality Translation & Synthesis Platform.', 30)}
          </h1>
          <div className="flex items-center space-x-4">
            <button
              onClick={toggleTheme}
              className="flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-600"
            >
              {isDarkMode ? (
                <>
                  <Sun size={16} className="mr-2" />
                  {truncateText('Light Mode', 10)}
                </>
              ) : (
                <>
                  <Moon size={16} className="mr-2" />
                  {truncateText('Dark Mode', 10)}
                </>
              )}
            </button>
            <div className="relative">
              <button 
                className="flex items-center space-x-2 focus:outline-none"
                onClick={() => setShowDropdown(!showDropdown)}
              >
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                  <User size={18} />
                </div>
                <span className="text-sm font-medium">
                  {truncateText(currentUser?.email.split('@')[0], 8)}
                </span>
              </button>
              
              {showDropdown && (
                <div className="absolute right-0 mt-2 w-56 bg-white rounded-md shadow-lg py-1 z-10 border border-gray-100">
                  <div className="px-4 py-3 border-b border-gray-100">
                    <p className="text-sm font-medium">
                      {truncateText(currentUser?.email, 20)}
                    </p>
                  </div>
                  <button 
                    className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    onClick={() => navigate('/history')}
                  >
                    <History size={16} className="mr-3 text-gray-500" />
                    {truncateText('History', 10)}
                  </button>
                  <button 
                    className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    onClick={() => navigate('/bookmark')}
                  >
                    <Bookmark size={16} className="mr-3 text-gray-500" />
                    {truncateText('Bookmarks', 10)}
                  </button>
                  <button 
                    className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    onClick={() => navigate('/settings')}
                  >
                    <Settings size={16} className="mr-3 text-gray-500" />
                    {truncateText('Settings', 10)}
                  </button>
                  <div className="border-t border-gray-100">
                    <button 
                      className="flex items-center w-full px-4 py-2 text-sm text-red-600 hover:bg-gray-50"
                      onClick={handleLogout}
                    >
                      <LogOut size={16} className="mr-3" />
                      {truncateText('Sign out', 10)}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow py-8 px-4 bg-gray-50 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto bg-white rounded-xl shadow-sm overflow-hidden dark:bg-gray-800 dark:border dark:border-gray-700">
          {/* Main Tabs */}
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('conversion')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'conversion'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Image Conversion
              </button>
              <button
                onClick={() => setActiveTab('segmentation')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'segmentation'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Tumor Segmentation
              </button>
              <button
                onClick={() => setActiveTab('overlay')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'overlay'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Tissue Overlay
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'conversion' && (
              <>
                <div className="mb-8">
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-2">
                    MRI Modality Conversion
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300">
                    Transform between T1, T2, and PD MRI modalities using our AI model
                  </p>
                </div>

                {/* Conversion Controls */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">Conversion Type</label>
                    <select 
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:text-gray-100"
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

                  {/* Output Format */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">Output Format</label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:text-gray-100"
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

                  {/* File Upload + Convert Button */}
                  <div className="flex items-end space-x-3">
                    <label className="flex-1">
                      <span className="sr-only">Upload files</span>
                      <div className="w-full flex items-center justify-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-100 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <Upload size={16} className="mr-2" />
                        Select Files
                        <input 
                          type="file" 
                          className="hidden" 
                          accept=".jpg,.jpeg,.png,.nii"
                          multiple
                          onChange={handleFileUpload} 
                        />
                      </div>
                    </label>

                    <button 
                      className="flex-1 flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                      onClick={handleConversion}
                      disabled={!inputImages.length || isLoading}
                    >
                      {isLoading 
                        ? <RefreshCw size={16} className="mr-2 animate-spin" /> 
                        : <ArrowRight size={16} className="mr-2" />
                      }
                      Convert
                    </button>
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <div className="mb-6 p-4 bg-red-50 dark:bg-red-900 border-l-4 border-red-400 dark:border-red-600 text-red-700 dark:text-red-200">
                    <div className="flex">
                      <svg className="h-5 w-5 mr-3 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                      <p className="text-sm">{error}</p>
                    </div>
                  </div>
                )}

                {/* Image Preview Area */}
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Input Image */}
                    <div className="rounded-lg border border-gray-200 dark:border-gray-600 overflow-hidden">
                      <div className="bg-gray-50 dark:bg-gray-700 p-3 border-b border-gray-200 dark:border-gray-600">
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-200">Input Image</h3>
                      </div>
                      <div 
                        className="h-64 bg-gray-50 dark:bg-gray-800 flex items-center justify-center"
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                      >
                        {previewUrls[currentImageIndex] ? (
                          <img 
                            src={previewUrls[currentImageIndex]} 
                            alt="Input" 
                            className="max-w-full max-h-full object-contain" 
                          />
                        ) : outputImages[currentImageIndex] ? (
                          <img 
                            src={outputImages[currentImageIndex]} 
                            alt="Input (from output fallback)" 
                            className="max-w-full max-h-full object-contain opacity-70" 
                          />
                        ) : (
                          <div className="text-center p-4 text-gray-500 dark:text-gray-400">
                            <Upload size={32} className="mx-auto mb-3" />
                            <p>Drag and drop or select files</p>
                            <p className="text-xs mt-1">Supports: JPG, PNG, NIfTI</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Output Image */}
                    <div className="rounded-lg border border-gray-200 dark:border-gray-600 overflow-hidden">
                      <div className="bg-gray-50 dark:bg-gray-700 p-3 border-b border-gray-200 dark:border-gray-600">
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-200">Output Image</h3>
                      </div>
                      <div className="h-64 bg-gray-50 dark:bg-gray-800 flex items-center justify-center">
                        {isLoading ? (
                          <div className="text-center text-gray-500 dark:text-gray-400">
                            <RefreshCw size={32} className="mx-auto mb-3 animate-spin" />
                            <p>Processing image...</p>
                          </div>
                        ) : outputImages[currentImageIndex] ? (
                          conversionType === 'ixi-segmentation' && Array.isArray(outputImages[currentImageIndex]) ? (
                            <div className="grid grid-cols-2 gap-2">
                              {outputImages[currentImageIndex].map((url, idx) => (
                                <img
                                  key={idx}
                                  src={url}
                                  alt={`Tissue ${idx + 1}`}
                                  className="object-contain border border-gray-600 rounded"
                                />
                              ))}
                            </div>
                          ) : (
                            <img 
                              src={outputImages[currentImageIndex]} 
                              alt="Output" 
                              className="max-w-full max-h-full object-contain" 
                            />
                          )
                        ) : (
                          <div className="text-center p-4 text-gray-400 dark:text-gray-500">
                            <p>Converted image will appear here</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Image Navigation */}
                  {/* {inputImages.length > 0 && (
                    <div className="flex items-center justify-between bg-gray-50 p-3 rounded-lg border border-gray-200">
                      <button
                        className="p-2 rounded-md bg-white border border-gray-300 shadow-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handlePrevImage}
                        disabled={currentImageIndex === 0}
                      >
                        <ChevronLeft size={20} />
                      </button>
                      
                      <div className="text-center">
                        <p className="text-sm text-gray-700">
                          Image {currentImageIndex + 1} of {inputImages.length}
                        </p>
                        <p className="text-xs text-gray-500 truncate max-w-xs">
                          {inputImages[currentImageIndex]?.name}
                        </p>
                      </div>

                      <button
                        className="p-2 rounded-md bg-white border border-gray-300 shadow-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handleNextImage}
                        disabled={currentImageIndex === inputImages.length - 1}
                      >
                        <ChevronRight size={20} />
                      </button>
                    </div>
                  )} */}

                  {/* Volume Viewer */}
                  {/* {volumeSlices.length > 0 && (
                    <div className="mt-6 rounded-lg border border-gray-200 overflow-hidden">
                      <div className="bg-gray-50 p-3 border-b border-gray-200">
                        <h3 className="text-sm font-medium text-gray-700">3D Volume Viewer</h3>
                      </div>
                      <div className="p-4">
                        <VolumeViewer slices={volumeSlices} />
                      </div>
                    </div>
                  )} */}
                </div>
              </>
            )}

        {/* Segmentation Panel (BraTS Only) */}
        {activeTab === 'segmentation' && (
          <div className="w-full max-w-4xl bg-gray-800 rounded-lg p-6 shadow-lg">
            <h2 className="text-2xl font-bold text-white mb-4">
              Tumor & Lesion Segmentation (Multi-Modal MRI)
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              This feature detects and highlights potential brain tumors or lesion regions based on AI-driven segmentation across multiple MRI inputs (T1n, T1ce, T2).
            </p>
            <Segmentation />
          </div>
        )}

        {/* Overlay Panel (IXI Tissue Segmentation) */}
        {activeTab === 'overlay' && (
          <div className="w-full max-w-6xl bg-gray-800 rounded-lg p-6 shadow-lg">
            <h2 className="text-2xl font-bold text-white mb-4">
              Brain Tissue Segmentation (CSF / GM / WM)
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              Visualizes cerebrospinal fluid (CSF), gray matter (GM), white matter (WM), and other structures from a mid-slice of a T1-weighted MRI scan.
            </p>
            <IXISegmentation />
          </div>
        )}

          {/* Info Tabs */}
          <div className="border-t border-gray-200">
            <div className="flex overflow-x-auto">
              <button
                onClick={() => setInfoTab('metrics')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'metrics'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Quality Metrics
              </button>
              <button
                onClick={() => setInfoTab('performance')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'performance'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Performance
              </button>
              <button
                onClick={() => setInfoTab('architecture')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'architecture'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Architecture
              </button>
              <button
                onClick={() => setInfoTab('histogram')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'histogram'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Intensity Analysis
              </button>
              <button
                onClick={() => setInfoTab('slices')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'slices'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Slice Viewer
              </button>
              <button
                onClick={() => setInfoTab('details')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'details'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Details
              </button>
              <button
                onClick={() => setInfoTab('settings')}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                  infoTab === 'settings'
                    ? 'text-blue-600 border-b-2 border-blue-500'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Settings
              </button>
            </div>

            <div className="p-6">
              {infoTab === 'metrics' && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <p className="text-sm text-blue-600 font-medium">SSIM</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {metrics ? metrics.ssim.toFixed(4) : '–'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Structural Similarity Index</p>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <p className="text-sm text-blue-600 font-medium">PSNR</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {metrics ? `${metrics.psnr.toFixed(2)} dB` : '–'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Peak Signal-to-Noise Ratio</p>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <p className="text-sm text-blue-600 font-medium">LPIPS</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {metrics?.lpips != null ? metrics.lpips.toFixed(4) : '–'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Perceptual Similarity</p>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <p className="text-sm text-blue-600 font-medium">Model</p>
                    <p className="text-lg font-semibold text-gray-800 mt-1">Multi-Input U-Net</p>
                    <p className="text-xs text-gray-500 mt-1">Architecture in use</p>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <p className="text-sm text-blue-600 font-medium">Processing Time</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {metrics?.time ? `${metrics.time.toFixed(2)}s` : '–'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Avg. Conversion Duration</p>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <p className="text-sm text-blue-600 font-medium">Model Version</p>
                    <p className="text-lg font-semibold text-gray-800 mt-1">v1.2.0</p>
                    <p className="text-xs text-gray-500 mt-1">Deployed backend version</p>
                  </div>
                </div>
              )}

              {infoTab === 'performance' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-800">Model Benchmarking</h3>
                  <img 
                    src="/assets/Model_Comparison_Chart.png"
                    alt="Model Comparison"
                    className="w-full rounded-lg border border-gray-200"
                  />
                  <p className="text-sm text-gray-600">
                    Comparative chart showcasing SSIM and PSNR scores for evaluated models. Multi-Input U-Net delivers superior perceptual and structural fidelity, outperforming GAN variants in stability and precision across test datasets.
                  </p>
                </div>
              )}

              {infoTab === 'architecture' && (
                <div className="space-y-8">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Model Architecture</h3>
                    <img 
                      src="/assets/Model_Architecture_Diagram.png"
                      alt="U-Net Architecture"
                      className="w-full rounded-lg border border-gray-200"
                    />
                    <p className="text-sm text-gray-600 mt-2">
                      The multi-input U-Net receives three MRI modalities (T1n, T1ce, T2w) and jointly predicts T2-FLAIR and segmentation output, enforcing modality alignment and multi-scale attention across encoder-decoder stages.
                    </p>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Workflow Pipeline</h3>
                    <img 
                      src="/assets/Workflow_Diagram.png"
                      alt="Workflow Diagram"
                      className="w-full rounded-lg border border-gray-200"
                    />
                    <p className="text-sm text-gray-600 mt-2">
                      Step-by-step pipeline: input loading → preprocessing → model inference → output decoding → overlay rendering. Integrated into our full-stack system with Gradio, Torch, and NIfTI visualization support.
                    </p>
                  </div>
                </div>
              )}

              {infoTab === 'histogram' && (
                <div className="space-y-6">
                  <div className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-gray-800">Intensity Distribution Comparison</h3>
                    <label className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 cursor-pointer">
                      <Upload size={14} className="mr-1.5" />
                      Upload Reference
                      <input 
                        type="file" 
                        className="hidden" 
                        accept=".jpg,.jpeg,.png,.nii"
                        onChange={handletargetImageUpload} 
                      />
                    </label>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="text-sm font-medium text-gray-700 mb-3">Generated Output</h4>
                      {outputImages[currentImageIndex] ? (
                        <PixelHistogram 
                          imageUrl={outputImages[currentImageIndex]}
                          label=""
                          color="#3B82F6"
                        />
                      ) : (
                        <div className="h-64 flex items-center justify-center text-gray-400">
                          <p>Convert an image to see distribution</p>
                        </div>
                      )}
                    </div>

                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="text-sm font-medium text-gray-700 mb-3">Reference Image</h4>
                      {targetImageUrl ? (
                        <PixelHistogram 
                          imageUrl={targetImageUrl}
                          label=""
                          color="#10B981"
                        />
                      ) : (
                        <div className="h-64 flex items-center justify-center text-gray-400">
                          <p>Upload a reference image</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {infoTab === 'slices' && volumeSlices.length > 0 && (
                <div className="bg-white p-6 rounded-lg border border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">3D Volume Slice Viewer</h3>
                  
                  <VolumeViewer slices={volumeSlices} />

                  {inputImages.length > 1 && (
                    <>
                      <hr className="my-6 border-t border-gray-300" />
                      <div className="flex items-center justify-between bg-gray-50 p-3 rounded-lg border border-gray-200">
                        <button
                          className="p-2 rounded-md bg-white border border-gray-300 shadow-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                          onClick={handlePrevImage}
                          disabled={currentImageIndex === 0}
                        >
                          <ChevronLeft size={20} />
                        </button>

                        <div className="text-center">
                          <p className="text-sm text-gray-700">
                            Image {currentImageIndex + 1} of {inputImages.length}
                          </p>
                          <p className="text-xs text-gray-500 truncate max-w-xs">
                            {inputImages[currentImageIndex]?.name}
                          </p>
                        </div>

                        <button
                          className="p-2 rounded-md bg-white border border-gray-300 shadow-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                          onClick={handleNextImage}
                          disabled={currentImageIndex === inputImages.length - 1}
                        >
                          <ChevronRight size={20} />
                        </button>
                      </div>
                    </>
                  )}

                  {/* Optional: keyboard left/right navigation */}
                  <script dangerouslySetInnerHTML={{
                    __html: `
                      document.onkeydown = function(e) {
                        if (document.activeElement.tagName.toLowerCase() === 'input') return;
                        if (e.key === 'ArrowLeft') {
                          document.getElementById('nav-left')?.click();
                        } else if (e.key === 'ArrowRight') {
                          document.getElementById('nav-right')?.click();
                        }
                      }
                    `
                  }} />
                </div>
              )}


            {infoTab === 'slices' && volumeSlices.length === 0 && (
              <div className="text-center text-gray-500 py-12">
                <p>No volume data available. Convert a NIfTI image to see the slices.</p>
              </div>
            )}
    
            {infoTab === 'details' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-700 mb-1 font-medium">Input Format</p>
                  <p className="text-lg text-gray-800">
                    {inputImages.length ? inputImages[0].type : '–'}
                  </p>
                </div>

                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-700 mb-1 font-medium">Output Format</p>
                  <p className="text-lg text-gray-800">{imageFormat.toUpperCase()}</p>
                </div>

                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-700 mb-1 font-medium">Model Version</p>
                  <p className="text-lg text-gray-800">v1.2.0</p>
                </div>

                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-700 mb-1 font-medium">Conversion Type</p>
                  <p className="text-lg text-gray-800">
                    {conversionOptions.find(opt => opt.value === conversionType)?.label || conversionType}
                  </p>
                </div>
              </div>
            )}
            
            {infoTab === 'settings' && (
              <div className="space-y-6">
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Quality Level
                  </label>
                  <input 
                    type="range" 
                    min="1" 
                    max="100" 
                    defaultValue="90" 
                    className="w-full accent-blue-500"
                  />
                </div>

                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Inference Parameters
                  </label>
                  <div className="flex flex-col md:flex-row gap-4">
                    <label className="flex items-center text-sm text-gray-700">
                      <input 
                        type="checkbox" 
                        className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500" 
                        defaultChecked 
                      />
                      Higher Accuracy
                    </label>
                    <label className="flex items-center text-sm text-gray-700">
                      <input 
                        type="checkbox" 
                        className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
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
  </main>

      {/* Footer */}
      <footer className="bg-white py-4 px-6 border-t border-gray-200 dark:bg-gray-800 dark:border-gray-700">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {truncateText('© 2025 MDS16. Monash University. For research use only.', 30)}
          </p>
          <div className="mt-2 md:mt-0">
            <button className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
              {truncateText('Terms', 5)}
            </button>
            <span className="mx-2 text-gray-300 dark:text-gray-600">•</span>
            <button className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
              {truncateText('Privacy', 5)}
            </button>
            <span className="mx-2 text-gray-300 dark:text-gray-600">•</span>
            <button className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
              {truncateText('Help', 5)}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default MedicalImageConverter;