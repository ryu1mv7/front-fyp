import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';

const Landing = () => {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5); // Default volume 50%

  const toggleMusic = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (playing) {
      audio.pause();
    } else {
      audio.play().catch(() => {});
    }
    setPlaying(!playing);
  };

  const handleVolumeChange = (e) => {
    const vol = parseFloat(e.target.value);
    setVolume(vol);
    if (audioRef.current) {
      audioRef.current.volume = vol;
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center px-4 relative">
      {/* Audio Player */}
      <audio ref={audioRef} src="/assets/custombgm.wav" loop preload="auto" volume={volume} />

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col items-end gap-2 z-10">
        <button 
          onClick={toggleMusic}
          className="text-sm bg-gray-800 px-3 py-1 rounded hover:bg-gray-700"
        >
          {playing ? 'Pause BGM' : 'Play BGM'}
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={volume}
          onChange={handleVolumeChange}
          className="w-32"
        />
      </div>

      {/* Main Content */}
      <div className="max-w-2xl w-full text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-6">
          Welcome to the <span className="text-blue-400">Medical Image Converter</span>
        </h1>
        <p className="text-gray-400 mb-10 text-lg">
          A multi-modal brain MRI synthesis and segmentation tool powered by deep learning. 
          Sign in to get started or create a new account.
        </p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <Link
            to="/login"
            className="bg-blue-600 hover:bg-blue-700 transition px-6 py-3 rounded font-semibold text-white shadow-md focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            Sign In
          </Link>
          <Link
            to="/signup"
            className="bg-gray-700 hover:bg-gray-600 transition px-6 py-3 rounded font-semibold text-white shadow-md focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            Create Account
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Landing;