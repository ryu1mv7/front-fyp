import React, { useRef, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Music } from 'lucide-react';

const Landing = () => {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);

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

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);

  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center px-4 relative overflow-hidden">
      {/* Audio Player */}
      <audio ref={audioRef} src="/assets/custombgm.wav" loop preload="auto" />

      {/* Audio Controls */}
      <div className="absolute top-4 right-4 flex flex-col items-end gap-2 z-10 animate-fade-in">
        <button
          onClick={toggleMusic}
          className="flex items-center text-sm bg-gray-800 px-3 py-1.5 rounded hover:bg-gray-700 transition"
        >
          <Music className="w-4 h-4 mr-1" />
          {playing ? 'Pause BGM' : 'Play BGM'}
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={volume}
          onChange={handleVolumeChange}
          className="w-32 accent-blue-500"
        />
      </div>

      {/* Floating Panel */}
      <div className="max-w-2xl w-full text-center p-10 bg-gray-900/60 backdrop-blur-md rounded-xl shadow-lg border border-gray-700 animate-slide-up">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-6 bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 text-transparent bg-clip-text animate-text-glow">
          Welcome to the Medical Image Converter
        </h1>

        <p className="text-gray-300 mb-4 text-lg leading-relaxed">
          A multi-modal brain MRI synthesis and segmentation platform powered by deep learning.
        </p>

        <p className="text-[0.65rem] text-gray-500 italic mb-6">
          Custom background music composed, produced, and recorded by Jarel Gomes.
        </p>

        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <Link
            to="/login"
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded font-semibold text-white shadow-md transition-all hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-400 animate-pulse"
          >
            Sign In
          </Link>
          <Link
            to="/signup"
            className="bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded font-semibold text-white shadow-md transition-all hover:scale-105 focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            Create Account
          </Link>
        </div>
      </div>

      {/* Animation styles */}
      <style>{`
        @keyframes textGlow {
          0%, 100% { text-shadow: 0 0 10px #3b82f6, 0 0 20px #06b6d4; }
          50% { text-shadow: 0 0 20px #8b5cf6, 0 0 30px #06b6d4; }
        }
        .animate-text-glow {
          animation: textGlow 3s ease-in-out infinite alternate;
        }
        .animate-fade-in {
          animation: fadeIn 1.2s ease-out both;
        }
        .animate-slide-up {
          animation: slideUp 1.2s ease-out both;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default Landing;
