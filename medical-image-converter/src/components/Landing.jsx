import React from 'react';
import { Link } from 'react-router-dom';

const Landing = () => {
  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center px-4">
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
