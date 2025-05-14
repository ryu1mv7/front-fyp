import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Link } from 'react-router-dom';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { resetPassword } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setMessage('');
      setError('');
      setLoading(true);
      
      await resetPassword(email);
      setMessage('Check your email inbox for further instructions');
    } catch (err) {
      setError('Failed to reset password: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="w-full max-w-md">
        <div className="bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">Reset Password</h2>
          
          {error && (
            <div className="bg-red-500 bg-opacity-20 border border-red-500 text-red-300 p-3 rounded mb-4">
              {error}
            </div>
          )}
          
          {message && (
            <div className="bg-green-500 bg-opacity-20 border border-green-500 text-green-300 p-3 rounded mb-4">
              {message}
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-gray-300 text-sm font-medium mb-2">
                Email Address
              </label>
              <input 
                type="email" 
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-3 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter your email"
              />
            </div>
            
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50 mb-4"
            >
              Reset Password
            </button>
          </form>
          
          <div className="flex justify-between text-sm mt-4">
            <Link to="/login" className="text-blue-400 hover:text-blue-300">
              Back to Login
            </Link>
            <Link to="/signup" className="text-blue-400 hover:text-blue-300">
              Create an account
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword; 